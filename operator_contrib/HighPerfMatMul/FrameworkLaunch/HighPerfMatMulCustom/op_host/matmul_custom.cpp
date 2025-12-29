/**
 * @file matmul_custom.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "matmul_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "graph/utils/type_utils.h"
using namespace matmul_tiling;

namespace optiling {
    // 定义L2SplitParams结构体，用于保存和计算L2切分的参数
    struct L2SplitParams
    {
        uint64_t outBase = 0;        // 切分L2时，外轴基本块长度
        uint64_t innerBase = 0;      // 切分L2时，内轴基本块长度
        uint64_t outValue = 0;       // 切分L2时，外轴数据量
        uint64_t innerValue = 0;     // 切分L2时，内轴数据量
        uint64_t outDtypeSize = 0;   // 切分L2时，外轴数据类型大小
        uint64_t innerDtypeSize = 0; // 切分L2时，内轴数据类型大小
        /*假设从L2读取数据时按512对齐，那么当L2cache中缓存数据按核依次分配时，
          假设核1和核2读取的数据地址偏移对齐512时相同，则它们无法同时读取，此时认为核1和核2有同地址冲突*/
        uint64_t maxConflictDim = 0; // 切分L2时，外轴最大冲突核数
        uint64_t minConflictDim = 0; // 切分L2时，内轴最小冲突核数
        uint64_t outTailCnt = 0;     // 切分L2时，外轴尾块个数
        uint64_t innerTailCnt = 0;   // 切分L2时，内轴尾块个数
    };
    // 定义InputParams结构体，用于保存输入输出信息
    struct InputParams
    {
        uint64_t M;
        uint64_t N;
        uint64_t K;
        uint64_t baseM;
        uint64_t baseN;
        uint64_t baseK;
        bool isATrans;
        bool isBTrans;
        uint64_t dtypeASize;
        uint64_t dtypeBSize;
        uint64_t dtypeCSize;
        uint64_t aicNum;
        double aicRatio;
    };

    // 初始化L2SplitParams结构体
    void InitL2SplitParams(L2SplitParams &l2SplitParams, const InputParams params)
    {
        l2SplitParams.outBase = std::max(params.baseM, 1UL);   // 设置outBase为baseM
        l2SplitParams.innerBase = std::max(params.baseN, 1UL); // 设置innerBase为baseN
        l2SplitParams.outValue = params.M;                     // 设置外轴数据量为M
        l2SplitParams.innerValue = params.N;                   // 设置内轴数据量为N
        l2SplitParams.outDtypeSize = params.dtypeASize;        // 设置外轴数据类型大小为左矩阵数据类型大小
        l2SplitParams.innerDtypeSize = params.dtypeBSize;      // 设置内轴数据类型大小为右矩阵数据类型大小
        if (params.baseN >= params.baseM)
        { // 当baseN >= baseM时，交换外轴和内轴
            l2SplitParams.outBase = params.baseN;
            l2SplitParams.innerBase = params.baseM;
            l2SplitParams.outValue = params.N;
            l2SplitParams.innerValue = params.M;
            l2SplitParams.outDtypeSize = params.dtypeBSize;
            l2SplitParams.innerDtypeSize = params.dtypeASize;
        }

        l2SplitParams.maxConflictDim = 6; // 24核最多冲突6核
        l2SplitParams.minConflictDim = 3; // 24核内轴不亲和最多冲突3核
        const uint64_t CORE_NUM20 = 20; // 20核时冲突轴数需要重新设置
        if (params.aicNum == CORE_NUM20)
        {                                     // 针对20核的场景
            l2SplitParams.maxConflictDim = 5; // 20核最多冲突5核
            l2SplitParams.minConflictDim = 4; // 20核内轴不亲和最多冲突4核
        }
    }
    // 求a为b的向上取整倍数
    uint64_t CeilDivision(const uint64_t a, const uint64_t b)
    {
        if (b == 0)
        {
            return a;
        }
        return (a + b - 1) / b;
    }
    // 获取总数据量大小, 由于该样例输入输出类型一致，所以类型长度相同
    uint64_t GetTotalSize(const uint64_t m, const uint64_t k, const uint64_t n,
                          const uint64_t inputOutputTypeSize)
    {
        uint64_t sizeA = m * k * inputOutputTypeSize;
        uint64_t sizeB = k * n * inputOutputTypeSize;
        uint64_t sizeC = m * n * inputOutputTypeSize;
        return sizeA + sizeB + sizeC;
    }
    // num1向上对齐到num2的倍数
    int64_t Align(const int32_t num1, const int32_t num2)
    {
        return static_cast<int64_t>(CeilDivision(num1, num2)) * num2;
    }

    // 判断尾块放在不冲突的核上是否能错位处理，并更新L2切分时内外轴尾块个数
    bool IsTailSmall(L2SplitParams &l2SplitParams, const uint64_t outL2Split, const uint64_t innerL2Split,
                     const uint64_t innerMaxConflict, const uint64_t aicNum)
    {
        if (outL2Split == 0 || innerL2Split == 0)
        {
            return false;
        }
        uint64_t outTailValue = ((l2SplitParams.outValue + outL2Split - 1) % outL2Split) + 1;
        uint64_t innerTailValue = ((l2SplitParams.innerValue + innerL2Split - 1) % innerL2Split) + 1;
        l2SplitParams.outTailCnt = CeilDivision(outTailValue, l2SplitParams.outBase);
        l2SplitParams.innerTailCnt = CeilDivision(innerTailValue, l2SplitParams.innerBase);
        bool isOutTailSmall = l2SplitParams.outTailCnt * l2SplitParams.maxConflictDim < aicNum;
        bool isInnerTailSmall = l2SplitParams.innerTailCnt * innerMaxConflict < aicNum;
        return (isOutTailSmall || isInnerTailSmall);
    }
    /*
    循环遍历内外轴计算最优切分策略
    outTile: 外轴切分次数，默认为1次不切分
    innerTile: 内轴切分次数，默认为1次不切分
    outL2Split: 单核上外轴数据量，默认为外轴总大小
    innerL2Split: 单核上内轴数据量，默认为内轴总大小
    isATrans: 外轴为N时传入左矩阵是否转置
    M: M
    N: N
    K: K
    baseM: baseM
    baseN: baseN
    dtypeASize: 左矩阵数据类型
    dtypeBSize: 右矩阵数据类型
    aicNum: 实际使用核数
    l2Ratio: L2数据置换比例，实际使用大小 / L2大小超出该比例则L2数据会有数据置换
    */
    void CalcTile(uint64_t &outTile, uint64_t &innerTile, uint64_t &outL2Split, uint64_t &innerL2Split, const InputParams params)
    {
        L2SplitParams l2SplitParams;
        InitL2SplitParams(l2SplitParams, params);
        uint64_t innerMaxConflict = params.isATrans ? l2SplitParams.minConflictDim : l2SplitParams.maxConflictDim; // 根据数据是否转置获取对应冲突核数

        uint64_t outerMinUseDim = params.aicNum / l2SplitParams.maxConflictDim; // 避免地址冲突，外轴最少需要分的份数
        uint64_t innerMinUseDim = params.aicNum / innerMaxConflict;
        uint64_t outOriShape = outL2Split;     // 待切分外轴大小，该样例中N为外轴
        uint64_t innerOriShape = innerL2Split; // 待切分内轴大小，该样例中M为内轴
        uint64_t outConflict = 0;
        uint64_t innerConflict = 0;
        bool enableCache = false;

        // 通过遍历的方式获取最优L2切分
        for (uint64_t outerUseDim = params.aicNum; outerUseDim >= outerMinUseDim; outerUseDim--) // 外轴有多少核读取不相同的数据
        {
            for (uint64_t innerUseDim = params.aicNum; innerUseDim >= innerMinUseDim; innerUseDim--) // 内轴有多少核读取不相同的数据
            {
                uint64_t outTileTmp = std::max(outOriShape / (l2SplitParams.outBase * outerUseDim), 1UL);       // 外轴切分后块数
                uint64_t innerTileTmp = std::max(innerOriShape / (l2SplitParams.innerBase * innerUseDim), 1UL); // 内轴切分后块数
                uint64_t outL2SplitTmp =
                    Align(CeilDivision(l2SplitParams.outValue, outTileTmp), l2SplitParams.outBase); // 切分后每块外轴大小
                uint64_t innerL2SplitTmp = Align(CeilDivision(l2SplitParams.innerValue, innerTileTmp),
                                                 l2SplitParams.innerBase); // 切分后每块内轴数大小

                // 由于不切K，可以用外轴数据量、K、内轴数据量计算得到当前L2中应该缓存的数据量
                uint64_t totalSize = GetTotalSize(innerL2SplitTmp, params.K, outL2SplitTmp, l2SplitParams.outDtypeSize);

                constexpr uint64_t DATA_REPLACEMENT_THRESHOLD = 100 * 1024 * 1024; // 1MB = 1024kb, L2 为192MB时，数据置换临界值为100MB。L2为其他大小时比例不变
                if (totalSize <= params.aicRatio * DATA_REPLACEMENT_THRESHOLD)
                { // totalSize小于置换临界值，确保不会出现L2数据置换
                    // 满足L2数据不置换、数据不冲突时，根据当前临时切分策略刷新尾块个数
                    if (IsTailSmall(l2SplitParams, outL2SplitTmp, innerL2SplitTmp, innerMaxConflict, params.aicNum))
                    {
                        continue;
                    }
                    uint64_t outConflictTmp = CeilDivision(params.aicNum, l2SplitParams.outTailCnt);
                    uint64_t innerConflictTmp = CeilDivision(params.aicNum, l2SplitParams.innerTailCnt);
                    // 刷新一次切分数据后，如果后续策略有内外轴尾块都为0的，会再次刷新。
                    bool isUpdate = !enableCache || (outConflict >= outConflictTmp && innerConflict >= innerConflictTmp);
                    if (isUpdate)
                    {
                        enableCache = true;             // 获取最优切分后不再更改设置，仅设置一次最优参数
                        outTile = outTileTmp;           // 把外轴切分块数刷新到入参
                        innerTile = innerTileTmp;       // 把内轴切分块数刷新到入参
                        outL2Split = outL2SplitTmp;     // 把切分后每块外轴长度刷新到入参
                        innerL2Split = innerL2SplitTmp; // 把切分后每块内轴长度刷新到入参
                    }
                }
            }
        }
    }

    void SetupMultiCoreMatmulTiling(matmul_tiling::MultiCoreMatmulTiling &cubeTiling, 
                                    const platform_ascendc::PlatformAscendC &ascendcPlatform, 
                                    const InputParams params, const uint64_t l1Size)
    {
        uint64_t ubSize{0UL};
        uint64_t l0CSize{0UL};
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0CSize);
        cubeTiling.SetAType(matmul_tiling::TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT, false);
        cubeTiling.SetBType(matmul_tiling::TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT, false);
        cubeTiling.SetCType(matmul_tiling::TPosition::GM, CubeFormat::ND, matmul_tiling::DataType::DT_FLOAT);
        cubeTiling.SetDim(params.aicNum);
        cubeTiling.SetShape(params.M, params.N, params.K);
        cubeTiling.SetOrgShape(params.M, params.N, params.K);
        cubeTiling.SetBias(false);
        cubeTiling.SetBufferSpace(l1Size, l0CSize, ubSize);
        cubeTiling.SetFixSplit(params.baseM, params.baseN, -1);
        cubeTiling.SetSingleShape(params.baseM, params.baseN, -1);
    }
    // 根据L1计算缓存参数
    void SplitL1(MatmulCustomTilingData &tiling, const InputParams params, const uint64_t l1Size)
    {
        constexpr uint64_t NUM_HALF = 2;                                                             // 常量2, L1内存放A、B矩阵，每块占一半
        constexpr uint64_t DB_SIZE = 2;                                                              // double Buffer
        uint64_t totalL1Size = l1Size + 256;                                                         // GetCoreMemSize获取的大小是减去rpc通信的，这里不需要保留，所以加上
        uint64_t reserveBTSize = 0;                                                                  // bias大小，这里没使用bias，设置为0
        uint64_t depthA1 = totalL1Size / NUM_HALF / params.baseM / params.baseK / params.dtypeASize; // L1分两部分，一部分给左矩阵，全载左矩阵的块数
        uint64_t depthB1 = totalL1Size / NUM_HALF / params.baseN / params.baseK / params.dtypeBSize; // L1分两部分，一部分给右矩阵，全载右矩阵的块数
        uint64_t depthASize = depthA1 * params.baseM * params.baseK * params.dtypeASize;             // 全载左矩阵小块占用的大小
        uint64_t depthBSize = depthB1 * params.baseN * params.baseK * params.dtypeBSize;             // 全载右矩阵小块占用的大小
        if (depthASize + depthBSize > totalL1Size - reserveBTSize)
        { // 全载左右矩阵小块占用的大小如果超出L1 - bias占用大小，则缩减全载块数
            if (params.baseM <= params.baseN)
            {
                depthA1 = depthA1 / NUM_HALF; // 如果baseM较小，则缩减左矩阵全载块数
            }
            else
            {
                depthB1 = depthB1 / NUM_HALF; // 如果baseN较小，则缩减右矩阵全载块数
            }
        }
        uint64_t stepKa = depthA1 / DB_SIZE; // 左矩阵缓存，此处除2开启double Buffer
        uint64_t stepKb = depthB1 / DB_SIZE; // 右矩阵缓存，此处除2开启double Buffer
        if (stepKa >= stepKb)
        {
            stepKa = stepKa / stepKb * stepKb; // 当stepKa大于stepKb时重新计算stepKa
        }
        else
        {
            stepKb = stepKb / stepKa * stepKa; // 当stepKb大于stepKa时重新计算stepKb
        }

        tiling.cubeTilingData.set_depthA1(depthA1);
        tiling.cubeTilingData.set_depthB1(depthB1);
        tiling.cubeTilingData.set_stepKa(stepKa);
        tiling.cubeTilingData.set_stepKb(stepKb);
    }
    // 获取L2最佳切分策略并设置到tiling
    void SplitL2(MatmulCustomTilingData &tiling, const InputParams params)
    {
        uint64_t mTile = 1;           // 根据L2反推切分后M的块数
        uint64_t nTile = 1;           // 根据L2反推切分后N的块数
        uint64_t mL2Split = params.M; // 根据L2反推切分后M大小
        uint64_t nL2Split = params.N; // 根据L2反推切分后N大小
        // 由于baseM < baseN ,这里默认N为外轴
        CalcTile(nTile, mTile, nL2Split, mL2Split, params);
        uint32_t mTileBlock = (mL2Split + params.baseM - 1) / params.baseM; // 向上对齐单次切分运算的baseM个数
        uint32_t nTileBlock = (nL2Split + params.baseN - 1) / params.baseN; // 向上对齐单次切分运算的baseN个数
        mTile = CeilDivision(params.M, mTileBlock * params.baseM);
        nTile = CeilDivision(params.N, nTileBlock * params.baseN);
        tiling.set_mTileBlock(mTileBlock);
        tiling.set_nTileBlock(nTileBlock);
        tiling.set_mTile(mTile);
        tiling.set_nTile(nTile);
    }

static ge::graphStatus TilingFunc(gert::TilingContext *context)
{
    constexpr uint64_t SIZE_256 = 256; // 常量256
    constexpr uint64_t SIZE_128 = 128; // 常量128
    constexpr uint64_t BASE_K_128 = 128; // 默认basek大小为128, 然后根据实际数据类型计算实际的basek大小
    constexpr uint64_t L2_SIZE_BASE = 192 * 1024 * 1024; // 设置L2大小，非真实值
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    auto shapeA = context->GetInputTensor(0)->GetOriginShape();
    auto shapeB = context->GetInputTensor(1)->GetOriginShape();
    uint64_t M = shapeA.GetDim(0);
    uint64_t N = shapeB.GetDim(1);
    uint64_t K = shapeA.GetDim(1);
    uint64_t baseM = 1;
    uint64_t baseN = 1;
    uint64_t baseK = 1;
    uint64_t aicNum{0UL};
    uint64_t l1Size{0UL};
    uint64_t l2Size{0UL};
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, l2Size);
    aicNum = ascendcPlatform.GetCoreNumAic();
    uint32_t typeLength = 4; // 获取输入数据类型占用字节数，本样例输入输出均为float，所以默认为4
    ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength); // 通过API获取实际数据类型大小
    double l2Ratio = 1.0 * l2Size / L2_SIZE_BASE; // L2置换边界比例为100/192，用实际l2Size / L2_SIZE_BASE * 100MB可以获取当前设备置换边界临界值
    baseM = SIZE_128; // 设置默认baseM为128
    baseN = SIZE_256; // 设置默认baseN为256
    baseK = BASE_K_128 / typeLength; // 设置baseK为128 / 数据类型大小
    InputParams params{M, N, K, baseM, baseN, baseK, false, false, typeLength, typeLength, typeLength, aicNum, l2Ratio};
    MultiCoreMatmulTiling cubeTiling(ascendcPlatform);
    // 设置TCubeTiling
    SetupMultiCoreMatmulTiling(cubeTiling, ascendcPlatform, params, l1Size);
    MatmulCustomTilingData tiling;
    if (cubeTiling.GetTiling(tiling.cubeTilingData) == -1) {
        return ge::GRAPH_FAILED;
    }
    context->SetBlockDim(aicNum);
    uint64_t totalSize = GetTotalSize(M, K, N, typeLength);
    constexpr uint64_t DATA_REPLACEMENT_THRESHOLD = 100 * 1024 * 1024; // 100MB
    if (totalSize > l2Ratio * DATA_REPLACEMENT_THRESHOLD)
    {
       SplitL2(tiling, params); // 优化L2cache
    }
    SplitL1(tiling, params, l1Size); // 优化L1

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    context->SetScheduleMode(1);
    size_t userWorkspaceSize = 0;
    size_t systemWorkspaceSize = static_cast<size_t>(ascendcPlatform.GetLibApiWorkSpaceSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = userWorkspaceSize + systemWorkspaceSize;
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling

namespace ops {
class MatmulCustom : public OpDef {
public:
    explicit MatmulCustom(const char *name) : OpDef(name)
    {
        this->Input("a")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Input("b")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Input("bias")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});
        this->Output("c")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});

        this->AICore()
            .SetTiling(optiling::TilingFunc)
            .AddConfig("ascend910b");
    }
};

OP_ADD(MatmulCustom);
} // namespace ops
