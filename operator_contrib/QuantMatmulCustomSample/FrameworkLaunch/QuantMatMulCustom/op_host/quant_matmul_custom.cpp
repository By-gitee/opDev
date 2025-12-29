#include "quant_matmul_custom_tiling.h"

namespace optiling {
static std::map<ge::DataType, matmul_tiling::DataType> DTYPE_MAP =
{
    {ge::DT_FLOAT, matmul_tiling::DataType::DT_FLOAT},
    {ge::DT_FLOAT16, matmul_tiling::DataType::DT_FLOAT16},
    {ge::DT_UINT8, matmul_tiling::DataType::DT_UINT8},
};

static uint32_t GetGeDataTypeSize(ge::DataType dataType)
{
    static std::map<ge::DataType, double> typeSize = {
        {ge::DT_FLOAT,                 sizeof(float)},
        {ge::DT_FLOAT16,               sizeof(uint16_t)},
        {ge::DT_UINT8,                 sizeof(uint8_t)},
    };
    if (typeSize.find(dataType) == typeSize.end()) {
        return 0;
    }
    return typeSize[dataType];
}

uint32_t GetMatMulCommArgs(gert::TilingContext* context, MatMulArgs &commonArgs)
{
    constexpr int64_t DIM_OFFSET_ONE = 1;
    constexpr int64_t DIM_OFFSET_TWO = 2;

    commonArgs.aType = context->GetInputDesc(0)->GetDataType() == ge::DT_FLOAT ? ge::DT_FLOAT16 : context->GetInputDesc(0)->GetDataType();
    commonArgs.bType = context->GetInputDesc(1)->GetDataType();
    commonArgs.cType = context->GetOutputDesc(0)->GetDataType();
    printf("DoOpTiling GetMatMulCommArgs aType[%d], bType[%d], cType[%d]\n", commonArgs.aType, commonArgs.bType, commonArgs.cType);

    auto shapeA = context->GetInputShape(0)->GetStorageShape();
    int64_t dimsASize = shapeA.GetDimNum();

    commonArgs.origM = shapeA.GetDim(dimsASize - DIM_OFFSET_TWO);
    commonArgs.origK = shapeA.GetDim(dimsASize - DIM_OFFSET_ONE);
    commonArgs.origBatchA = 1;
    for (int32_t index = 0; index < dimsASize - DIM_OFFSET_TWO; index++) {
        commonArgs.origBatchA *= shapeA.GetDim(index);
    }

    std::vector<int64_t> shapeB;
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    const auto* negative_slope4 = attrs->GetListInt(0);
    if (negative_slope4 == nullptr) {
        printf("get attr negative_slope4 failed");
    } else {
        size_t size = negative_slope4->GetSize();
        for (size_t i = 0; i < size; i++) {
            shapeB.push_back(*(negative_slope4->GetData() + i));
        }
    }
    int64_t dimsBSize = shapeB.size();

    commonArgs.origN = shapeB.at(dimsBSize - DIM_OFFSET_ONE);
    commonArgs.origBatchB = 1;
    for (int32_t index = 0; index < dimsBSize - DIM_OFFSET_TWO; index++) {
        commonArgs.origBatchB *= shapeB.at(index);
    }

    printf("DoOpTiling GetMatMulCommArgs dimsASize[%ld], dimsBSize[%ld] \n", dimsASize, dimsBSize);
    printf("DoOpTiling GetMatMulCommArgs origM[%lu], origN[%lu], origK[%lu], origBatchA[%lu], origBatchB[%lu]\n", commonArgs.origM, commonArgs.origN, commonArgs.origK, commonArgs.origBatchA, commonArgs.origBatchB);

    return ge::GRAPH_SUCCESS;
}

#define GEMV_A_NORM_B_NORM_2BIT 1000
#define GEMV_A_NORM_B_NORM_4BIT 1001
#define GEMM_A_NORM_B_NORM_2BIT 1010
#define GEMM_A_NORM_B_NORM_4BIT 1011

uint32_t GetTilingKey(gert::TilingContext* context, MatMulArgs commonArgs)
{
    uint32_t bCount = commonArgs.origN * commonArgs.origK;
    uint32_t bSize = context->GetInputTensor(1)->GetSize();
    constexpr uint8_t uint8BitCount = 8;
    uint32_t bitPerNum = bSize * uint8BitCount / bCount;
    printf("try get correct bit per num of B, bSize=%u, bCount=%u\n", bSize, bCount);
    if (commonArgs.origM == 1) {
        if (bitPerNum == 2) {
            return GEMV_A_NORM_B_NORM_2BIT;
        }
        if (bitPerNum == 4) {
            return GEMV_A_NORM_B_NORM_4BIT;
        }
    } else {
        if (bitPerNum == 2) {
            return GEMM_A_NORM_B_NORM_2BIT;
        }
        if (bitPerNum == 4) {
            return GEMM_A_NORM_B_NORM_4BIT;
        }
    }
    printf("can not get correct bit per num of B, bitPerNum = %u\n", bitPerNum);
    return UINT32_MAX;
}

void PrintCustomTiling(QuantMatmulTilingData &tilingData)
{
    printf("tiling.customTilingData.tileN                   = %u \n", tilingData.customTilingData.get_tileN());
    printf("tiling.customTilingData.tileK                   = %u \n", tilingData.customTilingData.get_tileK());
    printf("tiling.customTilingData.blockA                  = %u \n", tilingData.customTilingData.get_blockA());
    printf("tiling.customTilingData.blockM                  = %u \n", tilingData.customTilingData.get_blockM());
    printf("tiling.customTilingData.blockKa                 = %u \n", tilingData.customTilingData.get_blockKa());
    printf("tiling.customTilingData.blockKb                 = %u \n", tilingData.customTilingData.get_blockKb());
    printf("tiling.customTilingData.blockN                  = %u \n", tilingData.customTilingData.get_blockN());

    printf("tiling.customTilingData.offsetK                 = %u \n", tilingData.customTilingData.get_offsetK());
    printf("tiling.customTilingData.offsetN                 = %u \n", tilingData.customTilingData.get_offsetN());
    printf("tiling.customTilingData.weightK                 = %u \n", tilingData.customTilingData.get_weightK());
    printf("tiling.customTilingData.weightN                 = %u \n", tilingData.customTilingData.get_weightN());

    printf("tiling.customTilingData.blockWeightK            = %u \n", tilingData.customTilingData.get_blockWeightK());
    printf("tiling.customTilingData.blockWeightN            = %u \n", tilingData.customTilingData.get_blockWeightN());
    printf("tiling.customTilingData.blockWeight             = %u \n", tilingData.customTilingData.get_blockWeight());
    printf("tiling.customTilingData.blockOffsetK            = %u \n", tilingData.customTilingData.get_blockOffsetK());
    printf("tiling.customTilingData.blockOffsetN            = %u \n", tilingData.customTilingData.get_blockOffsetN());
    printf("tiling.customTilingData.blockOffset             = %u \n", tilingData.customTilingData.get_blockOffset());
    printf("tiling.customTilingData.blockWeightAndOffset    = %u \n", tilingData.customTilingData.get_blockWeightAndOffset());

    printf("tiling.customTilingData.sizeBlockAntiqIn        = %u \n", tilingData.customTilingData.get_sizeBlockAntiqIn());
    printf("tiling.customTilingData.sizeBlockAntiqOut       = %u \n", tilingData.customTilingData.get_sizeBlockAntiqOut());
    printf("tiling.customTilingData.sizeAExpand             = %u \n", tilingData.customTilingData.get_sizeAExpand());
}

// 在basse matmul tiling基础上，做quant matmul的定制化处理
uint32_t DoQuantMatmulCustomTiling(uint32_t tilingKey, uint64_t l1Size, const MatMulArgs &commonArgs, QuantMatmulTilingData &tilingData)
{
    //定制算子group固定为32
    constexpr uint32_t groupNum = 32;

    //由于weight/offset/scale的TQue合并，weight将被视为half类型（2Bytes），存储数据量为uint8_t(1Bytes)的2倍
    uint32_t actNum = 2 * 2;
    if (tilingKey == GEMM_A_NORM_B_NORM_2BIT || tilingKey == GEMV_A_NORM_B_NORM_2BIT) {
        actNum = 4 * 2;
    }

    TCubeTiling &tiling = tilingData.cubeTilingData;
    // depthB1中的db将被视为Kb方向db，需要满足db*stepKb*baseK为Kb的因数
    uint32_t depthB1Db = tiling.get_depthB1() / tiling.get_stepN() / tiling.get_stepKb();
    uint32_t realStepKb = tiling.get_stepKb() * depthB1Db;
    // M,Ka 暂不支持L1层切分，满载于L1上
    uint32_t blockM = tiling.get_M();
    uint32_t blockKb = tiling.get_baseK() * realStepKb;
    uint32_t blockN = tiling.get_baseN() * tiling.get_stepN();
    // 由于A/B矩阵在Matmul高阶API入口时已位于L1上，Matmul高阶API会认为singleCoreM/N/K已全载于L1，singleCoreM/N/K需重新计算
    tilingData.cubeTilingData.set_singleCoreM(blockM);
    tilingData.cubeTilingData.set_singleCoreK(blockKb);
    tilingData.cubeTilingData.set_singleCoreN(blockN);
    printf("DoQuantMatmulCustomTiling tilingKey[%u], actNum[%u], singleCoreM[%u], singleCoreK[%u], singleCoreN[%u]\n", tilingKey, actNum, blockM, blockKb, blockN);

    // 算子层tiling参数
    uint32_t tileN = tiling.get_N() / blockN;
    uint32_t tileK = tiling.get_Kb() / blockKb;
    uint32_t blockA = blockM * blockKb;
    tilingData.customTilingData.set_tileN(tileN);
    tilingData.customTilingData.set_tileK(tileK);
    tilingData.customTilingData.set_blockA(blockA);
    tilingData.customTilingData.set_blockM(blockM);
    tilingData.customTilingData.set_blockKa(tiling.get_Ka());
    tilingData.customTilingData.set_blockKb(blockKb);
    tilingData.customTilingData.set_blockN(blockN);

    // Antiq完整数据
    uint32_t weightK = tiling.get_Kb();
    uint32_t weightN = tiling.get_N() / actNum;
    uint32_t offsetK = tiling.get_Kb() / groupNum;
    uint32_t offsetN = tiling.get_N();
    tilingData.customTilingData.set_offsetK(offsetK);
    tilingData.customTilingData.set_offsetN(offsetN);
    tilingData.customTilingData.set_weightK(weightK);
    tilingData.customTilingData.set_weightN(weightN);

    //Antiq切分数据
    uint32_t blockWeightK = blockKb;
    uint32_t blockWeightN = blockN / actNum;
    uint32_t blockOffsetK = blockKb / groupNum;
    uint32_t blockOffsetN = blockN;
    uint32_t blockWeight = blockWeightK * blockWeightN;
    uint32_t blockOffset = blockOffsetK * blockOffsetN;
    uint32_t blockWeightAndOffset = blockWeight + blockOffset;
    tilingData.customTilingData.set_blockWeightK(blockWeightK);
    tilingData.customTilingData.set_blockWeightN(blockWeightN);
    tilingData.customTilingData.set_blockWeight(blockWeight);
    tilingData.customTilingData.set_blockOffsetK(blockOffsetK);
    tilingData.customTilingData.set_blockOffsetN(blockOffsetN);
    tilingData.customTilingData.set_blockOffset(blockOffset);
    tilingData.customTilingData.set_blockWeightAndOffset(blockWeightAndOffset);

    uint32_t aTypeSize = GetGeDataTypeSize(commonArgs.aType);
    if (aTypeSize == 0) {
        return ge::GRAPH_PARAM_INVALID;
    }
    printf("DoQuantMatmulCustomTiling aTypeSize[%u]\n", aTypeSize);

    uint32_t sizeA = tiling.get_M() * tiling.get_Ka() * aTypeSize;
    uint32_t sizeBlockAntiqIn = (blockWeight + 2 * blockOffset) * aTypeSize;
    uint32_t sizeBlockAntiqOut = blockKb * blockN * aTypeSize;
    uint32_t sizeAExpand = l1Size / 2 - sizeBlockAntiqOut;
    printf("DoQuantMatmulCustomTiling sizeA[%u], sizeAExpand[%u], l1Size[%lu]\n", sizeA, sizeAExpand, l1Size);

    sizeAExpand = sizeA > sizeAExpand ? sizeA : sizeAExpand;
    tilingData.customTilingData.set_sizeBlockAntiqIn(sizeBlockAntiqIn);
    tilingData.customTilingData.set_sizeBlockAntiqOut(sizeBlockAntiqOut);
    tilingData.customTilingData.set_sizeAExpand(sizeAExpand);

    PrintCustomTiling(tilingData);

    return ge::GRAPH_SUCCESS;
}


uint32_t DoOpTiling(gert::TilingContext* context, QuantMatmulTilingData &tilingData)
{
    printf("start DoOpTiling !!!\n");

    uint64_t l1Size;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1Size);
    uint32_t blockDim = ascendcPlatform.GetCoreNumAic();

    MatMulArgs commonArgs;
    GetMatMulCommArgs(context, commonArgs);

    matmul_tiling::MultiCoreMatmulTiling cubeTiling(ascendcPlatform);

    cubeTiling.SetDim(blockDim);
    optiling::QuantMatMulTiling basicTiling(ascendcPlatform, commonArgs);

    if (basicTiling.GetTiling(tilingData.cubeTilingData) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    uint32_t tilingKey = GetTilingKey(context, commonArgs);
    if (tilingKey == UINT32_MAX) {
        return ge::GRAPH_PARAM_INVALID;
    }

    uint32_t ret = DoQuantMatmulCustomTiling(tilingKey, l1Size, commonArgs, tilingData);
    if (ret != ge::GRAPH_SUCCESS) {
        printf("DoQuantMatmulCustomTiling failed ret[%d].\n", ret);
        return ret;
    }

    context->SetTilingKey(tilingKey);
    context->SetBlockDim(blockDim);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    QuantMatmulTilingData tiling;
    if (DoOpTiling(context, tiling) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;

    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    const gert::RuntimeAttrs* attrs = context->GetAttrs();
    const auto* negative_slope4 = attrs->GetListInt(0);
    if (negative_slope4 == nullptr) {
        printf("get attr negative_slope4 failed");
    } else {
        size_t size = negative_slope4->GetSize();
        y_shape->SetDimNum(size);
        for (size_t i = 0; i < size; i++) {
            y_shape->SetDim(i, *(negative_slope4->GetData() + i));
        }
    }

    std::vector<int64_t> shapeValues;
    shapeValues.reserve(x1_shape->GetDimNum() + y_shape->GetDimNum());

    for (size_t i = 0; i < x1_shape->GetDimNum(); ++i) {
        shapeValues.push_back(x1_shape->GetDim(i));
    }

    for (size_t i = 0; i < y_shape->GetDimNum(); ++i) {
        shapeValues.push_back(y_shape->GetDim(i));
    }

    if (!shapeValues.empty()) {
        int64_t firstValue = shapeValues[0];
        int64_t fourthValue = shapeValues[3];

        *y_shape = gert::Shape({firstValue, fourthValue});
    }
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext* context)
{
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return GRAPH_SUCCESS;
}
}


namespace ops {
class QuantMatmulCustom : public OpDef {
public:
    explicit QuantMatmulCustom(const char* name) : OpDef(name)
    {
        this->Input("a")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_NCHW});
        this->Input("b")
            .ParamType(REQUIRED)
            .DataType({ge::DT_UINT8})
            .Format({ge::FORMAT_NCHW});
        this->Input("scale")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_NCHW});
        this->Input("offset")
            .ParamType(OPTIONAL)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_NCHW});
        this->Output("c")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_NCHW});
        this->Attr("input1_shape").AttrType(OPTIONAL).ListInt({1, 1});
        this->Attr("transformer_nd").AttrType(OPTIONAL).Bool(true);

        this->SetInferShape(ge::InferShape);
        this->SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("kirin9020");

    }
};

OP_ADD(QuantMatmulCustom);
}
