/**
 * @file quant_group_matmul_custom.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "quant_group_matmul_custom_tiling.h"

using AscendC::LocalTensor;
using AscendC::GlobalTensor;
using AscendC::MatmulType;
using AscendC::TPosition;

using aT = MatmulType<TPosition::GM, CubeFormat::ND, int8_t>;
using bT = MatmulType<TPosition::GM, CubeFormat::NZ, int8_t>;
using BiasT = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;
using cT = MatmulType<TPosition::GM, CubeFormat::ND, int32_t>;

using MT = matmul::MatmulImpl<aT, bT, cT, BiasT, CFG_MDL>;

constexpr uint32_t BROADCAST_DIM = 2;
constexpr uint64_t SYNC_AIV_TO_AIC = 3;
constexpr uint64_t SYNC_AIC_TO_AIV = 5;
constexpr uint32_t BUFFER_NUM = 2;

struct MNConfig {
    uint32_t m = 0;
    uint32_t baseM = 0;
    uint32_t baseN = 0;
    uint32_t mIdx = 0;
    uint32_t nIdx = 0;
    uint32_t blockDimM = 0;
    uint32_t blockDimN = 0;
    uint32_t singleM = 0;
    uint32_t singleN = 0;
    uint32_t offsetM = 0;
    uint64_t workSpaceOffset = 0;
};


template <typename T>
__aicore__ inline void DataCopyPad2D(const LocalTensor<T> dst, const GlobalTensor<T> src, uint32_t dim1, uint32_t dim0,
                                     uint32_t srcDim0) {
    AscendC::DataCopyExtParams params;
    params.blockCount = dim1;
    params.blockLen = dim0 * sizeof(T);
    params.srcStride = (srcDim0 - dim0) * sizeof(T);
    // 32: int32 -> float16, 为防止跨行数据进入同一32B block，提前每行按偶数block对齐
    params.dstStride = AscendC::Ceil(dim0 * sizeof(T), 32) % 2;

    AscendC::DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
    DataCopyPad(dst, src, params, padParams);
}

template <typename T>
__aicore__ inline void DataCopyPad2D(const GlobalTensor<T> dst, const LocalTensor<T> src, uint32_t dim1, uint32_t dim0,
                                     uint32_t srcDim0, uint32_t dstDim0) {
    AscendC::DataCopyExtParams params;
    params.blockCount = dim1;
    params.blockLen = dim0 * sizeof(T);
    // 32: ub访问粒度为32B
    params.srcStride = (srcDim0 - dim0) * sizeof(T) / 32;
    params.dstStride = (dstDim0 - dim0) * sizeof(T);
    DataCopyPad(dst, src, params);
}

class QuantGroupMatmul {
public:
    __aicore__ inline QuantGroupMatmul(MT &matmul) : mm(matmul) {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias,
        GM_ADDR groupList, GM_ADDR scale, GM_ADDR perTokenScale, GM_ADDR y, GM_ADDR workspace,
        QuantGroupMatmulCustomTilingData *tilingData, AscendC::TPipe *tPipeIn);
    __aicore__ inline void Process();
private:
    __aicore__ inline void InitUbBuffer();
    __aicore__ inline void MMCompute(uint32_t groupIdx, MNConfig& mnConfig);
    __aicore__ inline void VectorCompute(uint32_t groupIdx, MNConfig& mnConfig);
    __aicore__ inline void PostCompute();
    __aicore__ inline void ComputeDequantAndActivate(MNConfig& mnConfig, uint32_t curVecBaseM, uint32_t alignBaseN,
                                                     uint32_t curVecBaseN, uint32_t offsetM);
    __aicore__ inline void DataCopyScale(uint32_t curBaseN, uint32_t alignBaseN, uint64_t scaleOffset);
    __aicore__ inline void DataCopyPerTokenScaleAndBrcb(MNConfig& mnConfig, uint32_t curBaseM, uint32_t alignBaseN,
                                                        uint32_t offsetM);

private:
    MT& mm;
    GlobalTensor<int8_t> xGm;
    GlobalTensor<int8_t> weightGm;
    GlobalTensor<int32_t> biasGm;
    GlobalTensor<int32_t> mmOutGm;
    GlobalTensor<float> scaleGm;
    GlobalTensor<float> perTokenScaleGm;
    GlobalTensor<int64_t> groupListGm;
    GlobalTensor<half> yGm;
    // define the que
    AscendC::TQue<TPosition::VECIN, 1> vecInQueue;
    AscendC::TQue<TPosition::VECOUT, 1> vecOutQueue;
    AscendC::TQue<TPosition::VECIN, 1> scaleInQueue;
    AscendC::TQue<TPosition::VECIN, 1> perTokenScaleInQueue;
    AscendC::TBuf<TPosition::VECCALC> tmpBuff;
    LocalTensor<float> scaleInUb;
    LocalTensor<float> perTokenScaleInUb;
    LocalTensor<float> dequantMiddleResult;
    LocalTensor<uint8_t> sharedTmpLocal;
    LocalTensor<float> mulsResultLocal;
    LocalTensor<float> pertokenBrcbLocal;
    LocalTensor<float> actResultLocal;
    uint32_t subBlockIdx;
    uint32_t coreIdx;
    uint32_t cubeCount = 0;
    AscendC::TPipe *pipe;
    QuantGroupMatmulCustomTilingData *tiling;
};

__aicore__ inline void QuantGroupMatmul::Init(GM_ADDR x, GM_ADDR weight, GM_ADDR bias,
        GM_ADDR groupList, GM_ADDR scale, GM_ADDR perTokenScale, GM_ADDR y, GM_ADDR workspace,
        QuantGroupMatmulCustomTilingData *tilingData, AscendC::TPipe *tPipeIn)
{
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(x));
    weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(weight));
    biasGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(bias));  // unused
    mmOutGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(workspace));
    scaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scale));
    perTokenScaleGm.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(perTokenScale));
    groupListGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(groupList));
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(y));
    tiling = tilingData;
    subBlockIdx = AscendC::GetSubBlockIdx();
    coreIdx = AscendC::GetBlockIdx();
    if ASCEND_IS_AIV {
        coreIdx /= AscendC::GetTaskRation();
    }
    pipe = tPipeIn;
    InitUbBuffer();
}

__aicore__ inline void QuantGroupMatmul::InitUbBuffer()
{
    if ASCEND_IS_AIC {
        return;
    }
    pipe->InitBuffer(scaleInQueue, BUFFER_NUM, tiling->mmTilingData.baseN * sizeof(float));
    pipe->InitBuffer(perTokenScaleInQueue, BUFFER_NUM, tiling->mmTilingData.baseM * sizeof(float));
    pipe->InitBuffer(vecInQueue, BUFFER_NUM, tiling->ubCalSize * sizeof(cT::T));
    pipe->InitBuffer(vecOutQueue, BUFFER_NUM, tiling->ubCalSize * sizeof(half));
    pipe->InitBuffer(tmpBuff, tiling->ubRestBytes);
    uint32_t ubCalSizeFloat = tiling->ubCalSize * sizeof(float);
    // ub分配，依次划分中间结果
    dequantMiddleResult = tmpBuff.GetWithOffset<float>(tiling->ubCalSize, 0);
    pertokenBrcbLocal = tmpBuff.GetWithOffset<float>(tiling->ubCalSize, ubCalSizeFloat);
    // 2: 偏移两份，前面为反量化输出和pertoken scale
    mulsResultLocal = tmpBuff.GetWithOffset<float>(tiling->ubCalSize, 2 * ubCalSizeFloat);
    // 3: 再上面的基础上再偏移一份
    actResultLocal = tmpBuff.GetWithOffset<float>(tiling->ubCalSize, 3 * ubCalSizeFloat);
    // api需要的临时空间，复用中间结果的空间
    // 2: ub临时空间总共4份，高层api分配两份
    sharedTmpLocal = tmpBuff.GetWithOffset<uint8_t>(2 * ubCalSizeFloat, 2 * ubCalSizeFloat);
}

__aicore__ inline void QuantGroupMatmul::Process()
{
    MNConfig mnConfig;
    mnConfig.baseM = tiling->mmTilingData.baseM;
    mnConfig.baseN = tiling->mmTilingData.baseN;
    mnConfig.singleM = mnConfig.baseM;
    mnConfig.singleN = mnConfig.baseN;
    mnConfig.blockDimN = AscendC::Ceil(tiling->n, mnConfig.singleN);
    for (uint32_t groupIdx = 0, preCount = 0; groupIdx < tiling->groupNum; ++groupIdx) {
        uint32_t m = static_cast<int32_t>(groupListGm.GetValue(groupIdx));
        if (m <= 0) {
            continue;
        }
        mnConfig.m = static_cast<uint32_t>(m);
        mnConfig.blockDimM = AscendC::Ceil(mnConfig.m, mnConfig.singleM);
        uint32_t curCount = preCount + mnConfig.blockDimN * mnConfig.blockDimM;
        uint32_t curBlock = coreIdx >= preCount ? coreIdx : coreIdx + tiling->coreNum;

        while (curBlock < curCount) {
            mnConfig.mIdx = (curBlock - preCount) / mnConfig.blockDimN;
            mnConfig.nIdx = (curBlock - preCount) % mnConfig.blockDimN;
            MMCompute(groupIdx, mnConfig);
            VectorCompute(groupIdx, mnConfig);
            curBlock += tiling->coreNum;
        }
        preCount = curCount % tiling->coreNum;
        mnConfig.offsetM += mnConfig.m;
    }
    PostCompute();
}

__aicore__ inline void QuantGroupMatmul::MMCompute(uint32_t groupIdx, MNConfig& mnConfig)
{
    uint32_t tailN = mnConfig.nIdx * mnConfig.singleN;
    uint32_t curSingleN = mnConfig.singleN;
    if (mnConfig.nIdx == mnConfig.blockDimN - 1) {
        curSingleN = tiling->n - tailN;
    }
    uint32_t curSingleM = mnConfig.singleM;
    if (mnConfig.mIdx == mnConfig.blockDimM - 1) {
        curSingleM = mnConfig.m - mnConfig.mIdx * mnConfig.singleM;
    }
    uint64_t xOffset = (mnConfig.offsetM + mnConfig.mIdx * mnConfig.singleM) * tiling->k;
    uint64_t weightOffset = groupIdx * tiling->n * tiling->k + tailN * tiling->k;  // for no transpose nz weight
    mnConfig.workSpaceOffset = mnConfig.singleN * mnConfig.singleM * \
                                   (coreIdx + (cubeCount % tiling->parallNum) * tiling->coreNum);
    if ASCEND_IS_AIC {
        if (cubeCount >= tiling->parallNum) {
            AscendC::CrossCoreWaitFlag(SYNC_AIV_TO_AIC);
        }
        mm.SetOrgShape(mnConfig.m, tiling->n, tiling->k);
        mm.SetSingleShape(curSingleM, curSingleN, tiling->k);
        mm.SetTensorA(xGm[xOffset]);
        auto weightSlice = weightGm[weightOffset];
        if (mnConfig.blockDimM == 1) {
            weightSlice.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        }
        mm.SetTensorB(weightSlice);
        uint64_t worskspaceOffset = mnConfig.workSpaceOffset;
        while (mm.Iterate()) {
            mm.GetTensorC(mmOutGm[worskspaceOffset], 0, true);
            AscendC::CrossCoreSetFlag<2, PIPE_FIX>(SYNC_AIC_TO_AIV);  // 2: mode为2, group内同步
            worskspaceOffset += (mnConfig.baseM * mnConfig.baseN);
        }
    }
    cubeCount++;
}

__aicore__ inline void QuantGroupMatmul::VectorCompute(uint32_t groupIdx, MNConfig& mnConfig)
{
    if ASCEND_IS_AIC {
        return;
    }
    uint32_t curCubeSingleN = mnConfig.singleN;
    if (mnConfig.nIdx == mnConfig.blockDimN - 1) {
        curCubeSingleN = tiling->n - mnConfig.nIdx * mnConfig.singleN;
    }
    uint32_t curCubeSingleM = mnConfig.singleM;
    if (mnConfig.mIdx == mnConfig.blockDimM - 1) {
        curCubeSingleM = mnConfig.m - mnConfig.mIdx * mnConfig.singleM;
    }
    uint32_t vecBaseM = tiling->ubCalSize / (AscendC::Ceil(mnConfig.baseN, uint32_t(8)) * 8);  //  8: num int32_t in 32B ub block
    vecBaseM = vecBaseM < curCubeSingleM ? vecBaseM : curCubeSingleM;
    uint32_t curVecBaseN = mnConfig.baseN;
    uint64_t scaleOffset = groupIdx * tiling->n + mnConfig.nIdx * mnConfig.singleN;
    uint64_t outOffset = (mnConfig.offsetM + mnConfig.mIdx * mnConfig.singleM) * tiling->n + \
                            mnConfig.nIdx * mnConfig.singleN;
    uint32_t taskRation = AscendC::GetTaskRation();
    for (uint32_t offsetN = 0, vecCount = 0; offsetN < curCubeSingleN; offsetN += mnConfig.baseN) {
        if (unlikely(offsetN + mnConfig.baseN >= curCubeSingleN)) {
            curVecBaseN = curCubeSingleN - offsetN;
        }
        uint32_t alignBaseN = AscendC::Ceil(curVecBaseN, uint32_t(8)) * 8;  //  8: num int32_t in 32B ub block
        DataCopyScale(curVecBaseN, alignBaseN, scaleOffset + offsetN);
        uint32_t curVecBaseM = vecBaseM;
        uint64_t mmOutOffset = mnConfig.workSpaceOffset + offsetN * mnConfig.baseM;
        AscendC::CrossCoreWaitFlag(SYNC_AIC_TO_AIV);
        for (uint32_t offsetM = 0; offsetM < curCubeSingleM; offsetM += vecBaseM) {
            vecCount++;
            if (taskRation != 0 && vecCount % taskRation != subBlockIdx) {
                continue;
            }
            if (unlikely(offsetM + vecBaseM >= curCubeSingleM)) { 
                curVecBaseM = curCubeSingleM - offsetM; 
            }
            // 使用AscendDequant接口做perchannel反量化
            LocalTensor<cT::T> mmOutLocal = vecInQueue.AllocTensor<cT::T>();
            DataCopyPad2D(mmOutLocal, mmOutGm[mmOutOffset + offsetM * curVecBaseN],
                          curVecBaseM, curVecBaseN, curVecBaseN);
            vecInQueue.EnQue(mmOutLocal);
            ComputeDequantAndActivate(mnConfig, curVecBaseM, alignBaseN, curVecBaseN, offsetM);
            LocalTensor<half> yLocal = vecOutQueue.DeQue<half>();
            DataCopyPad2D(yGm[outOffset + offsetM * tiling->n + offsetN], yLocal,
                          curVecBaseM, curVecBaseN, alignBaseN, tiling->n);
            vecOutQueue.FreeTensor(yLocal);
        }
        scaleInQueue.FreeTensor(scaleInUb);
    }
    AscendC::CrossCoreSetFlag<2, PIPE_MTE2>(SYNC_AIV_TO_AIC);  // 2: mode为2, group内同步
}

__aicore__ inline void QuantGroupMatmul::PostCompute() {
    if ASCEND_IS_AIC {
        if (likely(cubeCount > 0)) {  // exits last cycle wait
            uint32_t loop = cubeCount < tiling->parallNum ? cubeCount : tiling->parallNum;
            for (int32_t idx = 0; idx < loop; ++idx) {
                AscendC::CrossCoreWaitFlag(SYNC_AIV_TO_AIC);
            }
        }
    }
}

__aicore__ inline void QuantGroupMatmul::ComputeDequantAndActivate(MNConfig& mnConfig, 
    uint32_t curVecBaseM, uint32_t alignBaseN, uint32_t curVecBaseN, uint32_t offsetM)
{
    DataCopyPerTokenScaleAndBrcb(mnConfig, curVecBaseM, alignBaseN, offsetM);
    LocalTensor<int32_t> mmOutInUb = vecInQueue.DeQue<cT::T>();

    AscendDequant(dequantMiddleResult, mmOutInUb, scaleInUb, sharedTmpLocal, {curVecBaseM, alignBaseN, curVecBaseN});
    AscendC::PipeBarrier<PIPE_V>();
    vecInQueue.FreeTensor(mmOutInUb);
    // pertoken反量化
    uint32_t computeSize = curVecBaseM * alignBaseN;
    Mul(mulsResultLocal, dequantMiddleResult, pertokenBrcbLocal, computeSize);
    AscendC::PipeBarrier<PIPE_V>();

    // 2: 分配两份大小给高层api作为临时空间
    LocalTensor<uint8_t> actTmpLocal = tmpBuff.GetWithOffset<uint8_t>(2 * tiling->ubCalSize * sizeof(float), 0);
    FasterGelu(actResultLocal, mulsResultLocal, actTmpLocal, computeSize);
    AscendC::PipeBarrier<PIPE_V>();
    LocalTensor<half> yLocalInUb = vecOutQueue.AllocTensor<half>();
    // Cast后获得最终输出
    Cast(yLocalInUb, actResultLocal, AscendC::RoundMode::CAST_NONE, computeSize);
    AscendC::PipeBarrier<PIPE_V>();
    vecOutQueue.EnQue(yLocalInUb);
}

__aicore__ inline void QuantGroupMatmul::DataCopyScale(uint32_t curBaseN, uint32_t alignBaseN, uint64_t scaleOffset)
{
    // GM拷贝scale
    AscendC::DataCopyPadExtParams<float> padParams;
    AscendC::DataCopyExtParams scaleParams{1, static_cast<uint32_t>(curBaseN * sizeof(float)), 1, 1, 0};
    LocalTensor<float> scaleLocal = scaleInQueue.AllocTensor<float>();
    DataCopyPad(scaleLocal, scaleGm[scaleOffset], scaleParams, padParams);
    scaleInQueue.EnQue(scaleLocal);

    scaleInUb = scaleInQueue.DeQue<float>();
    scaleInUb.SetSize(alignBaseN);
}

__aicore__ inline void QuantGroupMatmul::DataCopyPerTokenScaleAndBrcb(MNConfig& mnConfig,
        uint32_t curBaseM, uint32_t alignBaseN, uint32_t offsetM)
{
    uint64_t perTokenScaleOffset = mnConfig.offsetM + mnConfig.mIdx * mnConfig.singleM + offsetM;
    // GM拷贝per token scale
    AscendC::DataCopyPadExtParams<float> padParams;
    AscendC::DataCopyExtParams perTokenScaleParams{1, static_cast<uint32_t>(curBaseM * sizeof(float)), 0, 0, 0};
    LocalTensor<float> perTokenScaleLocal = perTokenScaleInQueue.AllocTensor<float>();
    DataCopyPad(perTokenScaleLocal, perTokenScaleGm[perTokenScaleOffset], perTokenScaleParams, padParams);
    perTokenScaleInQueue.EnQue(perTokenScaleLocal);

    perTokenScaleInUb = perTokenScaleInQueue.DeQue<float>();
    const uint32_t broadCastDst[BROADCAST_DIM] = {curBaseM, alignBaseN};
    const uint32_t broadCastSrc[BROADCAST_DIM] = {curBaseM, 1};
    AscendC::BroadCast<float, BROADCAST_DIM, 1>(pertokenBrcbLocal, perTokenScaleInUb, broadCastDst, broadCastSrc,
                                                sharedTmpLocal);
    perTokenScaleInQueue.FreeTensor(perTokenScaleInUb);
}

__aicore__ inline void CopyTiling(QuantGroupMatmulCustomTilingData *tiling, GM_ADDR tilingGM)
{
    uint32_t *ptr = reinterpret_cast<uint32_t *>(tiling);
    auto tiling32 = reinterpret_cast<__gm__ uint32_t *>(tilingGM);

    for (uint32_t i = 0; i < sizeof(QuantGroupMatmulCustomTilingData) / sizeof(uint32_t); i++, ptr++) {
        *ptr = *(tiling32 + i);
    }
    return;
}

extern "C" __global__ __aicore__ void quant_group_matmul_custom(GM_ADDR x, GM_ADDR weight, GM_ADDR bias,
        GM_ADDR groupList, GM_ADDR scale, GM_ADDR perTokenScale, GM_ADDR y, GM_ADDR workspace, GM_ADDR tilingGm)
{
    AscendC::TPipe tPipe;
    QuantGroupMatmulCustomTilingData tilingData;
    CopyTiling(&tilingData, tilingGm);
    MT mm;
    if ASCEND_IS_AIC {
        mm.SetSubBlockIdx(0);
        mm.Init(&tilingData.mmTilingData, &tPipe);
    }
    QuantGroupMatmul op(mm);
    op.Init(x, weight, bias, groupList, scale, perTokenScale, y, workspace, &tilingData, &tPipe);
    op.Process();
}