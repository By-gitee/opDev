/**
 * @file add_custom_v1.h
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef ADD_CUSTOM_V1_H
#define ADD_CUSTOM_V1_H
#include "kernel_operator.h"
#include "add_custom_tiling.h"

using AscendC::TPosition;
class KernelAddV1 {
public:
    __aicore__ inline KernelAddV1() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, AddCustomTilingData *tilingPtr)
    {
        tiling = tilingPtr;
        xGm.SetGlobalBuffer((__gm__ float *)x + AscendC::GetBlockIdx() * TILE_N);
        yGm.SetGlobalBuffer((__gm__ float *)y + AscendC::GetBlockIdx() * TILE_N);
        zGm.SetGlobalBuffer((__gm__ float *)z + AscendC::GetBlockIdx() * TILE_N);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_M * TILE_N * sizeof(float));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, TILE_M * TILE_N * sizeof(float));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, TILE_M * TILE_N * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < tiling->loopOuter; i++) {
            for (uint32_t j = 0; j < M_A / TILE_M; j++) {
                CopyIn(i, j);
                Compute();
                CopyOut(i, j);
            }
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t progressOuter, uint32_t progressInner)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::LocalTensor<float> yLocal = inQueueY.AllocTensor<float>();
        AscendC::DataCopyParams paramsX;
        paramsX.blockCount = TILE_M;
        paramsX.blockLen = TILE_N * sizeof(float) / BLOCK_SIZE;
        paramsX.srcStride = (N_A - TILE_N) * sizeof(float) / BLOCK_SIZE;
        paramsX.dstStride = 0;
        AscendC::DataCopy(xLocal, xGm[progressInner * TILE_M * N_A], paramsX);

        AscendC::DataCopyParams paramsY;
        paramsY.blockCount = TILE_M;
        paramsY.blockLen = TILE_N * sizeof(float) / BLOCK_SIZE;
        paramsY.srcStride = (N_B - TILE_N) * sizeof(float) / BLOCK_SIZE;
        paramsY.dstStride = 0;
        AscendC::DataCopy(yLocal, yGm[progressOuter * N_A + progressInner * TILE_M * N_B], paramsY);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = inQueueY.DeQue<float>();
        AscendC::LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
        AscendC::Add(zLocal, xLocal, yLocal, TILE_M * TILE_N);
        outQueueZ.EnQue<float>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(int32_t progressOuter, int32_t progressInner)
    {
        AscendC::LocalTensor<float> zLocal = outQueueZ.DeQue<float>();
        AscendC::DataCopyParams paramsZ;
        paramsZ.blockCount = TILE_M;
        paramsZ.blockLen = TILE_N * sizeof(float) / BLOCK_SIZE;
        paramsZ.srcStride = 0;
        paramsZ.dstStride = (N_B - TILE_N) * sizeof(float) / BLOCK_SIZE;
        AscendC::DataCopy(zGm[progressOuter * N_A + progressInner * TILE_M * N_B], zLocal, paramsZ);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    static constexpr int32_t BUFFER_NUM = 2;
    static constexpr int32_t BLOCK_SIZE = 32;
    static constexpr uint32_t M_A = 5120U;
    static constexpr uint32_t N_A = M_A;
    static constexpr uint32_t M_B = M_A;
    static constexpr uint32_t N_B = N_A * 3U;
    static constexpr uint32_t TILE_M = 64U;
    static constexpr uint32_t TILE_N = 128U;

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    AscendC::GlobalTensor<float> zGm;
    AddCustomTilingData *tiling;
};
#endif // ADD_CUSTOM_V1_H