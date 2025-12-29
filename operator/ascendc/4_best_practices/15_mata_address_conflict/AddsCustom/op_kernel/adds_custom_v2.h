/**
 * @file adds_custom_v2.h
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef ADDS_CUSTOM_V2_H
#define ADDS_CUSTOM_V2_H
#include "kernel_operator.h"
#include "adds_custom_tiling.h"

using AscendC::TPosition;
class KernelAddsV2 {
public:
    __aicore__ inline KernelAddsV2() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR z, AddsCustomTilingData *tilingPtr)
    {
        tiling = tilingPtr;
        xGm.SetGlobalBuffer((__gm__ float *)x + AscendC::GetBlockIdx() * tiling->tileN);
        zGm.SetGlobalBuffer((__gm__ float *)z + AscendC::GetBlockIdx() * tiling->tileN);
        // the gm address conflict happens when multi cores visit the same addr range(512Bytes)
        // we disable the L2 cache mode to highlight the influence of the gm address conflict
        xGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        zGm.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, tiling->tileM * tiling->tileN * sizeof(float));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, tiling->tileM * tiling->tileN * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        for (int32_t i = 0; i < tiling->loopOneCore; i++) {
            // adjust the loop order to avoid the gm address conflict:
            // the loop order of core0  : 0, 1, 2, 3, ..., 13, 14, 15
            // the loop order of core1  : 1, 2, 3, 4, ..., 14, 15, 0
            // ...
            // the loop order of core15 : 15, 0, 1, 2, ..., 12, 13, 14
            int32_t newProgress = (i + AscendC::GetBlockIdx()) % tiling->loopOneCore;
            // the following two SyncAll in this case are unnecessary actually,
            // we just used them to highlight the influence of gm address conflict in each loop
            AscendC::SyncAll();
            CopyIn(newProgress);
            Compute();
            AscendC::SyncAll();
            CopyOut(newProgress);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.AllocTensor<float>();
        AscendC::DataCopyParams params;
        params.blockCount = tiling->tileM;
        params.blockLen = tiling->tileN * sizeof(float) / BLOCK_SIZE;
        params.srcStride = (tiling->n - tiling->tileN) * sizeof(float) / BLOCK_SIZE;
        params.dstStride = 0;
        AscendC::DataCopy(xLocal, xGm[progress * tiling->tileM * tiling->n], params);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
        AscendC::LocalTensor<float> zLocal = outQueueZ.AllocTensor<float>();
        constexpr float scale = 2.0;
        AscendC::Adds(zLocal, xLocal, scale, tiling->tileM * tiling->tileN);
        outQueueZ.EnQue<float>(zLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<float> zLocal = outQueueZ.DeQue<float>();
        AscendC::DataCopyParams params;
        params.blockCount = tiling->tileM;
        params.blockLen = tiling->tileN * sizeof(float) / BLOCK_SIZE;
        params.srcStride = 0;
        params.dstStride = (tiling->n - tiling->tileN) * sizeof(float) / BLOCK_SIZE;
        AscendC::DataCopy(zGm[progress * tiling->tileM * tiling->n], zLocal, params);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    static constexpr int32_t BUFFER_NUM = 2;
    static constexpr int32_t BLOCK_SIZE = 32;

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> zGm;
    AddsCustomTilingData *tiling;
};
#endif // ADDS_CUSTOM_V2_H