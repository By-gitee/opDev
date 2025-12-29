/**
 * @file add_custom_v1.cpp
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "add_custom_tiling.h"
#include "kernel_operator.h"

using AscendC::TPosition;
namespace {
constexpr uint32_t TILE_LENGTH = 4096;
}

class KernelAddV1 {
public:
    __aicore__ inline KernelAddV1() = default;
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t singleCoreLength)
    {
        xGm.SetGlobalBuffer((__gm__ float *)x + AscendC::GetBlockIdx() * singleCoreLength, singleCoreLength);
        yGm.SetGlobalBuffer((__gm__ float *)y + AscendC::GetBlockIdx() * singleCoreLength, singleCoreLength);
        zGm.SetGlobalBuffer((__gm__ float *)z + AscendC::GetBlockIdx() * singleCoreLength, singleCoreLength);
        loopCount = singleCoreLength / TILE_LENGTH;
    }
    __aicore__ inline void Process()
    {
        AscendC::LocalTensor<float> xLocal(AscendC::TPosition::VECCALC, xAddr, TILE_LENGTH);
        AscendC::LocalTensor<float> yLocal(AscendC::TPosition::VECCALC, yAddr, TILE_LENGTH);
        AscendC::LocalTensor<float> zLocal(AscendC::TPosition::VECCALC, zAddr, TILE_LENGTH);

        // one buffer
        for (uint32_t i = 0; i < loopCount; i++) {
            // dependency of PIPE_V & PIPE_MTE2 caused by xLocal/yLocal between 2 sequential loops
            if (i != 0) {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            }
            AscendC::DataCopy(xLocal, xGm[i * TILE_LENGTH], TILE_LENGTH);
            AscendC::DataCopy(yLocal, yGm[i * TILE_LENGTH], TILE_LENGTH);
            // dependency of PIPE_MTE2 & PIPE_V caused by xLocal/yLocal in one single loop
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            if (i != 0) {
                // dependency of PIPE_MTE3 & PIPE_V caused by zLocal between 2 sequential loops
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            }
            AscendC::Add(zLocal, xLocal, yLocal, TILE_LENGTH);
            if (i != (loopCount - 1)) {
                // dependency of PIPE_V & PIPE_MTE2 caused by xLocal/yLocal between 2 sequential loops
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            }
            // dependency of PIPE_V & PIPE_MTE3 caused by zLocal in one single loop
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::DataCopy(zGm[i * TILE_LENGTH], zLocal, TILE_LENGTH);
            if (i != (loopCount - 1)) {
                // dependency of PIPE_MTE3 & PIPE_V caused by zLocal between 2 sequential loops
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            }
        }
    }

private:
    static constexpr uint32_t xAddr = 0;
    static constexpr uint32_t yAddr = TILE_LENGTH * sizeof(float);
    static constexpr uint32_t zAddr = TILE_LENGTH * sizeof(float) * 2;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    AscendC::GlobalTensor<float> zGm;
    uint32_t loopCount;
};

extern "C" __global__ __aicore__ void add_custom_v1(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR tiling)
{
    AscendC::InitSocState();
    KernelAddV1 op;
    op.Init(x, y, z, ((__gm__ AddCustomTilingData *)tiling)->singleCoreLength);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
void add_custom_do_v1(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y, uint8_t *z, uint8_t *tiling)
{
    add_custom_v1<<<blockDim, nullptr, stream>>>(x, y, z, tiling);
}
#endif
