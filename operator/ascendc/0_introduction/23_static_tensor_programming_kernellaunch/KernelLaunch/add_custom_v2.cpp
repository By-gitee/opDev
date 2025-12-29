/**
 * @file add_custom_v2.cpp
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

class KernelAddV2 {
public:
    __aicore__ inline KernelAddV2() = default;
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t singleCoreLength)
    {
        xGm.SetGlobalBuffer((__gm__ float *)x + AscendC::GetBlockIdx() * singleCoreLength, singleCoreLength);
        yGm.SetGlobalBuffer((__gm__ float *)y + AscendC::GetBlockIdx() * singleCoreLength, singleCoreLength);
        zGm.SetGlobalBuffer((__gm__ float *)z + AscendC::GetBlockIdx() * singleCoreLength, singleCoreLength);
        loopCount = singleCoreLength / TILE_LENGTH;
    }
    __aicore__ inline void Process()
    {
        // ping
        AscendC::LocalTensor<float> xLocalPing(AscendC::TPosition::VECCALC, xAddrPing, TILE_LENGTH);
        AscendC::LocalTensor<float> yLocalPing(AscendC::TPosition::VECCALC, yAddrPing, TILE_LENGTH);
        AscendC::LocalTensor<float> zLocalPing(AscendC::TPosition::VECCALC, zAddrPing, TILE_LENGTH);
        // pong
        AscendC::LocalTensor<float> xLocalPong(AscendC::TPosition::VECCALC, xAddrPong, TILE_LENGTH);
        AscendC::LocalTensor<float> yLocalPong(AscendC::TPosition::VECCALC, yAddrPong, TILE_LENGTH);
        AscendC::LocalTensor<float> zLocalPong(AscendC::TPosition::VECCALC, zAddrPong, TILE_LENGTH);

        // double buffer
        for (uint32_t i = 0; i < loopCount / 2; i++) {
            // ping part
            // dependency of PIPE_V & PIPE_MTE2 caused by xLocalPing/yLocalPing between 2 sequential loops
            if (i != 0) {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            }
            AscendC::DataCopy(xLocalPing, xGm[2 * i * TILE_LENGTH], TILE_LENGTH);
            AscendC::DataCopy(yLocalPing, yGm[2 * i * TILE_LENGTH], TILE_LENGTH);
            // dependency of PIPE_MTE2 & PIPE_V caused by xLocalPing/yLocalPing in one single loop
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            if (i != 0) {
                // dependency of PIPE_MTE3 & PIPE_V caused by zLocalPing between 2 sequential loops
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            }
            AscendC::Add(zLocalPing, xLocalPing, yLocalPing, TILE_LENGTH);
            if (i != (loopCount / 2 - 1)) {
                // dependency of PIPE_V & PIPE_MTE2 caused by xLocalPing/yLocalPing between 2 sequential loops
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            }
            // dependency of PIPE_V & PIPE_MTE3 caused by zLocalPing in one single loop
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::DataCopy(zGm[2 * i * TILE_LENGTH], zLocalPing, TILE_LENGTH);
            if (i != (loopCount / 2 - 1)) {
                // dependency of PIPE_MTE3 & PIPE_V caused by zLocalPing between 2 sequential loops
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            }

            // pong part
            // dependency of PIPE_V & PIPE_MTE2 caused by xLocalPong/yLocalPong between 2 sequential loops
            if (i != 0) {
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            }
            AscendC::DataCopy(xLocalPong, xGm[(2 * i + 1) * TILE_LENGTH], TILE_LENGTH);
            AscendC::DataCopy(yLocalPong, yGm[(2 * i + 1) * TILE_LENGTH], TILE_LENGTH);
            // dependency of PIPE_MTE2 & PIPE_V caused by xLocalPong/yLocalPong in one single loop
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID1);
            if (i != 0) {
                // dependency of PIPE_MTE3 & PIPE_V caused by zLocalPong between 2 sequential loops
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
            }
            AscendC::Add(zLocalPong, xLocalPong, yLocalPong, TILE_LENGTH);
            if (i != (loopCount / 2 - 1)) {
                // dependency of PIPE_V & PIPE_MTE2 caused by xLocalPong/yLocalPong between 2 sequential loops
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            }
            // dependency of PIPE_V & PIPE_MTE3 caused by zLocalPong in one single loop
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID1);
            AscendC::DataCopy(zGm[(2 * i + 1) * TILE_LENGTH], zLocalPong, TILE_LENGTH);
            if (i != (loopCount / 2 - 1)) {
                // dependency of PIPE_MTE3 & PIPE_V caused by zLocalPong between 2 sequential loops
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
            }
        }

        // tail block
        if (loopCount % 2 != 0) {
            // dependency of PIPE_V & PIPE_MTE2 caused by xLocalPing/yLocalPing with the previous for loop
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::DataCopy(xLocalPing, xGm[(loopCount - 1) * TILE_LENGTH], TILE_LENGTH);
            AscendC::DataCopy(yLocalPing, yGm[(loopCount - 1) * TILE_LENGTH], TILE_LENGTH);
            // dependency of PIPE_MTE2 & PIPE_V caused by xLocalPing/yLocalPing in one loop
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
            // dependency of PIPE_MTE3 & PIPE_V caused by zLocalPing with the previous for loop
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::Add(zLocalPing, xLocalPing, yLocalPing, TILE_LENGTH);
            // dependency of PIPE_V & PIPE_MTE3 caused by zLocalPing in one loop
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
            AscendC::DataCopy(zGm[(loopCount - 1) * TILE_LENGTH], zLocalPing, TILE_LENGTH);
        }
    }

private:
    static constexpr uint32_t xAddrPing = 0;
    static constexpr uint32_t yAddrPing = TILE_LENGTH * sizeof(float);
    static constexpr uint32_t zAddrPing = TILE_LENGTH * sizeof(float) * 2;
    static constexpr uint32_t xAddrPong = TILE_LENGTH * sizeof(float) * 3;
    static constexpr uint32_t yAddrPong = TILE_LENGTH * sizeof(float) * 4;
    static constexpr uint32_t zAddrPong = TILE_LENGTH * sizeof(float) * 5;
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    AscendC::GlobalTensor<float> zGm;
    uint32_t loopCount;
};

extern "C" __global__ __aicore__ void add_custom_v2(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR tiling)
{
    AscendC::InitSocState();
    KernelAddV2 op;
    op.Init(x, y, z, ((__gm__ AddCustomTilingData *)tiling)->singleCoreLength);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
void add_custom_do_v2(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y, uint8_t *z, uint8_t *tiling)
{
    add_custom_v2<<<blockDim, nullptr, stream>>>(x, y, z, tiling);
}
#endif
