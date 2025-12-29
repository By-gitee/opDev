/**
 * @file add_custom_v4.cpp
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
constexpr uint32_t DST_START_ADDRESS = 0x20000;
}

class KernelAddV4 {
public:
    __aicore__ inline KernelAddV4() = default;
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
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
        for (uint32_t i = 0; i < loopCount; i++) {
            int32_t eventID = (i % 2 == 0 ? EVENT_ID0 : EVENT_ID1);
            AscendC::LocalTensor<float> &xLocal = (i % 2 == 0 ? xLocalPing : xLocalPong);
            AscendC::LocalTensor<float> &yLocal = (i % 2 == 0 ? yLocalPing : yLocalPong);
            AscendC::LocalTensor<float> &zLocal = (i % 2 == 0 ? zLocalPing : zLocalPong);
            // dependency of PIPE_MTE3 & PIPE_MTE2 caused by xLocal/yLocal between 2 sequential loops
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventID);
            AscendC::DataCopy(xLocal, xGm[i * TILE_LENGTH], TILE_LENGTH);
            AscendC::DataCopy(yLocal, yGm[i * TILE_LENGTH], TILE_LENGTH);

            // dependency of PIPE_MTE2 & PIPE_V caused by xLocal/yLocal in one single loop
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(eventID);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(eventID);
            AscendC::Add(zLocal, xLocal, yLocal, TILE_LENGTH);
            // dependency of PIPE_V & PIPE_MTE3 caused by zLocal in one single loop
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(eventID);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(eventID);
            AscendC::DataCopy(zGm[i * TILE_LENGTH], zLocal, TILE_LENGTH);
            // dependency of PIPE_MTE3 & PIPE_MTE2 caused by zLocal between 2 sequential loops
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventID);
        }
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
    }

private:
    // according to bank conflict rule:
    // rr conflict happened when 2 read requests are in the same bank group
    // rw conflict happened when read and write requests are in the same bank
    // so we adjust the address to avoid bank conflicts
    static constexpr uint32_t xAddrPing = 0x0;
    static constexpr uint32_t yAddrPing = TILE_LENGTH * sizeof(float) + 256;
    static constexpr uint32_t zAddrPing = DST_START_ADDRESS;
    static constexpr uint32_t xAddrPong = TILE_LENGTH * sizeof(float) * 2 + 256;
    static constexpr uint32_t yAddrPong = TILE_LENGTH * sizeof(float) * 3 + 512;
    static constexpr uint32_t zAddrPong = DST_START_ADDRESS + TILE_LENGTH * sizeof(float);
    AscendC::GlobalTensor<float> xGm;
    AscendC::GlobalTensor<float> yGm;
    AscendC::GlobalTensor<float> zGm;
    uint32_t loopCount;
};

extern "C" __global__ __aicore__ void add_custom_v4(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR tiling)
{
    AscendC::InitSocState();
    KernelAddV4 op;
    op.Init(x, y, z, ((__gm__ AddCustomTilingData *)tiling)->singleCoreLength);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
void add_custom_do_v4(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y, uint8_t *z, uint8_t *tiling)
{
    add_custom_v4<<<blockDim, nullptr, stream>>>(x, y, z, tiling);
}
#endif
