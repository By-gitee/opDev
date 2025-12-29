/**
 * @file abs_gather_mask.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "kernel_operator.h"
constexpr int32_t BLOCK_BYTE_SIZE = 36; // copy 18 elements for each row in a tensor
constexpr int32_t BLOCKLEN_CEIL =
    (BLOCK_BYTE_SIZE + 32 - 1) / 32 * 32 / sizeof(half); // round up with respect to 32 bytes
constexpr int32_t BLOCK_ELEMENT_NUM = BLOCK_BYTE_SIZE / sizeof(half);
constexpr int32_t USE_CORE_NUM = 8; // num of core used
constexpr int32_t TILE_NUM = 8;    // split data into 16 tiles for each core
constexpr int32_t BUFFER_NUM = 2;   // tensor num for each queue
constexpr int32_t TOTAL_LENGTH = USE_CORE_NUM * TILE_NUM * BUFFER_NUM * BLOCK_ELEMENT_NUM;
constexpr int32_t BLOCK_LENGTH = TOTAL_LENGTH / USE_CORE_NUM;
constexpr int32_t TILE_LENGTH = BLOCK_LENGTH / TILE_NUM / BUFFER_NUM;
class KernelAbsGatherMask {
public:
    __aicore__ inline KernelAbsGatherMask() {}
    __aicore__ inline void Init(GM_ADDR inputGM, GM_ADDR outputGM)
    {
        srcGlobal.SetGlobalBuffer((__gm__ half *)(inputGM) + BLOCK_LENGTH * AscendC::GetBlockIdx(), BLOCK_LENGTH);
        dstGlobal.SetGlobalBuffer((__gm__ half *)(outputGM) + BLOCK_LENGTH * AscendC::GetBlockIdx(), BLOCK_LENGTH);
        pipe.InitBuffer(inQueue, BUFFER_NUM, BLOCKLEN_CEIL * sizeof(half));
        pipe.InitBuffer(outQueue, BUFFER_NUM, BLOCKLEN_CEIL * sizeof(half));
        pipe.InitBuffer(outQueueTail, BUFFER_NUM, 32);
        pipe.InitBuffer(tmpPattern, 32);
    }
    __aicore__ inline void Process()
    {
        const int32_t loopCount = TILE_NUM * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress) // GM->UB
    {
        AscendC::LocalTensor<half> inputLocal = inQueue.AllocTensor<half>();
        AscendC::DataCopy(inputLocal, srcGlobal[progress * TILE_LENGTH],
                          BLOCKLEN_CEIL); // each time copy 32 half elements to UB
        inQueue.EnQue(inputLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<half> outputLocal = outQueue.AllocTensor<half>();
        AscendC::LocalTensor<half> inputLocal = inQueue.DeQue<half>();
        AscendC::Abs(outputLocal, inputLocal, BLOCKLEN_CEIL); // main calculation
        AscendC::LocalTensor<uint16_t> bufPattern = tmpPattern.Get<uint16_t>();
        uint16_t tmpValue = 0;
        AscendC::Duplicate<uint16_t>(bufPattern, tmpValue, 16);
        bufPattern.SetValue(0, 0b1111111111111100); // select the last 14 elements of the first 16 positions
        bufPattern.SetValue(1, 0b0000000000000011); // select the first 2 elements of the next 16 positions
        uint32_t mask = 32;
        uint64_t rsvdCnt = 0;
        AscendC::LocalTensor<half> tailLocal = outQueueTail.AllocTensor<half>();
        AscendC::GatherMask(tailLocal, outputLocal, bufPattern, true, mask, {1, 1, 8, 8}, rsvdCnt);
        outQueue.EnQue<half>(outputLocal);
        outQueueTail.EnQue<half>(tailLocal);
        inQueue.FreeTensor(inputLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<half> outputLocal = outQueue.DeQue<half>();
        AscendC::LocalTensor<half> tailLocal = outQueueTail.DeQue<half>();
        uint32_t copyLenMain = TILE_LENGTH * sizeof(half) / 32 * 32 / sizeof(half);
        uint32_t offsetMain = progress * TILE_LENGTH;
        AscendC::DataCopy(dstGlobal[offsetMain], outputLocal, copyLenMain);
        AscendC::PipeBarrier<PIPE_MTE3>();
        uint32_t tailLen = 32 / sizeof(half);
        uint32_t offsetTail = offsetMain + (TILE_LENGTH - tailLen);
        AscendC::DataCopy(dstGlobal[offsetTail], tailLocal, tailLen);
        outQueue.FreeTensor(outputLocal);
        outQueueTail.FreeTensor(tailLocal);
    }

private:
    AscendC::GlobalTensor<half> srcGlobal;
    AscendC::GlobalTensor<half> dstGlobal;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueue, outQueueTail;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpPattern;
};
extern "C" __global__ __aicore__ void abs_gather_mask_custom(GM_ADDR inputGM, GM_ADDR outputGM)
{
    KernelAbsGatherMask op;
    op.Init(inputGM, outputGM);
    op.Process();
}