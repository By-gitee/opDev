#include "kernel_operator.h"
constexpr int32_t BLOCK_BYTE_SIZE = 14; // equivalent to the definition of blockLen of DataCopyPad
constexpr int32_t BLOCK_GROUP_NUM = 16; // equivalent to the definition of blockCount of DataCopyPad
constexpr int32_t BLOCK_ELEMENT_NUM = BLOCK_BYTE_SIZE / sizeof(half);
constexpr int32_t BLOCKLEN_CEIL = 32 / sizeof(half); // since BLOCK_BYTE_SIZE<32
constexpr int32_t USE_CORE_NUM = 8;                  // num of core used
constexpr int32_t TILE_NUM = 8;                      // split data into 16 tiles for each core
constexpr int32_t BUFFER_NUM = 2;                    // tensor num for each queue
constexpr int32_t TOTAL_LENGTH = USE_CORE_NUM * TILE_NUM * BUFFER_NUM * BLOCK_GROUP_NUM * BLOCK_ELEMENT_NUM;
constexpr int32_t BLOCK_LENGTH = TOTAL_LENGTH / USE_CORE_NUM; // length computed of each core
constexpr int32_t TILE_LENGTH = BLOCK_LENGTH / TILE_NUM / BUFFER_NUM;

__aicore__ inline void CopyTiling(PadTiling *tiling, GM_ADDR tilingGM)
{
    uint32_t *ptr = reinterpret_cast<uint32_t *>(tiling);
    auto tiling32 = reinterpret_cast<__gm__ uint32_t *>(tilingGM);

    for (uint32_t i = 0; i < sizeof(PadTiling) / sizeof(uint32_t); i++, ptr++) {
        *ptr = *(tiling32 + i);
    }
    return;
}

class KernelAbsPad {
public:
    __aicore__ inline KernelAbsPad() {}
    __aicore__ inline void Init(GM_ADDR inputGM, GM_ADDR outputGM, PadTiling tiling)
    {
        this->tiling = tiling;
        srcGlobal.SetGlobalBuffer((__gm__ half *)(inputGM) + BLOCK_LENGTH * AscendC::GetBlockIdx(), BLOCK_LENGTH);
        dstGlobal.SetGlobalBuffer((__gm__ half *)(outputGM) + BLOCK_LENGTH * AscendC::GetBlockIdx(), BLOCK_LENGTH);
        pipe.InitBuffer(inQueue, BUFFER_NUM, BLOCK_GROUP_NUM * BLOCKLEN_CEIL * sizeof(half));
        pipe.InitBuffer(outQueue, BUFFER_NUM, BLOCK_GROUP_NUM * BLOCKLEN_CEIL * sizeof(half));
        AscendC::Fill(dstGlobal, BLOCK_LENGTH, half(0.0));
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
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<half> inputLocal = inQueue.AllocTensor<half>();
        for (int32_t i = 0; i < BLOCK_GROUP_NUM; i++) {
            const uint32_t srcGmIdx = progress * TILE_LENGTH + BLOCK_ELEMENT_NUM * i;
            AscendC::DataCopy(inputLocal[BLOCKLEN_CEIL * i], srcGlobal[srcGmIdx], BLOCKLEN_CEIL);
        }
        inQueue.EnQue(inputLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<half> inputLocal = inQueue.DeQue<half>();
        AscendC::LocalTensor<half> outputLocal = outQueue.AllocTensor<half>();
        AscendC::PadParams padParams = {0, BLOCKLEN_CEIL - BLOCK_ELEMENT_NUM, 0};
        AscendC::Pad(outputLocal, inputLocal, padParams, this->tiling);
        AscendC::Abs(outputLocal, outputLocal, BLOCK_GROUP_NUM * BLOCKLEN_CEIL); // main calculation
        outQueue.EnQue<half>(outputLocal);
        inQueue.FreeTensor(inputLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<half> outputLocal = outQueue.DeQue<half>();
        AscendC::SetAtomicAdd<half>();
        for (int32_t i = 0; i < BLOCK_GROUP_NUM; i++) {
            const uint32_t srcGmIdx = progress * TILE_LENGTH + i * BLOCK_ELEMENT_NUM;
            AscendC::DataCopy<half>(dstGlobal[srcGmIdx], outputLocal[i * BLOCK_GROUP_NUM], BLOCKLEN_CEIL);
        }
        AscendC::SetAtomicNone();
        outQueue.FreeTensor(outputLocal);
    }

private:
    AscendC::GlobalTensor<half> srcGlobal;
    AscendC::GlobalTensor<half> dstGlobal;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueue;
    PadTiling tiling;
};


extern "C" __global__ __aicore__ void abs_pad_custom(GM_ADDR inputGM, GM_ADDR outputGM, GM_ADDR tilingData)
{
    KernelAbsPad op;
    PadTiling tiling;
    CopyTiling(&tiling, tilingData);
    op.Init(inputGM, outputGM, tiling);
    op.Process();
}