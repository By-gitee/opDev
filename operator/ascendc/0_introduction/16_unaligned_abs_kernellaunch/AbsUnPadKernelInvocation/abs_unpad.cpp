#include "kernel_operator.h"
constexpr int32_t BLOCK_BYTE_SIZE = 28; // equivalent to the definition of blockLen of DataCopyPad
constexpr int32_t BLOCK_GROUP_NUM = 16; // equivalent to the definition of blockCount of DataCopyPad
constexpr int32_t BLOCK_ELEMENT_NUM = BLOCK_BYTE_SIZE / sizeof(half);
constexpr int32_t BLOCKLEN_CEIL = 32 / sizeof(half); // since BLOCK_BYTE_SIZE<32
constexpr int32_t USE_CORE_NUM = 8;                  // num of core used
constexpr int32_t TILE_NUM = 8;                      // split data into 8 tiles for each core
constexpr int32_t BUFFER_NUM = 2;                    // tensor num for each queue
constexpr int32_t TOTAL_LENGTH = USE_CORE_NUM * TILE_NUM * BUFFER_NUM * BLOCK_GROUP_NUM * BLOCK_ELEMENT_NUM;
constexpr int32_t BLOCK_LENGTH = TOTAL_LENGTH / USE_CORE_NUM;         // length computed of each core
constexpr int32_t TILE_LENGTH = BLOCK_LENGTH / TILE_NUM / BUFFER_NUM; // tensor num for each queue
class KernelAbsUnPad {
public:
    __aicore__ inline KernelAbsUnPad() {}
    __aicore__ inline void Init(GM_ADDR inputGM, GM_ADDR outputGM, UnPadTiling tiling)
    {
        this->tiling = tiling;
        srcGlobal.SetGlobalBuffer((__gm__ half *)(inputGM) + BLOCK_LENGTH * AscendC::GetBlockIdx(), BLOCK_LENGTH);
        dstGlobal.SetGlobalBuffer((__gm__ half *)(outputGM) + BLOCK_LENGTH * AscendC::GetBlockIdx(), BLOCK_LENGTH);
        pipe.InitBuffer(inQueue, BUFFER_NUM, BLOCK_GROUP_NUM * BLOCKLEN_CEIL * sizeof(half));
        pipe.InitBuffer(outQueue, BUFFER_NUM, BLOCK_GROUP_NUM * BLOCKLEN_CEIL * sizeof(half));
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
    __aicore__ inline void CopyIn(const int32_t progress)
    {
        AscendC::LocalTensor<half> inputLocal = inQueue.AllocTensor<half>();
        for (int32_t i = 0; i < BLOCK_GROUP_NUM; i++) {
            const uint32_t srcGmIdx = progress * BLOCK_ELEMENT_NUM * BLOCK_GROUP_NUM + BLOCK_ELEMENT_NUM * i;
            DataCopy(inputLocal[BLOCKLEN_CEIL * i], srcGlobal[srcGmIdx], BLOCKLEN_CEIL);
        }
        inQueue.EnQue(inputLocal);
    }
    __aicore__ inline void Compute(const int32_t progress)
    {
        AscendC::LocalTensor<half> inputLocal = inQueue.DeQue<half>();
        AscendC::LocalTensor<half> outputLocal = outQueue.AllocTensor<half>();
        AscendC::Abs(inputLocal, inputLocal, BLOCK_GROUP_NUM * BLOCKLEN_CEIL); // main calculation
        AscendC::UnPadParams unPadParams;
        unPadParams.rightPad = BLOCKLEN_CEIL - BLOCK_ELEMENT_NUM; // delete 2 dummy half each row
        AscendC::UnPad<half>(outputLocal, inputLocal, unPadParams, this->tiling);
        outQueue.EnQue<half>(outputLocal);
        inQueue.FreeTensor(inputLocal);
    }
    __aicore__ inline void CopyOut(const int32_t progress)
    {
        AscendC::LocalTensor<half> outputLocal = outQueue.DeQue<half>();
        AscendC::DataCopy(dstGlobal[progress * TILE_LENGTH], outputLocal, TILE_LENGTH);
        outQueue.FreeTensor(outputLocal);
    }

private:
    AscendC::GlobalTensor<half> srcGlobal;
    AscendC::GlobalTensor<half> dstGlobal;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueue;
    UnPadTiling tiling;
};

__aicore__ inline void CopyTiling(UnPadTiling *tiling, GM_ADDR tilingGM)
{
    uint32_t *ptr = reinterpret_cast<uint32_t *>(tiling);
    auto tiling32 = reinterpret_cast<__gm__ uint32_t *>(tilingGM);

    for (uint32_t i = 0; i < sizeof(UnPadTiling) / sizeof(uint32_t); i++, ptr++) {
        *ptr = *(tiling32 + i);
    }
    return;
}


extern "C" __global__ __aicore__ void abs_unpad_custom(GM_ADDR inputGM, GM_ADDR outputGM, GM_ADDR tilingData)
{
    KernelAbsUnPad op;
    UnPadTiling tiling;
    CopyTiling(&tiling, tilingData);
    op.Init(inputGM, outputGM, tiling);
    op.Process();
}