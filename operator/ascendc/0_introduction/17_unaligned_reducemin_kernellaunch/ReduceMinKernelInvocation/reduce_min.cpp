#include "kernel_operator.h"
constexpr int32_t BLOCK_BYTE_SIZE = 8; // equivalent to the definition of blockLen of DataCopyPad
constexpr int32_t BLOCK_GROUP_NUM = 4; // equivalent to the definition of blockCount of DataCopyPad
constexpr int32_t BLOCK_ELEMENT_NUM = BLOCK_BYTE_SIZE / sizeof(half);
constexpr int32_t BLOCKLEN_CEIL = 32 / sizeof(half); // since BLOCK_BYTE_SIZE<32
constexpr int32_t USE_CORE_NUM = 4;                  // num of core used
constexpr int32_t TILE_NUM = 1;
constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t DEFAULT_SYNCALL_NEED_SIZE = 8;
constexpr int32_t TOTAL_LENGTH = USE_CORE_NUM * TILE_NUM * BUFFER_NUM * BLOCK_GROUP_NUM * BLOCK_ELEMENT_NUM;
constexpr int32_t BLOCK_LENGTH = TOTAL_LENGTH / USE_CORE_NUM;         // length computed of each core
constexpr int32_t TAIL_LENGTH = BLOCKLEN_CEIL - BLOCK_ELEMENT_NUM;    // length of tail block in the last core
class KernelReduceMin {
public:
    __aicore__ inline KernelReduceMin() {}
    __aicore__ inline void Init(GM_ADDR inputGM, GM_ADDR outputGM, GM_ADDR syncGM)
    {
        uint32_t blockLength = BLOCK_LENGTH;
        if (AscendC::GetBlockIdx() == USE_CORE_NUM - 1) {
            blockLength = TAIL_LENGTH + BLOCK_LENGTH;
        }
        srcGlobal.SetGlobalBuffer((__gm__ half *)(inputGM) + BLOCK_LENGTH * AscendC::GetBlockIdx(), BLOCK_LENGTH);
        dstGlobal.SetGlobalBuffer((__gm__ half *)(outputGM) + BLOCK_LENGTH * AscendC::GetBlockIdx(), blockLength);
        syncGlobal.SetGlobalBuffer((__gm__ int32_t *)(syncGM), USE_CORE_NUM * DEFAULT_SYNCALL_NEED_SIZE);
        // clear dstGm before doing calculations
        AscendC::Fill<half>(dstGlobal, blockLength, 0);

        pipe.InitBuffer(inQueue, BUFFER_NUM, BLOCK_GROUP_NUM * BLOCKLEN_CEIL * sizeof(half));
        pipe.InitBuffer(outQueue, BUFFER_NUM, BLOCK_GROUP_NUM * BLOCKLEN_CEIL * sizeof(half));
        pipe.InitBuffer(workLocalTbuf, BLOCKLEN_CEIL * sizeof(half));
        pipe.InitBuffer(syncLocalTbuf, USE_CORE_NUM * DEFAULT_SYNCALL_NEED_SIZE * sizeof(int32_t));

        AscendC::LocalTensor<int32_t> SyncLocal = syncLocalTbuf.Get<int32_t>();
        AscendC::SyncAll(syncGlobal, SyncLocal);
    }
    __aicore__ inline void Process()
    {
        const int32_t loopCount = TILE_NUM * BUFFER_NUM;
        // tiling strategy, pipeline parallel
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
        for (int i = 0; i < BLOCK_GROUP_NUM; i++) {
            AscendC::DataCopy(inputLocal[i * BLOCKLEN_CEIL], srcGlobal[i * BLOCK_ELEMENT_NUM],
                              BLOCKLEN_CEIL); // each time copy 16 half elements to UB
        }
        inQueue.EnQue(inputLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<half> outputLocal = outQueue.AllocTensor<half>();
        AscendC::LocalTensor<half> workLocal = workLocalTbuf.Get<half>();
        AscendC::LocalTensor<half> inputLocal = inQueue.DeQue<half>();
        AscendC::Duplicate<half>(outputLocal, static_cast<half>(0), BLOCK_GROUP_NUM * BLOCKLEN_CEIL);
        AscendC::Duplicate<half>(workLocal, static_cast<half>(0), BLOCKLEN_CEIL);

        uint64_t Mask0 = ((uint64_t)1 << BLOCK_ELEMENT_NUM) -
                         1; // mask mode controls only the first 4 elements do ReduceMin calculation
        uint64_t Mask[2] = {Mask0, 0};
        // main calculation
        for (int i = 0; i < BLOCK_GROUP_NUM; i++) {
            AscendC::ReduceMin<half>(outputLocal[i * BLOCKLEN_CEIL], inputLocal[i * BLOCKLEN_CEIL], workLocal, Mask, 1,
                                     8, false);
        }
        outQueue.EnQue<half>(outputLocal);
        inQueue.FreeTensor(inputLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<half> outputLocal = outQueue.DeQue<half>();
        AscendC::SetAtomicAdd<half>();
        for (int i = 0; i < BLOCK_GROUP_NUM; i++) {
            AscendC::DataCopy<half>(dstGlobal[i * BLOCK_ELEMENT_NUM], outputLocal[i * BLOCKLEN_CEIL], BLOCKLEN_CEIL);
        }
        AscendC::SetAtomicNone();
        outQueue.FreeTensor(outputLocal);
    }

private:
    AscendC::GlobalTensor<half> srcGlobal;
    AscendC::GlobalTensor<half> dstGlobal;
    AscendC::GlobalTensor<int32_t> syncGlobal;
    AscendC::TPipe pipe;
    AscendC::TBuf<> workLocalTbuf;
    AscendC::TBuf<> syncLocalTbuf;

    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> workQueue;
};
extern "C" __global__ __aicore__ void reduce_min_custom(GM_ADDR inputGM, GM_ADDR outputGM, GM_ADDR syncGM)
{
    KernelReduceMin op;
    op.Init(inputGM, outputGM, syncGM);
    op.Process();
}