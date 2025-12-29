/*
 *
 *
 * Function :
 *
 */
#include "kernel_operator.h"

// constexpr int32_t TOTAL_LENGTH_IN = 8 * 2048;// 8 * 2048 // total length of
// data
constexpr int32_t TOTAL_LENGTH = 8 * 2048;
constexpr int32_t USE_CORE_NUM = 8;  // num of core used
constexpr int32_t BLOCK_LENGTH =
    TOTAL_LENGTH / USE_CORE_NUM;   // length computed of each core
constexpr int32_t TILE_NUM = 16;   // split data into 8 tiles for each core
constexpr int32_t BUFFER_NUM = 1;  // tensor num for each queue
constexpr int32_t TILE_LENGTH =
    BLOCK_LENGTH / TILE_NUM /
    BUFFER_NUM;  // seperate to 2 parts, due to double buffer

class KernelAddcdiv {
 public:
  __aicore__ inline KernelAddcdiv() {}
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR out) {
    this->blockLength = BLOCK_LENGTH;
    this->tileNum = TILE_NUM;
    ASSERT(tileNum != 0 && "tile num can not be zero!");
    this->tileLength = TILE_LENGTH;
    this->value = (half)1.0;  // 与gen_data.py内value保存一致

    xGm.SetGlobalBuffer((__gm__ half*)x + this->blockLength * AscendC::GetBlockIdx(),
                        this->blockLength);
    yGm.SetGlobalBuffer((__gm__ half*)y + this->blockLength * AscendC::GetBlockIdx(),
                        this->blockLength);
    zGm.SetGlobalBuffer((__gm__ half*)z + this->blockLength * AscendC::GetBlockIdx(),
                        this->blockLength);
    outGm.SetGlobalBuffer((__gm__ half*)out + this->blockLength * AscendC::GetBlockIdx(),
                          this->blockLength);
    pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(half));
    pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(half));
    pipe.InitBuffer(inQueueZ, BUFFER_NUM, this->tileLength * sizeof(half));
    pipe.InitBuffer(outQueueOUT, BUFFER_NUM, this->tileLength * sizeof(half));
  }
  __aicore__ inline void Process() {
    int32_t loopCount = this->tileNum * BUFFER_NUM;
    for (int32_t i = 0; i < loopCount; i++) {
      CopyIn(i);
      Compute(i);
      CopyOut(i);
    }
  }

 private:
  __aicore__ inline void CopyIn(int32_t progress) {
    AscendC::LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
    AscendC::LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();
    AscendC::LocalTensor<half> zLocal = inQueueZ.AllocTensor<half>();
    AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
    AscendC::DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);
    AscendC::DataCopy(zLocal, zGm[progress * this->tileLength], this->tileLength);
    inQueueX.EnQue(xLocal);
    inQueueY.EnQue(yLocal);
    inQueueZ.EnQue(zLocal);
  }
  __aicore__ inline void Compute(int32_t progress) {
    AscendC::LocalTensor<half> xLocal = inQueueX.DeQue<half>();
    AscendC::LocalTensor<half> yLocal = inQueueY.DeQue<half>();
    AscendC::LocalTensor<half> zLocal = inQueueZ.DeQue<half>();
    AscendC::LocalTensor<half> outLocal = outQueueOUT.AllocTensor<half>();
    // compute
    AscendC::Div(outLocal, yLocal, zLocal, this->tileLength);
    AscendC::Muls(outLocal, outLocal, this->value, this->tileLength);
    AscendC::Add(outLocal, xLocal, outLocal, this->tileLength);
    outQueueOUT.EnQue<half>(outLocal);
    inQueueX.FreeTensor(xLocal);
    inQueueY.FreeTensor(yLocal);
    inQueueZ.FreeTensor(zLocal);
  }
  __aicore__ inline void CopyOut(int32_t progress) {
    AscendC::LocalTensor<half> outLocal = outQueueOUT.DeQue<half>();
    AscendC::DataCopy(outGm[progress * this->tileLength], outLocal, this->tileLength);
    outQueueOUT.FreeTensor(outLocal);
  }

 private:
  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY, inQueueZ;
  AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueOUT;
  AscendC::GlobalTensor<half> xGm;
  AscendC::GlobalTensor<half> yGm;
  AscendC::GlobalTensor<half> zGm;
  AscendC::GlobalTensor<half> outGm;
  half value;
  uint32_t blockLength;
  uint32_t tileNum;
  uint32_t tileLength;
};

extern "C" __global__ __aicore__ void addcdiv_custom(GM_ADDR x, GM_ADDR y,
                                                     GM_ADDR z, GM_ADDR out) {
  KernelAddcdiv op;
  op.Init(x, y, z, out);
  op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
// call of kernel function
void addcdiv_custom_do(uint32_t blockDim, void* l2ctrl, void* stream,
                       uint8_t* x, uint8_t* y, uint8_t* z, uint8_t* out) {
  addcdiv_custom<<<blockDim, l2ctrl, stream>>>(x, y, z, out);
}
#endif