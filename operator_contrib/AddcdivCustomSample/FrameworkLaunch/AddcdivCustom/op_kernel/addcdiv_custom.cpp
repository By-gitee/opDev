#include "kernel_operator.h"
constexpr int32_t BUFFER_NUM = 2;

class KernelAddcdiv {
 public:
  __aicore__ inline KernelAddcdiv() {}
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR out,
                              float value, uint32_t blockLength,
                              uint32_t tileNum, uint32_t tileLength,
                              uint32_t lasttileLength, uint32_t formerNum,
                              uint32_t formerLength, uint32_t formertileNum,
                              uint32_t formertileLength,
                              uint32_t formerlasttileLength, uint32_t tailNum,
                              uint32_t tailLength, uint32_t tailtileNum,
                              uint32_t tailtileLength,
                              uint32_t taillasttileLength, uint32_t tilingKey) {
    this->value = static_cast<half>(value);

    if (tilingKey == 1) {
      this->blockLength = blockLength;
      this->tileNum = tileNum;
      this->tileLength = tileLength / BUFFER_NUM;
      this->lasttileLength = lasttileLength;

      xGm.SetGlobalBuffer((__gm__ half*)x + this->blockLength * AscendC::GetBlockIdx(),
                          this->blockLength);
      yGm.SetGlobalBuffer((__gm__ half*)y + this->blockLength * AscendC::GetBlockIdx(),
                          this->blockLength);
      zGm.SetGlobalBuffer((__gm__ half*)z + this->blockLength * AscendC::GetBlockIdx(),
                          this->blockLength);
      outGm.SetGlobalBuffer(
          (__gm__ half*)out + this->blockLength * AscendC::GetBlockIdx(),
          this->blockLength);
    }

    if (tilingKey == 2) {
      this->formerNum = formerNum;
      this->formerLength = formerLength;
      this->formertileNum = formertileNum;
      this->formertileLength = formertileLength;
      this->formerlasttileLength = formerlasttileLength;

      this->tailNum = tailNum;
      this->tailLength = tailLength;
      this->tailtileNum = tailtileNum;
      this->tailtileLength = tailtileLength;
      this->taillasttileLength = taillasttileLength;

      if (AscendC::GetBlockIdx() < this->formerNum) {  //分到大块核的处理
        this->tileLength = this->formertileLength / BUFFER_NUM;
        this->lasttileLength = this->formerlasttileLength;
        this->tileNum = this->formertileNum * BUFFER_NUM;
        xGm.SetGlobalBuffer(
            (__gm__ half*)x + this->formerLength * AscendC::GetBlockIdx(),
            this->formerLength);
        yGm.SetGlobalBuffer(
            (__gm__ half*)y + this->formerLength * AscendC::GetBlockIdx(),
            this->formerLength);
        zGm.SetGlobalBuffer(
            (__gm__ half*)z + this->formerLength * AscendC::GetBlockIdx(),
            this->formerLength);
        outGm.SetGlobalBuffer(
            (__gm__ half*)out + this->formerLength * AscendC::GetBlockIdx(),
            this->formerLength);
      } else {  //分到小块核的处理，需要处理的数据量比大核少alignNum个
        this->tileLength = this->tailtileLength / BUFFER_NUM;
        this->lasttileLength = this->taillasttileLength;
        this->tileNum = this->tailtileNum * BUFFER_NUM;
        xGm.SetGlobalBuffer(
            (__gm__ half*)x + this->formerLength * this->formerNum +
                this->tailLength * (AscendC::GetBlockIdx() - this->formerNum),
            this->tailLength);
        yGm.SetGlobalBuffer(
            (__gm__ half*)y + this->formerLength * this->formerNum +
                this->tailLength * (AscendC::GetBlockIdx() - this->formerNum),
            this->tailLength);
        zGm.SetGlobalBuffer(
            (__gm__ half*)z + this->formerLength * this->formerNum +
                this->tailLength * (AscendC::GetBlockIdx() - this->formerNum),
            this->tailLength);
        outGm.SetGlobalBuffer(
            (__gm__ half*)out + this->formerLength * this->formerNum +
                this->tailLength * (AscendC::GetBlockIdx() - this->formerNum),
            this->tailLength);
      }
    }

    pipe.InitBuffer(inQueueIN, BUFFER_NUM, this->tileLength * 3 * sizeof(half));
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
    AscendC::LocalTensor<half> inLocal = inQueueIN.AllocTensor<half>();

    if (BUFFER_NUM == 1) {
      if (progress == this->tileNum - 1) {
        if (progress == 0) {
          //如果只有一包，则搬运的起始地址为0，tileLength为实际分块的数据量
          AscendC::DataCopy(inLocal[0], xGm[0], this->tileLength);
          AscendC::DataCopy(inLocal[this->tileLength], yGm[0], this->tileLength);
          AscendC::DataCopy(inLocal[2 * (this->tileLength)], zGm[0], this->tileLength);
        } else {
          //将最后一个分块的起始地址向前移动tileLength-lasttileLength
          AscendC::DataCopy(
              inLocal[0],
              xGm[(progress - 1) * this->tileLength + this->lasttileLength],
              this->tileLength);
          AscendC::DataCopy(
              inLocal[this->tileLength],
              yGm[(progress - 1) * this->tileLength + this->lasttileLength],
              this->tileLength);
          AscendC::DataCopy(
              inLocal[2 * (this->tileLength)],
              zGm[(progress - 1) * this->tileLength + this->lasttileLength],
              this->tileLength);
        }
      } else {
        AscendC::DataCopy(inLocal[0], xGm[progress * this->tileLength],
                 this->tileLength);
        AscendC::DataCopy(inLocal[this->tileLength], yGm[progress * this->tileLength],
                 this->tileLength);
        AscendC::DataCopy(inLocal[2 * (this->tileLength)],
                 zGm[progress * this->tileLength], this->tileLength);
      }
    }
    if (BUFFER_NUM == 2) {
      //开启double
      //buffer时，由于将输入数据分成了相等的2部分，分块大小为不开启double
      //buffer的一半， 所以需要对最后两个分块数据的起始地址做处理
      if ((progress == (this->tileNum * BUFFER_NUM - 2)) ||
          (progress == (this->tileNum * BUFFER_NUM - 1))) {
        //分块大小变为tileLength的一半
        //倒数第2个分块数据的起始地址向前移动（tileLength-lasttileLength)，最后一个分块的起始地址以此为基础进行移动
        AscendC::DataCopy(
            inLocal[0],
            xGm[(progress - 2) * (this->tileLength) + this->lasttileLength],
            (this->tileLength));
        AscendC::DataCopy(
            inLocal[this->tileLength],
            yGm[(progress - 2) * (this->tileLength) + this->lasttileLength],
            (this->tileLength));
        AscendC::DataCopy(
            inLocal[2 * (this->tileLength)],
            zGm[(progress - 2) * (this->tileLength) + this->lasttileLength],
            (this->tileLength));

      }

      else {
        AscendC::DataCopy(inLocal[0], xGm[progress * (this->tileLength)],
                 (this->tileLength));
        AscendC::DataCopy(inLocal[this->tileLength], yGm[progress * this->tileLength],
                 this->tileLength);
        AscendC::DataCopy(inLocal[2 * (this->tileLength)],
                 zGm[progress * this->tileLength], this->tileLength);
      }
    }

    inQueueIN.EnQue(inLocal);
  }
  __aicore__ inline void Compute(int32_t progress) {
    AscendC::LocalTensor<half> inLocal = inQueueIN.DeQue<half>();
    AscendC::LocalTensor<half> xLocal = inLocal;
    AscendC::LocalTensor<half> yLocal = inLocal[this->tileLength];
    AscendC::LocalTensor<half> zLocal = inLocal[2 * (this->tileLength)];

    AscendC::LocalTensor<half> outLocal = outQueueOUT.AllocTensor<half>();

    //采用div + Muls + Add实现
    AscendC::Div(outLocal, yLocal, zLocal, this->tileLength);
    AscendC::Muls(outLocal, outLocal, this->value, this->tileLength);
    AscendC::Add(outLocal, xLocal, outLocal, this->tileLength);
    outQueueOUT.EnQue<half>(outLocal);

    inQueueIN.FreeTensor(inLocal);
  }
  __aicore__ inline void CopyOut(int32_t progress) {
    AscendC::LocalTensor<half> outLocal = outQueueOUT.DeQue<half>();

    if (BUFFER_NUM == 1) {
      if (progress == this->tileNum - 1) {
        if (progress == 0) {
          //如果只有一包，则搬运的起始地址为0，tileLength为实际分块的数据量
          AscendC::DataCopy(outGm[0], outLocal, this->tileLength);
        } else {
          //将最后一个分块的起始地址向前移动tileLength-lasttileLength
          AscendC::DataCopy(
              outGm[(progress - 1) * this->tileLength + this->lasttileLength],
              outLocal, this->tileLength);
        }
      } else {
        AscendC::DataCopy(outGm[progress * this->tileLength], outLocal,
                 this->tileLength);
      }
    }
    if (BUFFER_NUM == 2) {
      //开启double
      //buffer时，由于将输入数据分成了相等的2部分，分块大小为不开启double
      //buffer的一半， 所以需要对最后两个分块数据的起始地址做处理
      if ((progress == (this->tileNum * BUFFER_NUM - 2)) ||
          (progress == (this->tileNum * BUFFER_NUM - 1))) {
        //分块大小变为tileLength的一半
        //倒数第2个分块数据的起始地址向前移动（tileLength-lasttileLength)，最后一个分块的起始地址以此为基础进行移动
        AscendC::DataCopy(
            outGm[(progress - 2) * (this->tileLength) + this->lasttileLength],
            outLocal, (this->tileLength));
      }

      else {
        AscendC::DataCopy(outGm[progress * (this->tileLength)], outLocal,
                 (this->tileLength));
      }
    }

    outQueueOUT.FreeTensor(outLocal);
  }

 private:
  AscendC::TPipe pipe;
  // AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY, inQueueZ;
  AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueIN;
  AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueOUT;
  AscendC::GlobalTensor<half> xGm;
  AscendC::GlobalTensor<half> yGm;
  AscendC::GlobalTensor<half> zGm;
  AscendC::GlobalTensor<half> outGm;
  half value;
  uint32_t blockLength;
  uint32_t tileNum;
  uint32_t tileLength;
  uint32_t lasttileLength;
  uint32_t formerNum;
  uint32_t formerLength;
  uint32_t formertileNum;
  uint32_t formertileLength;
  uint32_t formerlasttileLength;
  uint32_t tailNum;
  uint32_t tailLength;
  uint32_t tailtileNum;
  uint32_t tailtileLength;
  uint32_t taillasttileLength;
};

extern "C" __global__ __aicore__ void addcdiv_custom(GM_ADDR x, GM_ADDR y,
                                                     GM_ADDR z, GM_ADDR out,
                                                     GM_ADDR workspace,
                                                     GM_ADDR tiling) {
  GET_TILING_DATA(tiling_data, tiling);
  // TODO: user kernel impl
  KernelAddcdiv op;

  uint32_t tilingKey = 1;
  if (TILING_KEY_IS(1)) {
    tilingKey = 1;
  } else if (TILING_KEY_IS(2)) {
    tilingKey = 2;
  } else {
    tilingKey = 1;
  }

  op.Init(x, y, z, out, tiling_data.value, tiling_data.blockLength,
          tiling_data.tileNum, tiling_data.tileLength,
          tiling_data.lasttileLength, tiling_data.formerNum,
          tiling_data.formerLength, tiling_data.formertileNum,
          tiling_data.formertileLength, tiling_data.formerlasttileLength,
          tiling_data.tailNum, tiling_data.tailLength, tiling_data.tailtileNum,
          tiling_data.tailtileLength, tiling_data.taillasttileLength,
          tilingKey);
  op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
// call of kernel function
void addcdiv_custom_do(uint32_t blockDim, void* l2ctrl, void* stream,
                       uint8_t* x, uint8_t* y, uint8_t* z, uint8_t* out,
                       uint8_t* workspace, uint8_t* tiling) {
  addcdiv_custom<<<blockDim, l2ctrl, stream>>>(x, y, z, out, workspace, tiling);
}
#endif
