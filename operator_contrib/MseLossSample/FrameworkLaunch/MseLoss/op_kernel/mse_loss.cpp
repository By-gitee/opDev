/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 */
#include "kernel_operator.h"
constexpr int32_t BUFFER_NUM = 1;

class KernelMseLoss {
public:
    __aicore__ inline KernelMseLoss() {}
    __aicore__ inline void Init(GM_ADDR predict, GM_ADDR label, GM_ADDR y, int mode, 
                                uint32_t totalLength, uint32_t blockLength,
                                uint32_t tileNum, uint32_t tileLength,
                                uint32_t lasttileLength) {
        ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");

        // 该算子有两种策略：NONE模式与其他模式
        //     NONE模式(mode=3)： 只需要将对应的xi与yi进行(xi - yi) ^ 2计算即可
        //     其他模式(mode=1,2)： 首先需要计算(xi - yi) ^ 2，此后进行一个规约计算(sum/mean)
        //                考虑到空间限制的问题，采取的策略是将每一次tiling计算(xi - yi) ^ 2后就进行一次规约操作，并将该结果存入一个变量里（在tempBuf里） 
        //                最后在tiling结束后对这些数量为reduce_num的数据再进行最后一次规约操作得到最终结果
        this->mode = mode;
        this->totalLength = static_cast<int32_t>(totalLength);
        this->totalLength_f32 = static_cast<float>(this->totalLength);

        if (this->mode == 3) {
            this->blockLength = blockLength;
            this->tileNum =
                tileNum ASSERT(tileNum != 0 && "tile num can not be zero!");
            this->tileLength = tileLength / BUFFER_NUM;
            this->lasttileLength = lasttileLength;

            xGm.SetGlobalBuffer((__gm__ DTYPE_Y*)predict + this->blockLength * AscendC::GetBlockIdx(),
                                this->blockLength);
            yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)label + this->blockLength * AscendC::GetBlockIdx(),
                                this->blockLength);
            outGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y + this->blockLength * AscendC::GetBlockIdx(),
                                 this->blockLength);
        } 
        else {
            this->blockLength = blockLength;
            this->tileNum =
                tileNum ASSERT(tileNum != 0 && "tile num can not be zero!");
            this->tileLength = tileLength / BUFFER_NUM;
            this->lasttileLength = lasttileLength;

            xGm.SetGlobalBuffer((__gm__ DTYPE_Y*)predict + this->blockLength * AscendC::GetBlockIdx(),
                                this->blockLength);
            yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)label + this->blockLength * AscendC::GetBlockIdx(),
                                this->blockLength);
            outGm.SetGlobalBuffer(
                (__gm__ DTYPE_Y*)y + this->blockLength * AscendC::GetBlockIdx(), 32);
        }

        this->reduce_num = this->tileNum * BUFFER_NUM;
        uint32_t reduce_align = (this->reduce_num + 31) / 32 * 32;

        if (this->mode == 3) {
            pipe.InitBuffer(inQueueIN, BUFFER_NUM, this->tileLength * 2 * sizeof(DTYPE_Y));
            pipe.InitBuffer(outQueueOUT, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
        } 
        else if (this->mode == 1 || this->mode == 2) {
            pipe.InitBuffer(inQueueIN, BUFFER_NUM, this->tileLength * 2 * sizeof(DTYPE_Y));
            pipe.InitBuffer(tempBuf, reduce_align * sizeof(DTYPE_Y));
        }
    }

    __aicore__ inline void Process() {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        if (this->mode == 3) {
            for (int32_t i = 0; i < loopCount; i++) {
                CopyIn_Strategy_1(i);
                Compute_Strategy_1(i);
                CopyOut_Strategy_1(i);
            }
        }
        else if (this->mode == 1 || this->mode == 2) {
            for (int32_t i = 0; i < loopCount; i++) {
                CopyIn_Strategy_2(i);
                Compute_Strategy_2(i);
            }
            AscendC::LocalTensor<DTYPE_Y> temp1 = tempBuf.Get<DTYPE_Y>();
            AscendC::LocalTensor<DTYPE_Y> temp2 = inQueueIN.AllocTensor<DTYPE_Y>();

            AscendC::Duplicate(temp2, (DTYPE_Y)0, this->tileLength);
            AscendC::ReduceSum<DTYPE_Y>(temp1, temp1, temp2, this->reduce_num);

            if (this->mode == 1) {
                DTYPE_Y len = static_cast<DTYPE_Y>(this->totalLength_f32);
                temp2.SetValue(0, len);
                AscendC::Div(temp1, temp1, temp2, 1);
            }
            outGm.SetValue(0, temp1.GetValue(0));
            inQueueIN.FreeTensor(temp2);
        }
    }

private:
    __aicore__ inline void CopyIn_Strategy_1(int32_t progress) {
        AscendC::LocalTensor<DTYPE_Y> inLocal = inQueueIN.AllocTensor<DTYPE_Y>();

        if (BUFFER_NUM == 1) {
            if (progress == this->tileNum - 1) {
                if (progress == 0) {
                    //如果只有一包，则搬运的起始地址为0，tileLength为实际分块的数据量
                    AscendC::DataCopy(inLocal[0], xGm[0], this->tileLength);
                    AscendC::DataCopy(inLocal[this->tileLength], yGm[0], this->tileLength);
                } 
                else {
                    //将最后一个分块的起始地址向前移动tileLength-lasttileLength
                    AscendC::DataCopy(
                        inLocal[0],
                        xGm[(progress - 1) * this->tileLength + this->lasttileLength],
                        this->tileLength);
                    AscendC::DataCopy(
                        inLocal[this->tileLength],
                        yGm[(progress - 1) * this->tileLength + this->lasttileLength],
                        this->tileLength);
                }
            } 
            else {
                AscendC::DataCopy(inLocal[0], xGm[progress * this->tileLength],
                        this->tileLength);
                AscendC::DataCopy(inLocal[this->tileLength], yGm[progress * this->tileLength],
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
                    inLocal[0],
                    xGm[(progress - 2) * (this->tileLength) + this->lasttileLength],
                    (this->tileLength));
                AscendC::DataCopy(
                    inLocal[this->tileLength],
                    yGm[(progress - 2) * (this->tileLength) + this->lasttileLength],
                    (this->tileLength));
            }
            else {
                AscendC::DataCopy(inLocal[0], xGm[progress * (this->tileLength)],
                        (this->tileLength));
                AscendC::DataCopy(inLocal[this->tileLength], yGm[progress * this->tileLength],
                        this->tileLength);
            }
        }
        inQueueIN.EnQue(inLocal);
    }

    __aicore__ inline void Compute_Strategy_1(int32_t progress) {
        AscendC::LocalTensor<DTYPE_Y> inLocal = inQueueIN.DeQue<DTYPE_Y>();
        AscendC::LocalTensor<DTYPE_Y> xLocal = inLocal;
        AscendC::LocalTensor<DTYPE_Y> yLocal = inLocal[this->tileLength];

        AscendC::LocalTensor<DTYPE_Y> outLocal = outQueueOUT.AllocTensor<DTYPE_Y>();

        AscendC::Sub(outLocal, xLocal, yLocal, this->tileLength);
        AscendC::Mul(outLocal, outLocal, outLocal, this->tileLength);
        
        outQueueOUT.EnQue<DTYPE_Y>(outLocal);

        inQueueIN.FreeTensor(inLocal);
    }

    __aicore__ inline void CopyOut_Strategy_1(int32_t progress) {
        AscendC::LocalTensor<DTYPE_Y> outLocal = outQueueOUT.DeQue<DTYPE_Y>();

        if (BUFFER_NUM == 1) {
            if (progress == this->tileNum - 1) {
                if (progress == 0) {
                    //如果只有一包，则搬运的起始地址为0，tileLength为实际分块的数据量
                    AscendC::DataCopy(outGm[0], outLocal, this->tileLength);
                } 
                else {
                    //将最后一个分块的起始地址向前移动tileLength-lasttileLength
                    AscendC::DataCopy(
                        outGm[(progress - 1) * this->tileLength + this->lasttileLength],
                        outLocal, this->tileLength);
                }
            } 
            else {
                AscendC::DataCopy(outGm[progress * this->tileLength], outLocal, this->tileLength);
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
                AscendC::DataCopy(outGm[progress * (this->tileLength)], outLocal, this->tileLength);
            }
        }
        outQueueOUT.FreeTensor(outLocal);
    }

    __aicore__ inline void CopyIn_Strategy_2(int32_t progress) {
        AscendC::LocalTensor<DTYPE_Y> inLocal = inQueueIN.AllocTensor<DTYPE_Y>();

        // 对于不同BUFFER_NUM的处理同CopyIn_Strategy_1
        if (BUFFER_NUM == 1) {
            if (progress == this->tileNum - 1) {
                if (progress == 0) {
                    AscendC::DataCopy(inLocal[0], xGm[0], this->tileLength);
                    AscendC::DataCopy(inLocal[this->tileLength], yGm[0], this->tileLength);
                } 
                else {
                    AscendC::DataCopy(
                        inLocal[0],
                        xGm[(progress - 1) * this->tileLength + this->lasttileLength],
                        this->tileLength);
                    AscendC::DataCopy(
                        inLocal[this->tileLength],
                        yGm[(progress - 1) * this->tileLength + this->lasttileLength],
                        this->tileLength);
                }
            } 
            else {
                AscendC::DataCopy(inLocal[0], xGm[progress * this->tileLength],
                        this->tileLength);
                AscendC::DataCopy(inLocal[this->tileLength], yGm[progress * this->tileLength],
                        this->tileLength);
            }
        }
        if (BUFFER_NUM == 2) {
            if ((progress == (this->tileNum * BUFFER_NUM - 2)) ||
                (progress == (this->tileNum * BUFFER_NUM - 1))) {
                AscendC::DataCopy(
                    inLocal[0],
                    xGm[(progress - 2) * (this->tileLength) + this->lasttileLength],
                    (this->tileLength));
                AscendC::DataCopy(
                    inLocal[this->tileLength],
                    yGm[(progress - 2) * (this->tileLength) + this->lasttileLength],
                    (this->tileLength));
            }
            else {
                AscendC::DataCopy(inLocal[0], xGm[progress * (this->tileLength)],
                        (this->tileLength));
                AscendC::DataCopy(inLocal[this->tileLength], yGm[progress * this->tileLength],
                        this->tileLength);
            }
        }
        inQueueIN.EnQue(inLocal);
    }

    __aicore__ inline void Compute_Strategy_2(int32_t progress) {
        AscendC::LocalTensor<DTYPE_Y> inLocal = inQueueIN.DeQue<DTYPE_Y>();
        AscendC::LocalTensor<DTYPE_Y> xLocal = inLocal;
        AscendC::LocalTensor<DTYPE_Y> yLocal = inLocal[this->tileLength];
        AscendC::LocalTensor<DTYPE_Y> temp1 = tempBuf.Get<DTYPE_Y>();

        AscendC::Sub(yLocal, xLocal, yLocal, this->tileLength);
        AscendC::Mul(yLocal, yLocal, yLocal, this->tileLength);
        AscendC::ReduceSum<DTYPE_Y>(yLocal, yLocal, xLocal, this->tileLength);
        temp1.SetValue(progress, yLocal.GetValue(0));
        
        inQueueIN.FreeTensor(inLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueIN;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueOUT;
    AscendC::TBuf<> tempBuf;
    AscendC::GlobalTensor<DTYPE_Y> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    AscendC::GlobalTensor<DTYPE_Y> outGm;
    int mode;
    float totalLength_f32;
    int32_t totalLength;
    uint32_t reduce_num;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t lasttileLength;
};


extern "C" __global__ __aicore__ void mse_loss(GM_ADDR predict, GM_ADDR label, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelMseLoss op;

    op.Init(predict, label, y, tiling_data.mode, tiling_data.totalLength, tiling_data.blockLength,
            tiling_data.tileNum, tiling_data.tileLength, tiling_data.lasttileLength);
    op.Process();
}