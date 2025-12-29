/**
* @file histogram.cpp
*
* Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

template<typename TYPE_X, typename TYPE_Y> class KernelHistogram {
    using T = TYPE_X;
public:
    __aicore__ inline KernelHistogram() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                uint32_t CoreDataNum, uint32_t finalTileNum, uint32_t tileDataNum, uint32_t TailDataNum,
                                int bins, float min, float max) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

        this->coreDataNum = CoreDataNum;
        this->tileNum = finalTileNum;
        this->tileDataNum = tileDataNum;
        this->tailDataNum = TailDataNum;
        this->bins = bins;
        this->min = min;
        this->max = max;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y, this->coreDataNum);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(DTYPE_X));

        pipe.InitBuffer(QueueBits, (this->tileDataNum+256)  * sizeof(uint8_t));// / 2);
        pipe.InitBuffer(QueueZero, (this->tileDataNum+256)  * sizeof(float));
        pipe.InitBuffer(QueueTmp1, (this->tileDataNum+256)  * sizeof(float));
        pipe.InitBuffer(QueueTmp2, (this->tileDataNum+256) * sizeof(float));
        if(this->tileDataNum < 256)
            pipe.InitBuffer(QueueBuff, 256 * sizeof(float));
        else
            pipe.InitBuffer(QueueBuff, this->tileDataNum * sizeof(float));
    }
    __aicore__ inline void Process() {
        auto tmp1 = QueueTmp1.Get<float>();
        auto tmp2 = QueueTmp2.Get<float>();
        auto buff = QueueBuff.Get<float>();

        int32_t loopCount = this->tileNum;

        this->processDataNum = this->tileDataNum;
        for (int32_t i = 0; i < loopCount; i++) 
        {
            if (i == this->tileNum - 1) 
            {
              this->processDataNum = this->tailDataNum;
            }

            LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
            DataCopy(xLocal, xGm[i * this->tileDataNum], (this->processDataNum+31)/32*32);
            inQueueX.EnQue(xLocal);
            xLocal = inQueueX.DeQue<DTYPE_X>();
            if constexpr (std::is_same_v<DTYPE_X, float>)
            {
                if(this->min == 0 && this->max == 0)
                {
                    ReduceMax(tmp1, xLocal, buff, this->processDataNum, false);
                    if(i == 0)
                        this->max = tmp1.GetValue(0);
                    else
                    {
                        if(this->max < tmp1.GetValue(0))
                            this->max = tmp1.GetValue(0);
                    }
                    ReduceMin(tmp1, xLocal, buff, this->processDataNum, false);
                    if(i == 0)
                        this->min = tmp1.GetValue(0);
                    else
                    {
                        if(this->min > tmp1.GetValue(0))
                            this->min = tmp1.GetValue(0);
                    }
                }
            }
            else
            {
                Cast(tmp2, xLocal, RoundMode::CAST_NONE, this->processDataNum);
                if(this->min == 0 && this->max == 0)
                {
                    ReduceMax(tmp1, tmp2, buff, this->processDataNum, false);
                    if(i == 0)
                        this->max = tmp1.GetValue(0);
                    else
                    {
                        if(this->max < tmp1.GetValue(0))
                            this->max = tmp1.GetValue(0);
                    }
                    ReduceMin(tmp1, tmp2, buff, this->processDataNum, false);
                    if(i == 0)
                        this->min = tmp1.GetValue(0);
                    else
                    {
                        if(this->min > tmp1.GetValue(0))
                            this->min = tmp1.GetValue(0);
                    }
                }
            }
            inQueueX.FreeTensor(xLocal);
        }

        this->processDataNum = this->tileDataNum;
        for (int32_t i = 0; i < loopCount; i++) {
            if (i == this->tileNum - 1) {
                this->processDataNum = this->tailDataNum;
            }
            CopyIn(i);
            Compute(i);
        }   
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        DataCopy(xLocal, xGm[progress * this->tileDataNum], (this->processDataNum+31)/32*32);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();

        auto tmp1 = QueueTmp1.Get<float>();
        auto tmp2 = QueueTmp2.Get<float>();
        auto buff = QueueBuff.Get<float>();
        auto bits = QueueBits.Get<uint8_t>();
        auto zero = QueueZero.Get<float>();

        float neg_min= -this->min;
        float interval = (this->max - this->min);
        float float_bins = this->bins;
        float var1;

        Duplicate(zero, float(1), this->processDataNum);

        if constexpr (std::is_same_v<DTYPE_X, float>)
        {
            Adds(tmp1, xLocal, neg_min, this->processDataNum);
        }
        else
        {
            Cast(tmp1, xLocal, RoundMode::CAST_NONE, this->processDataNum);
            Adds(tmp1, tmp1, neg_min, this->processDataNum);
        }
        Muls(tmp1, tmp1, float_bins, this->processDataNum);
        Duplicate(tmp2, (float)interval, this->processDataNum);
        Div(tmp2, tmp1, tmp2, this->processDataNum);
        Cast(tmp1, tmp2, RoundMode::CAST_FLOOR, this->processDataNum);
        for(int32_t i=0; i<this->bins; i++)
        {
            Duplicate(tmp2, (float)i, this->processDataNum);
            Compare(bits, tmp2, tmp1, CMPMODE::EQ, (this->processDataNum+255)/256*256);
            Select(tmp2, bits, zero, float(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum);
            ReduceSum(tmp2, tmp2, buff, this->processDataNum);
            var1 = tmp2.GetValue(0);
            if(progress == 0)
                yGm.SetValue(i, (DTYPE_Y)(var1));
            else
                yGm.SetValue(i, (float)yGm.GetValue(i) + (var1));
        }

        Duplicate(tmp2, (float)this->max, this->processDataNum);
        if constexpr (std::is_same_v<DTYPE_X, float>)
        {
            Compare(bits, tmp2, xLocal, CMPMODE::EQ, (this->processDataNum+255)/256*256);
        }
        else
        {
            Cast(tmp1, xLocal, RoundMode::CAST_NONE, this->processDataNum);
            Compare(bits, tmp2, tmp1, CMPMODE::EQ, (this->processDataNum+255)/256*256);
        }
        Select(tmp2, bits, zero, float(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, this->processDataNum);
        ReduceSum(tmp2, tmp2, buff, this->processDataNum);
        var1 = tmp2.GetValue(0);
        yGm.SetValue(this->bins-1, (float)yGm.GetValue(this->bins-1) + (var1));

        inQueueX.FreeTensor(xLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TBuf<QuePosition::VECCALC> QueueTmp1, QueueTmp2, QueueBuff, QueueBits, QueueZero;

    GlobalTensor<DTYPE_X> xGm;
    GlobalTensor<DTYPE_Y> yGm;

    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
    int bins;
    float min;
    float max;
};
extern "C" __global__ __aicore__ void histogram(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    KernelHistogram<DTYPE_X, DTYPE_Y> op;
    op.Init(x, y, 
            tiling_data.CoreDataNum, tiling_data.finalTileNum, tiling_data.tileDataNum, tiling_data.TailDataNum, 
            tiling_data.bins, tiling_data.min, tiling_data.max);  
    op.Process();
}