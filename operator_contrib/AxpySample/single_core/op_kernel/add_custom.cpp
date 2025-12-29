/**
 * @file add_custom.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "kernel_operator.h"

constexpr int32_t  BUFFER_SIZE = 2048;
constexpr int32_t  ITERATION_COUNT = 55;
class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,uint32_t processNum,uint32_t tileNum)
    {
        xGm.SetGlobalBuffer((__gm__ half *)x  , processNum*tileNum);
        yGm.SetGlobalBuffer((__gm__ half *)y , processNum*tileNum);
        zGm.SetGlobalBuffer((__gm__ half *)z ,processNum*tileNum);
        
        pipe.InitBuffer(inQueueX, 1, BUFFER_SIZE * sizeof(half));
        pipe.InitBuffer(inQueueY, 1, BUFFER_SIZE * sizeof(half));
        pipe.InitBuffer(outQueueZ, 1, BUFFER_SIZE * sizeof(half));
        this->tileNum=tileNum;
        this->processNum=processNum;
    }
    __aicore__ inline void Process()
    {
        for (int32_t i = 0; i < this->tileNum; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(const int32_t progress)
    {
        AscendC::LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        AscendC::LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();
        AscendC::DataCopy(xLocal, xGm[progress * BUFFER_SIZE], BUFFER_SIZE);
        AscendC::DataCopy(yLocal, yGm[progress * BUFFER_SIZE], BUFFER_SIZE);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute(const int32_t progress)
    {
        AscendC::LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        AscendC::LocalTensor<half> yLocal = inQueueY.DeQue<half>();
        AscendC::LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>();
        
        for(int32_t start=0;start<ITERATION_COUNT;start++){
        AscendC::Muls(xLocal,xLocal,(half)1.0,this->processNum);
        AscendC::Add(xLocal,xLocal,yLocal,this->processNum);
        }
        AscendC::Muls(zLocal,xLocal,(half)1.0,this->processNum);
        
        outQueueZ.EnQue<half>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(const int32_t progress)
    {
        AscendC::LocalTensor<half> zLocal = outQueueZ.DeQue<half>();
        AscendC::DataCopy(zGm[progress * BUFFER_SIZE], zLocal, BUFFER_SIZE);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueueZ;
    AscendC::GlobalTensor<half> xGm;
    AscendC::GlobalTensor<half> yGm;
    AscendC::GlobalTensor<half> zGm;
    uint32_t tileNum;
    uint32_t processNum;
};

extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, 
                                                 GM_ADDR y, 
                                                 GM_ADDR z,
                                                 GM_ADDR workspace, 
                                                 GM_ADDR tiling){
GET_TILING_DATA(tiling_data, tiling);
    KernelAdd op;
    op.Init(x, y, z,tiling_data.processNum,tiling_data.tileNum);
    op.Process();
}