/**
* Copyright (c) 2025 Huawei Technologies Co., Ltd.
* This program is free software, you can redistribute it and/or modify it under the terms and conditions of
* CANN Open Software License Agreement Version 2.0 (the "License").
* Please refer to the License for details. You may not use this file except in compliance with the License.
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
* INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
* See LICENSE in the root of the software repository for the full text of the License.
*/

#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;                                     // tensor num for each queue

template <typename T>
class KernelScatter {
public:
    __aicore__ inline KernelScatter() {}
    __aicore__ inline void Init(GM_ADDR srcGm, GM_ADDR dstOffsetGm, GM_ADDR dstGm, uint32_t count)
    {
        mElementCount = count;
        xGm.SetGlobalBuffer((__gm__ half *)srcGm, mElementCount);
        yGm.SetGlobalBuffer((__gm__ uint32_t *)dstOffsetGm, mElementCount);
        zGm.SetGlobalBuffer((__gm__ half *)dstGm, mElementCount);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, mElementCount * sizeof(half));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, mElementCount * sizeof(uint32_t));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, mElementCount * sizeof(half));
    }
    __aicore__ inline void Process()
    {
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        AscendC::LocalTensor<uint32_t> yLocal = inQueueY.AllocTensor<uint32_t>();
        AscendC::DataCopy(xLocal, xGm, mElementCount);
        AscendC::DataCopy(yLocal, yGm, mElementCount);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        AscendC::LocalTensor<uint32_t> yLocal = inQueueY.DeQue<uint32_t>();
        AscendC::LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>();
        for (uint32_t i = 0; i < mElementCount; ++i) {
            auto offset = yLocal.GetValue(i) / sizeof(T);
            auto srcValue = xLocal.GetValue(i);
            zLocal.SetValue(offset, srcValue);
        }
        outQueueZ.EnQue<half>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<half> zLocal = outQueueZ.DeQue<half>();
        AscendC::DataCopy(zGm, zLocal, mElementCount);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<half> xGm;
    AscendC::GlobalTensor<uint32_t> yGm;
    AscendC::GlobalTensor<half> zGm;
    uint32_t mElementCount;
};
                                                                             
extern "C" __global__ __aicore__ void scatter_custom(GM_ADDR srcGm, GM_ADDR dstOffsetGm, GM_ADDR dstGm)   
{
    uint32_t count = 128;                                                                                                
    KernelScatter<half> op;                                                                                
    op.Init(srcGm, dstOffsetGm, dstGm, count);                                                            
    op.Process();                                                                                         
}

#ifndef ASCENDC_CPU_DEBUG
void scatter_custom_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y, uint8_t *z)
{
    scatter_custom<<<blockDim, nullptr, stream>>>(x, y, z);
}
#endif