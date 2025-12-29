/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"

class KernelOnes {
public:
    __aicore__ inline KernelOnes() {}
    __aicore__ inline void Init(GM_ADDR z, uint32_t onesLength, uint32_t processTime, uint32_t lastLength)
    {
        this->onesLength = onesLength;
        this->processTime = processTime;
        this->lastLength = lastLength;

        outGm.SetGlobalBuffer((__gm__ int32_t *)z);
        pipe.InitBuffer(inOutQueue, BUFFER_NUM, this->onesLength * sizeof(int32_t));
    }
    __aicore__ inline void Process()
    {
        for (int32_t i = 0; i < this->processTime; i++) {
            Compute(i);
            CopyOut(i);
        }
    }

private:

    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<int32_t> xLocal = inOutQueue.AllocTensor<int32_t>();
        int32_t inputVal(1);
        AscendC::Duplicate<int32_t>(xLocal, inputVal, this->onesLength);
        inOutQueue.EnQue<int32_t>(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<int32_t> xLocal = inOutQueue.DeQue<int32_t>();
        if (progress == this->processTime - 1)
        {
            AscendC::DataCopy(outGm[progress * this->onesLength], xLocal, this->lastLength);
        }else{
            AscendC::DataCopy(outGm[progress * this->onesLength], xLocal, this->onesLength);
        }
        inOutQueue.FreeTensor(xLocal);
    }

private:
    static constexpr int32_t BUFFER_NUM = 2; 
    AscendC::TPipe pipe;
    AscendC::TQueBind<AscendC::TPosition::VECIN, AscendC::TPosition::VECOUT, BUFFER_NUM> inOutQueue;
    AscendC::GlobalTensor<int32_t> outGm;
    uint32_t onesLength;
    uint32_t processTime;
    uint32_t lastLength;
};


extern "C" __global__ __aicore__ void ones(GM_ADDR h, GM_ADDR w, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelOnes op;
    op.Init(out, tiling_data.onesLength, tiling_data.processTime, tiling_data.lastLength);
    op.Process();
}