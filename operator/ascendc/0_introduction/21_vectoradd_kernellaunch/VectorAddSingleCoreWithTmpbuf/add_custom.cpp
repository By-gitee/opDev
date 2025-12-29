/**
 * @file add_custom.cpp
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "kernel_operator.h"
constexpr uint32_t TOTAL_LENGTH = 2048;


class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z)
    {
        xGm.SetGlobalBuffer((__gm__ bfloat16_t *)x, TOTAL_LENGTH);
        yGm.SetGlobalBuffer((__gm__ bfloat16_t *)y, TOTAL_LENGTH);
        zGm.SetGlobalBuffer((__gm__ bfloat16_t *)z, TOTAL_LENGTH);

        pipe.InitBuffer(inQueueX, 1, TOTAL_LENGTH * sizeof(bfloat16_t));
        pipe.InitBuffer(inQueueY, 1, TOTAL_LENGTH * sizeof(bfloat16_t));
        pipe.InitBuffer(outQueueZ, 1, TOTAL_LENGTH * sizeof(bfloat16_t));

        pipe.InitBuffer(tmpBuf0, TOTAL_LENGTH * sizeof(float));
        pipe.InitBuffer(tmpBuf1, TOTAL_LENGTH * sizeof(float));
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
        AscendC::LocalTensor<bfloat16_t> xLocal = inQueueX.AllocTensor<bfloat16_t>();
        AscendC::LocalTensor<bfloat16_t> yLocal = inQueueY.AllocTensor<bfloat16_t>();

        AscendC::DataCopy(xLocal, xGm, TOTAL_LENGTH);
        AscendC::DataCopy(yLocal, yGm, TOTAL_LENGTH);

        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<bfloat16_t> xLocal = inQueueX.DeQue<bfloat16_t>();
        AscendC::LocalTensor<bfloat16_t> yLocal = inQueueY.DeQue<bfloat16_t>();
        AscendC::LocalTensor<bfloat16_t> zLocal = outQueueZ.AllocTensor<bfloat16_t>();

        AscendC::LocalTensor<float> tmpTensor0 = tmpBuf0.Get<float>();
        AscendC::LocalTensor<float> tmpTensor1 = tmpBuf1.Get<float>();

        AscendC::Cast(tmpTensor0, xLocal, AscendC::RoundMode::CAST_NONE, TOTAL_LENGTH);
        AscendC::Cast(tmpTensor1, yLocal, AscendC::RoundMode::CAST_NONE, TOTAL_LENGTH);

        AscendC::Add(tmpTensor0, tmpTensor0, tmpTensor1, TOTAL_LENGTH);
        AscendC::Cast(zLocal, tmpTensor0, AscendC::RoundMode::CAST_RINT, TOTAL_LENGTH); 

        outQueueZ.EnQue<bfloat16_t>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<bfloat16_t> zLocal = outQueueZ.DeQue<bfloat16_t>();
        AscendC::DataCopy(zGm, zLocal, TOTAL_LENGTH);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;

    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueZ;

    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf0;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf1;

    AscendC::GlobalTensor<bfloat16_t> xGm;
    AscendC::GlobalTensor<bfloat16_t> yGm;
    AscendC::GlobalTensor<bfloat16_t> zGm;
};

extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z)
{
    KernelAdd op;
    op.Init(x, y, z);
    op.Process();
}
