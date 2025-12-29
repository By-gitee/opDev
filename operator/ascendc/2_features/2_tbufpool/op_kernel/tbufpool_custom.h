/**
 * @file tbufpool_custom.h
 *
 * Copyright (C) 2024-2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef TBUFPOOL_CUSTOM_H
#define TBUFPOOL_CUSTOM_H
#include "../op_host/tbufpool_custom_tiling.h"
#include "kernel_operator.h"


namespace MyCustomKernel {
constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t BUFFER_NUM_T1 = 1;
constexpr int32_t BUFFER_NUM_T2 = 1;
constexpr int32_t BUFFER_LENGTH = 4096 * sizeof(float);
constexpr int32_t BUFF_POOL_LENGTH = 2048 * sizeof(float);
constexpr int32_t INIT_TENSOR_LENGTH = 1024 * sizeof(float);
constexpr int32_t COMPUTE_LENGTH = 1024;

class TbufPoolImpl {
    public:
        __aicore__ inline TbufPoolImpl() {}
        __aicore__ inline void Init(__gm__ uint8_t* src0Gm, __gm__ uint8_t* src1Gm, __gm__ uint8_t* dstGm, 
                                     TbufPoolTilingData tiling, AscendC::TPipe* pipeIn)
        {
            pipe = pipeIn;
            src0Global.SetGlobalBuffer((__gm__ float*)src0Gm);
            src1Global.SetGlobalBuffer((__gm__ float*)src1Gm);
            dstGlobal.SetGlobalBuffer((__gm__ float*)dstGm);
            pipe->InitBufPool(tbufPool0, BUFFER_LENGTH);
            tbufPool0.InitBuffer(srcQue0, BUFFER_NUM, BUFF_POOL_LENGTH); // Total src0
            tbufPool0.InitBufPool(tbufPool1, BUFF_POOL_LENGTH);
            tbufPool0.InitBufPool(tbufPool2, BUFF_POOL_LENGTH, tbufPool1);
            tbufPool1.InitBuffer(srcQue1, BUFFER_NUM_T1, INIT_TENSOR_LENGTH);
            tbufPool1.InitBuffer(dstQue0, BUFFER_NUM_T1, INIT_TENSOR_LENGTH);
            tbufPool2.InitBuffer(srcQue2, BUFFER_NUM_T2, INIT_TENSOR_LENGTH);
            tbufPool2.InitBuffer(dstQue1, BUFFER_NUM_T2, INIT_TENSOR_LENGTH);
        }
        __aicore__ inline void Process()
        {
            //stage 1
            CopyIn();
            Compute();
            CopyOut();
            tbufPool1.Reset();
            //stage 2
            CopyIn1();
            Compute1();
            CopyOut1();
            tbufPool2.Reset();
            tbufPool0.Reset();
        }
  
    private:
        __aicore__ inline void CopyIn()
        {
            AscendC::LocalTensor<float> src0Local = srcQue0.AllocTensor<float>();
            AscendC::LocalTensor<float> src1Local = srcQue1.AllocTensor<float>();
            AscendC::DataCopy(src0Local, src0Global, COMPUTE_LENGTH);
            AscendC::DataCopy(src1Local, src1Global, COMPUTE_LENGTH);
            srcQue0.EnQue(src0Local);
            srcQue1.EnQue(src1Local);
        }
        __aicore__ inline void Compute()
        {
            AscendC::LocalTensor<float> src0Local = srcQue0.DeQue<float>();
            AscendC::LocalTensor<float> src1Local = srcQue1.DeQue<float>();
            AscendC::LocalTensor<float> dstLocal = dstQue0.AllocTensor<float>();
            AscendC::Add(dstLocal, src0Local, src1Local, COMPUTE_LENGTH);
            dstQue0.EnQue<float>(dstLocal);
            srcQue0.FreeTensor(src0Local);
            srcQue1.FreeTensor(src1Local);
        }
        __aicore__ inline void CopyOut()
        {
            AscendC::LocalTensor<float> dstLocal = dstQue0.DeQue<float>();
            AscendC::DataCopy(dstGlobal, dstLocal, COMPUTE_LENGTH);
            dstQue0.FreeTensor(dstLocal);
        }
        __aicore__ inline void CopyIn1()
        {
            AscendC::LocalTensor<float> src0Local = srcQue0.AllocTensor<float>();
            AscendC::LocalTensor<float> src1Local = srcQue2.AllocTensor<float>();
            AscendC::DataCopy(src0Local, src0Global[COMPUTE_LENGTH], COMPUTE_LENGTH);
            AscendC::DataCopy(src1Local, src1Global[COMPUTE_LENGTH], COMPUTE_LENGTH);
            srcQue0.EnQue(src0Local);
            srcQue2.EnQue(src1Local);
        }
        __aicore__ inline void Compute1()
        {
            AscendC::LocalTensor<float> src0Local = srcQue0.DeQue<float>();
            AscendC::LocalTensor<float> src1Local = srcQue2.DeQue<float>();
            AscendC::LocalTensor<float> dstLocal = dstQue1.AllocTensor<float>();
            AscendC::Sub(dstLocal, src0Local, src1Local, COMPUTE_LENGTH);
            dstQue1.EnQue<float>(dstLocal);
            srcQue0.FreeTensor(src0Local);
            srcQue2.FreeTensor(src1Local);
        }
        __aicore__ inline void CopyOut1()
        {
            AscendC::LocalTensor<float> dstLocal = dstQue1.DeQue<float>();
            AscendC::DataCopy(dstGlobal[COMPUTE_LENGTH], dstLocal, COMPUTE_LENGTH);
            dstQue1.FreeTensor(dstLocal);
        }
    private:
        AscendC::TPipe* pipe;
        AscendC::TBufPool<AscendC::TPosition::VECCALC> tbufPool0; 
        AscendC::TBufPool<AscendC::TPosition::VECCALC> tbufPool1; 
        AscendC::TBufPool<AscendC::TPosition::VECCALC> tbufPool2;
        AscendC::TQue<AscendC::TPosition::VECIN, 1> srcQue0; 
        AscendC::TQue<AscendC::TPosition::VECIN, 1> srcQue1; 
        AscendC::TQue<AscendC::TPosition::VECIN, 1> srcQue2;
        AscendC::TQue<AscendC::TPosition::VECOUT, 1> dstQue0; 
        AscendC::TQue<AscendC::TPosition::VECOUT, 1> dstQue1;
        AscendC::GlobalTensor<float> src0Global; 
        AscendC::GlobalTensor<float> src1Global; 
        AscendC::GlobalTensor<float> dstGlobal;
    };
}// namespace MyCustomKernel

#endif  // TBUFPOOL_CUSTOM_H