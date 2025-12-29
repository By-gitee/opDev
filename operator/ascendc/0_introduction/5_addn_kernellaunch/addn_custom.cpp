/**
 * @file addn_custom.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"

constexpr int32_t TILE_NUM = 8;   // split data into 8 tiles for each core
constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint64_t totalLength)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileLength = this->blockLength / TILE_NUM / BUFFER_NUM;
        xGm.SetGlobalBuffer((__gm__ half *)x + blockLength * AscendC::GetBlockIdx(), blockLength);
        yGm.SetGlobalBuffer((__gm__ half *)y + blockLength * AscendC::GetBlockIdx(), blockLength);
        zGm.SetGlobalBuffer((__gm__ half *)z + blockLength * AscendC::GetBlockIdx(), blockLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(half));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, tileLength * sizeof(half));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, tileLength * sizeof(half));
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = TILE_NUM * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        AscendC::LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();
        AscendC::DataCopy(xLocal, xGm[progress * tileLength], tileLength);
        AscendC::DataCopy(yLocal, yGm[progress * tileLength], tileLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        AscendC::LocalTensor<half> yLocal = inQueueY.DeQue<half>();
        AscendC::LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>();
        AscendC::Add(zLocal, xLocal, yLocal, tileLength);
        outQueueZ.EnQue<half>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<half> zLocal = outQueueZ.DeQue<half>();
        AscendC::DataCopy(zGm[progress * tileLength], zLocal, tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<half> xGm;
    AscendC::GlobalTensor<half> yGm;
    AscendC::GlobalTensor<half> zGm;
    uint32_t blockLength;
    uint32_t tileLength;
};

// assuming the input tensor is a list of two tensors, each with 2-dimensions shape
constexpr int32_t SHAPE_DIM = 2;
constexpr int32_t TENSOR_DESC_NUM = 2;

extern "C" __global__ __aicore__ void addn_custom(GM_ADDR srcList, GM_ADDR dst)
{
    AscendC::ListTensorDesc listTensorDesc((reinterpret_cast<__gm__ void *>(srcList)),
                                           (1 + (1 + SHAPE_DIM + 1) * TENSOR_DESC_NUM) * sizeof(uint64_t),
                                           TENSOR_DESC_NUM);

    uint64_t buf[SHAPE_DIM] = {0};
    AscendC::TensorDesc<int32_t> tensorDesc;
    tensorDesc.SetShapeAddr(buf);
    listTensorDesc.GetDesc(tensorDesc, 0);
    uint64_t totalLength = tensorDesc.GetShape(0) * tensorDesc.GetShape(1);
    __gm__ uint8_t *x = listTensorDesc.GetDataPtr<__gm__ uint8_t>(0);
    __gm__ uint8_t *y = listTensorDesc.GetDataPtr<__gm__ uint8_t>(1);

    KernelAdd op;
    op.Init(x, y, dst, totalLength);
    op.Process();
}
