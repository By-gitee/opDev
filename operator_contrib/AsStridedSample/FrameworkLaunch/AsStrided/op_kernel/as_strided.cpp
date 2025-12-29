/**
* @file as_strided.cpp
*
* Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#include <type_traits>
#include "kernel_operator.h"

#define K_MAX_SHAPE_DIM 0

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 1;

template<typename T> class KernelAsStrided {
public:
    __aicore__ inline KernelAsStrided() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR size, GM_ADDR stride, GM_ADDR storage_offset, GM_ADDR y, 
                            int32_t size_dimensional) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->size_dimensional = size_dimensional;
        sizeGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_SIZE *>(size), size_dimensional);
        strideGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_STRIDE *>(stride), size_dimensional);
        storage_offsetGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_STORAGE_OFFSET *>(storage_offset), 1);

        this->y_size = 1;
        this->x_size = 0;
        for(int i = 0; i < size_dimensional; i++)
        {
            this->size_nd[i] = sizeGm.GetValue(i);
            this->stride_nd[i] = strideGm.GetValue(i);
            this->y_size *= this->size_nd[i];
            this->x_size += this->size_nd[i] * this->stride_nd[i];
        }
        this->storage_nd = storage_offsetGm.GetValue(0);

        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_X *>(x), this->x_size);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_Y *>(y), this->y_size);
        
        this->x_size = (this->x_size+31)/32*32;

        pipe.InitBuffer(inQueueX, 1, this->x_size * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, (this->y_size+31)/32*32*2 * sizeof(DTYPE_Y));
        pipe.InitBuffer(QueueTemp1, 64 * sizeof(uint32_t));
        pipe.InitBuffer(QueueTemp2, 4 * sizeof(uint32_t));
        pipe.InitBuffer(QueueTemp3, 64 * sizeof(DTYPE_Y));
    }
    __aicore__ inline void Process() {
            if(size_dimensional == 2)
            {
                LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
                LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
                auto mask_scl1 = QueueTemp1.Get<uint32_t>();

                DataCopy(xLocal, xGm[this->storage_nd], this->x_size);

                int loop = this->size_nd[0];
                int32_t count = 0;
                int input_index = 0;
                uint64_t rsvdCnt;
                int z_stride_nd;
                int32_t mask_num = this->size_nd[0]*this->stride_nd[1];
                Duplicate(mask_scl1, 0xFFFFFFFF, mask_num/32+2);
                mask_scl1.SetValue(0,  0xFFFFFFFF<<this->stride_nd[0]);

                int i = 0;
                int z = 0;
                for (; i <this->stride_nd[1]; i++) 
                {
                    z_stride_nd = input_index;
                    for (z=0; z < this->size_nd[1]; z++) 
                    {
                        yLocal.SetValue(count, xGm.GetValue(z_stride_nd+storage_offsetGm.GetValue(0)));
                        count++;
                        z_stride_nd += this->stride_nd[1];
                    }
                    input_index += (this->stride_nd[0]);
                }
                
                for (; i <loop; i++) 
                {
                    z_stride_nd = input_index;
                    if(i%this->stride_nd[1] == 0)
                    {
                        if((i+this->stride_nd[1] )< loop)
                            GatherMask(yLocal[this->size_nd[1]*i], yLocal[this->size_nd[1]*(i-this->stride_nd[1])], mask_scl1, true, mask_num, { 1, 1, 8, 8 }, rsvdCnt);
                        else
                            GatherMask(yLocal[this->size_nd[1]*i], yLocal[this->size_nd[1]*(i-this->stride_nd[1])], mask_scl1, true, (loop-i)*this->size_nd[0], { 1, 1, 8, 8 }, rsvdCnt);
                    }
                    z = this->size_nd[1] - this->stride_nd[0];
                    count+=z;
                    z_stride_nd += z*this->stride_nd[1];
                    for (; z < this->size_nd[1]; z++) 
                    {
                        yLocal.SetValue(count, xLocal.GetValue(z_stride_nd));
                        count++;
                        z_stride_nd += this->stride_nd[1];
                    }
                    input_index += (this->stride_nd[0]);
                }
                DataCopy(yGm, yLocal, (y_size+31)/32*32);
                outQueueY.FreeTensor(yLocal);
                inQueueX.FreeTensor(xLocal);
            }
            else
            {
                LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
                int loop = this->y_size / this->size_nd[size_dimensional - 1];

                for (int i = 0; i <loop; i++) 
                {
                    int input_index = this->storage_nd;
                    int temp = i;
                    for (int j = size_dimensional - 2; j >=0 ; j--)
                    {
                        int index = temp % this->size_nd[j];
                        input_index += (index * strideGm.GetValue(j));
                        temp = temp / this->size_nd[j];
                        if(temp == 0) break;
                    }
                    int z_stride_nd = 0;
                    int temp1 = i*this->size_nd[size_dimensional - 1];
                    for (int z = 0; z < this->size_nd[size_dimensional - 1]; z++) 
                    {
                        yLocal.SetValue(temp1 + z, xGm.GetValue(input_index + z_stride_nd));
                        z_stride_nd += strideGm.GetValue(size_dimensional - 1);
                    }
                }
                DataCopy(yGm, yLocal, (y_size+31)/32*32);
                outQueueY.FreeTensor(yLocal);
            }
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> QueueTemp1, QueueTemp2, QueueTemp3;

    GlobalTensor<DTYPE_X> xGm;
    GlobalTensor<DTYPE_SIZE> sizeGm;
    GlobalTensor<DTYPE_STRIDE> strideGm;
    GlobalTensor<DTYPE_STORAGE_OFFSET> storage_offsetGm;
    GlobalTensor<DTYPE_Y> yGm;
    int32_t size_dimensional;
    int32_t x_size;
    int32_t y_size;
    int size_nd[10];
    int stride_nd[10];
    int storage_nd;
};

extern "C" __global__ __aicore__ void as_strided(GM_ADDR x, GM_ADDR size, GM_ADDR stride, GM_ADDR storage_offset, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    KernelAsStrided<DTYPE_X> op;
    op.Init(x, size, stride, storage_offset, y, 
                tiling_data.size_dimensional);
    op.Process();
}