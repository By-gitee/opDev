/**
* @file cumsum.cpp
*
* Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#include <type_traits>
#include "kernel_operator.h"

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

template<typename T> class KernelCumsum {
public:
    __aicore__ inline KernelCumsum() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR axis, GM_ADDR y, 
                            int32_t size, int32_t x_ndarray[], int32_t x_dimensional, 
                            bool exclusive, bool reverse, int32_t tileDataMaxNum) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

        this->x_dimensional = x_dimensional;
        this->exclusive = exclusive;
        this->reverse = reverse;
        this->tileDataMaxNum = tileDataMaxNum;

        axisGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_AXIS *>(axis), 1);

        for(int i = 0; i < x_dimensional; i++)
        {
            this->x_ndarray[i] = x_ndarray[i];
        }

        this->size =size;

        this->dim = axisGm.GetValue(0);
        if(this->dim < 0)
            this->dim += this->x_dimensional;

        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_X *>(x), size+32);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_Y *>(y), size+32);

        int32_t cycles = 1;
        int32_t interval = 1;
        int32_t loopCount = 1;

        for(int32_t i = 0; i < this->dim; i++)
        {
            loopCount *= this->x_ndarray[i];
        }

        cycles = this->x_ndarray[this->dim];
        
        for(int32_t i = this->dim+1; i < this->x_dimensional; i++)
        {
            interval *= this->x_ndarray[i];
        }

        this->cycles = cycles;
        this->interval = interval;
        this->loopCount = loopCount;

        this->circulate = interval > 0 ? interval / tileDataMaxNum : 0;
        this->SingleData = tileDataMaxNum;
        if (tileDataMaxNum != 0) {
            this->lastHoleData = interval % tileDataMaxNum;
        } else {
            // Handle the case where tileDataMaxNum is 0, for example:
            this->lastHoleData = 0; // or any other appropriate action
        }
        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileDataMaxNum * sizeof(float));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, tileDataMaxNum * sizeof(float));
        pipe.InitBuffer(QueueTemp, tileDataMaxNum * sizeof(float));

    }

    __aicore__ inline void Process() {
        if(this->exclusive == false && this->reverse == false)
        {
            if constexpr (std::is_same_v<T, float>)
            {
                if(interval % 8 == 0)
                {
                    LocalTensor<float> temp1 = QueueTemp.Get<float>();;

                    for(int32_t z = 0; z < loopCount; z++)
                    {
                        this->SingleData = this->tileDataMaxNum;

                        const int32_t PROCESS_DATA_PER_TIME = 3;
                        for(int32_t i = 0; i < cycles; i+=PROCESS_DATA_PER_TIME)
                        {
                            CopyIn(z, i, 0);
                            Compute(z, i, 0, temp1);
                            CopyOut(z, i, 0);
                        }
                    }
                }
                else
                {
                    for(int z = 0; z < loopCount; z++)
                    {
                        for(int i = 0; i < cycles; i++)
                        {
                            for(int j = 0; j < interval; j++)
                            {
                                if(i == 0)
                                {
                                    yGm.SetValue(z*cycles*interval+i*interval+j, xGm.GetValue(z*cycles*interval+i*interval+j));
                                }
                                else
                                {
                                    yGm.SetValue(z*cycles*interval+i*interval+j, (DTYPE_X)((float)yGm.GetValue(z*cycles*interval+(i-1)*interval+j) + (float)xGm.GetValue(z*cycles*interval+i*interval+j)));
                                }                               
                            }
                        }
                    }
                }
            }
            else
            {
                for(int z = 0; z < loopCount; z++)
                {
                    for(int i = 0; i < cycles; i++)
                    {
                        for(int j = 0; j < interval; j++)
                        {
                            if(i == 0)
                            {
                                yGm.SetValue(z*cycles*interval+i*interval+j, xGm.GetValue(z*cycles*interval+i*interval+j));
                            }
                            else
                            {
                                yGm.SetValue(z*cycles*interval+i*interval+j, (DTYPE_X)((float)yGm.GetValue(z*cycles*interval+(i-1)*interval+j) + (float)xGm.GetValue(z*cycles*interval+i*interval+j)));
                            }                            
                        }
                    }
                }
            }
        }
        else if(this->exclusive == true && this->reverse == false)
        {
            for(int z = 0; z < loopCount; z++)
            {
                for(int i = 0; i < cycles; i++)
                {
                    for(int j = 0; j < interval; j++)
                    {
                        if(i == 0)
                        {
                            yGm.SetValue(z*cycles*interval+i*interval+j, 0);
                        }
                        else
                        {
                            yGm.SetValue(z*cycles*interval+i*interval+j, (DTYPE_X)((float)yGm.GetValue(z*cycles*interval+(i-1)*interval+j) + (float)xGm.GetValue(z*cycles*interval+(i-1)*interval+j)));
                        }                        
                    }
                }
            }
        }
        else if(this->exclusive == false && this->reverse == true)
        {
            for(int z = 0; z < loopCount; z++)
            {
                for(int i = cycles -1; i >= 0; i--)
                {
                    for(int j = interval-1; j >= 0; j--)
                    {
                        if(i == cycles -1)
                        {
                            yGm.SetValue(z*cycles*interval+i*interval+j, xGm.GetValue(z*cycles*interval+i*interval+j));
                        }
                        else
                        {
                            yGm.SetValue(z*cycles*interval+i*interval+j, (DTYPE_X)((float)yGm.GetValue(z*cycles*interval+(i+1)*interval+j) + (float)xGm.GetValue(z*cycles*interval+i*interval+j)));
                        }
                    }
                }
            }
        }
        else
        {
            for(int z = 0; z < loopCount; z++)
            {
                for(int i = cycles -1; i >= 0; i--)
                {
                    for(int j = interval-1; j >= 0; j--)
                    {
                        if(i == cycles -1)
                        {
                            yGm.SetValue(z*cycles*interval+i*interval+j, 0);
                        }
                        else
                        {
                            yGm.SetValue(z*cycles*interval+i*interval+j, (DTYPE_X)((float)yGm.GetValue(z*cycles*interval+(i+1)*interval+j) + (float)xGm.GetValue(z*cycles*interval+(i+1)*interval+j)));
                        }
                    }
                }
            }
        }  
    }

private:
     __aicore__ inline void CopyIn(int32_t i, int32_t j, int32_t z)
    {
        LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        if((this->cycles-1) == j)
            DataCopy(xLocal, xGm[i*cycles*interval+j*interval + z*this->tileDataMaxNum], this->interval);
        else if((this->cycles-2) == j)
            DataCopy(xLocal, xGm[i*cycles*interval+j*interval + z*this->tileDataMaxNum], this->interval*2);
        else
            DataCopy(xLocal, xGm[i*cycles*interval+j*interval + z*this->tileDataMaxNum], this->interval*3);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t i, int32_t j, int32_t z, LocalTensor<DTYPE_Y> temp1)
    {
        LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
        if( j == 0)
        {
            Duplicate(temp1, static_cast<DTYPE_X>(0), this->interval);
        }

        if constexpr (std::is_same_v<T, float>)
        {
            if((this->cycles-1) == j)
            {
                Add(temp1,temp1,xLocal,this->interval);
                Adds(yLocal,temp1,static_cast<DTYPE_X>(0),this->interval);       
            }
            else if((this->cycles-2) == j)
            {
                Add(yLocal,temp1,xLocal,this->interval);
                Add(yLocal[this->interval],yLocal,xLocal[this->interval],this->interval);
                Adds(temp1,yLocal[this->interval],static_cast<DTYPE_X>(0),this->interval);  
            }
            else
            {
                Add(yLocal,temp1,xLocal,this->interval);
                Add(yLocal[this->interval],yLocal,xLocal[this->interval],this->interval);
                Add(yLocal[this->interval*2], yLocal[this->interval], xLocal[this->interval*2],this->interval);
                Adds(temp1,yLocal[this->interval*2],static_cast<DTYPE_X>(0),this->interval);  
            }
        }

        outQueueY.EnQue<DTYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t i, int32_t j, int32_t z)
    {
        LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();

        if((this->cycles-1) == j)
        {
            DataCopy(yGm[i*cycles*interval + j*interval + z*this->tileDataMaxNum ], yLocal, this->interval);
        }
        else if((this->cycles-2) == j)
        {
            DataCopy(yGm[i*cycles*interval + j*interval + z*this->tileDataMaxNum ], yLocal, this->interval*2);
        }
        else
        {
            DataCopy(yGm[i*cycles*interval + j*interval + z*this->tileDataMaxNum ], yLocal, this->interval*3);
        }

        outQueueY.FreeTensor(yLocal);
    }
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    TBuf<QuePosition::VECCALC> QueueTemp;

    GlobalTensor<DTYPE_X> xGm;
    GlobalTensor<DTYPE_AXIS> axisGm;
    GlobalTensor<DTYPE_Y> yGm;

    int32_t x_ndarray[10];
    int32_t x_dimensional;
    int32_t dim;
    bool exclusive;
    bool reverse;
    int32_t size;

    int32_t cycles;
    int32_t interval;
    int32_t loopCount;

    int32_t tileDataMaxNum;
    
    int32_t circulate;
    int32_t SingleData;
    int32_t lastHoleData;
};

extern "C" __global__ __aicore__ void cumsum(GM_ADDR x, GM_ADDR axis, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    KernelCumsum<DTYPE_X> op;
    op.Init(x, axis, y, 
            tiling_data.size, tiling_data.x_ndarray, tiling_data.x_dimensional, 
            tiling_data.exclusive, tiling_data.reverse, tiling_data.tileDataMaxNum);
    op.Process();
}