/**
 * @file group_barrier.cpp
 *
 * Copyright (C) 2022-2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "kernel_operator.h"
constexpr int32_t BUFFER_NUM = 1; // tensor num for each queue
constexpr int32_t ARRIVE_NUM = 2; // arriver num
constexpr int32_t WAIT_NUM = 6; // wait num
constexpr int32_t SIZE_NUM = 1; // tensor size
constexpr int32_t TEMP = ARRIVE_NUM * WAIT_NUM; // initial number

class KernelGroupBarrier {
public:
    __aicore__ inline KernelGroupBarrier() {}
    __aicore__ inline void Init(GM_ADDR barworkspace, GM_ADDR out)
    {
        this->barworkspace = barworkspace;
        uint32_t oneBlockNum = 32 / sizeof(int32_t);
        this->totalLength = (SIZE_NUM + oneBlockNum - 1) / oneBlockNum * oneBlockNum;
        output_global.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(out), totalLength);
        pipe.InitBuffer(inQueueX, 1, sizeof (int32_t) *  totalLength);
        pipe.InitBuffer(outQueue, 1, sizeof (int32_t) *  totalLength);
    }
    __aicore__ inline void Process()
    {
        AscendC::GroupBarrier<AscendC::PipeMode::MTE3_MODE> barA(barworkspace, ARRIVE_NUM, WAIT_NUM); // apply GroupBarrier
        AscendC::InitOutput<int32_t> (output_global, totalLength, 0); // init output zero
        AscendC::SyncAll(); // sync all
        auto id = AscendC::GetBlockIdx();
        if (id >= 0 && id < ARRIVE_NUM) // arrive realize
        {
            AscendC::LocalTensor<int32_t> input_local = inQueueX.AllocTensor<int32_t> ();
            AscendC::SetAtomicNone();
            input_local.SetValue(0, TEMP); // set initial number
            AscendC::SetAtomicAdd<int32_t>(); // add initial number
            AscendC::DataCopy(output_global, input_local, totalLength);
            AscendC::SetAtomicNone();
            inQueueX.FreeTensor(input_local);
            barA.Arrive(id); //arrive end
        }
        else if (id >= ARRIVE_NUM && id < ARRIVE_NUM + WAIT_NUM)  // wait realize
        {
            barA.Wait(id - ARRIVE_NUM); // wait end
            AscendC::PRINTF("OUTPUT = %d\n", output_global.GetValue(0));    
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueue;
    AscendC::GlobalTensor<int32_t> input_global;
    AscendC::GlobalTensor<int32_t> output_global;
    uint32_t totalLength;
    GM_ADDR barworkspace;
};

extern "C" __global__ __aicore__ void group_barrier(GM_ADDR barworkspace, GM_ADDR out, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelGroupBarrier op;
    op.Init(barworkspace, out);
    op.Process();
}