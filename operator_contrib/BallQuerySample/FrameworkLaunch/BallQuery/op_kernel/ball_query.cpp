/* Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved. 
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.
 * You may not use this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "kernel_operator.h"

#include "ball_query_norm_fp16.h"
#include "ball_query_norm_fp32_perf.h"
#include "ball_query_stack.h"

extern "C" __global__ __aicore__ void ball_query(
        GM_ADDR xyz, 
        GM_ADDR center_xyz, 
        GM_ADDR xyz_batch_cnt, 
        GM_ADDR center_xyz_batch_cnt, 
        GM_ADDR idx, 
        GM_ADDR workspace, 
        GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (TILING_KEY_IS(10000001)) {
        BallQueryNormFP16<half> op;
        op.Init(xyz, center_xyz, idx, &tiling_data);
        op.Process();
    }
    else if(TILING_KEY_IS(10000002)){
        BallQueryStack<half> op;
        op.Init(xyz, center_xyz, xyz_batch_cnt, center_xyz_batch_cnt, idx, &tiling_data);
        op.Process();
    }
    else if(TILING_KEY_IS(10000003)){
        BallQueryNormFP32Perf<float> op;
        op.Init(xyz, center_xyz, idx, &tiling_data);
        op.Process();
    }
    else if(TILING_KEY_IS(10000004)){
        BallQueryStack<float> op;
        op.Init(xyz, center_xyz, xyz_batch_cnt, center_xyz_batch_cnt, idx, &tiling_data);
        op.Process();
    }
}