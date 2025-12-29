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
#include "depth_to_space_NCHW.cpp"
#include "depth_to_space_NHWC.cpp"
#include "depth_to_space_NHWC_NO_C1.cpp"

extern "C" __global__ __aicore__ void depth_to_space(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    if (TILING_KEY_IS(10000001)) {
        DepthToSpaceNHWC<DTYPE_X> op;
        op.Init(x, y, &tiling_data);
        op.Process();
    } else if (TILING_KEY_IS(10000002)) {
        DepthToSpaceNCHW<DTYPE_X> op;
        op.Init(x, y, &tiling_data);
        op.Process();
    } else if (TILING_KEY_IS(10000003)) {
        DepthToSpaceNHWC_NO_C1<DTYPE_X> op;
        op.Init(x, y, &tiling_data);
        op.Process();
    }
}