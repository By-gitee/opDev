/**
 * @file adds_custom.cpp
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "kernel_operator.h"
#include "adds_custom_v1.h"
#include "adds_custom_v2.h"
#include "adds_custom_v3.h"

extern "C" __global__ __aicore__ void adds_custom(GM_ADDR x, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AddsCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    if (TILING_KEY_IS(1UL)) {
        KernelAddsV1 op;
        op.Init(x, z, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(2UL)) {
        KernelAddsV2 op;
        op.Init(x, z, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(3UL)) {
        KernelAddsV3 op;
        op.Init(x, z, &tilingData);
        op.Process();
    }
}
