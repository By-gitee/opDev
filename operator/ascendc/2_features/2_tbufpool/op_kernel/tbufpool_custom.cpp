/**
 * @file tbufpool_custom.cpp
 *
 * Copyright (C) 2024-2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

#include "./tbufpool_custom.h"
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void tbufpool_custom(GM_ADDR src0Gm, GM_ADDR src1Gm, GM_ADDR dstGm, TbufPoolTilingData tiling)
{
    AscendC::TPipe pipe;
    MyCustomKernel::TbufPoolImpl op;
    op.Init(src0Gm, src1Gm, dstGm, tiling, &pipe);
    op.Process();
}