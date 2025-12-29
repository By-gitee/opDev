/**
 * @file mmad_custom.cpp
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifdef CUSTOM_ASCEND310P
#include "mmad_custom.h"
#else
#include "mmad_custom_cube_only.h"
#endif

extern "C" __global__ __aicore__ void mmad_custom(GM_ADDR a, GM_ADDR b, GM_ADDR c)
{
    KernelMmad op;
    op.Init(a, b, c);
    op.Process();
}
