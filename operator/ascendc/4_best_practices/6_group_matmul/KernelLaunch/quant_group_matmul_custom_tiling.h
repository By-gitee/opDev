/**
 * @file quant_group_matmul_custom_tiling.h
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef QUANT_GROUP_MATMUL_CUSTOM_TILING_H
#define QUANT_GROUP_MATMUL_CUSTOM_TILING_H

#include "kernel_tiling/kernel_tiling.h"


struct QuantGroupMatmulCustomTilingData
{
    uint32_t coreNum;
    uint32_t groupNum;
    uint32_t totalInGroup;
    uint32_t k;
    uint32_t n;
    uint32_t ubCalSize;
    uint32_t ubRestBytes;
    uint32_t parallNum;
    TCubeTiling mmTilingData;
};

#endif // QUANT_GROUP_MATMUL_CUSTOM_TILING_H