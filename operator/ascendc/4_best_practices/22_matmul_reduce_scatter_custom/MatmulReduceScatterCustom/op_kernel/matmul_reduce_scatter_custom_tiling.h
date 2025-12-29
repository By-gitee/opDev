/**
 * @file matmul_reduce_scatter_custom_tiling.h
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef __MATMUL_REDUCE_SCATTER_TILING_H__
#define __MATMUL_REDUCE_SCATTER_TILING_H__

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

struct ReduceScatterRCSTiling {
    uint32_t rankDim;
    uint32_t rankM;
    uint32_t rankK;
    uint32_t rankN;
    uint32_t isTransposeA;
    uint32_t isTransposeB;
    uint32_t tileCnt;
    uint32_t tailM;
    uint32_t tailCnt;
    uint8_t determinism;
    uint32_t dataType;
};

class MatmulReduceScatterCustomTilingData {
public:
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    TCubeTiling matmulTiling;
    TCubeTiling tailTiling;
    ReduceScatterRCSTiling param;
};

#endif //__MATMUL_REDUCE_SCATTER_TILING_H__