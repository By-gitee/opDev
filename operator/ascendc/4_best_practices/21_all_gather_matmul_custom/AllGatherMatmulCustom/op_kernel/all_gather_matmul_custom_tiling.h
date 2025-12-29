/**
 * @file all_gather_matmul_custom_tiling.h
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef __ALL_GATHER_MATMUL_CUSTOM_TILING_H__
#define __ALL_GATHER_MATMUL_CUSTOM_TILING_H__

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

struct AllGatherMatmulTiling {
    uint32_t rankM;
    uint32_t rankN;
    uint32_t rankK;
    uint32_t tileNum;
    uint32_t tailM;
    uint32_t tailNum;
};

class AllGatherMatmulCustomTilingData {
public:
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    TCubeTiling localTiling;
    TCubeTiling tileTiling;
    TCubeTiling tailTiling;
    AllGatherMatmulTiling cfg;
};

#endif //__ALL_GATHER_MATMUL_CUSTOM_TILING_H__
