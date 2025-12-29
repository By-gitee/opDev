/**
 * @file matmul_custom_tiling.h
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef MATMUL_TILING_H
#define MATMUL_TILING_H

#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(MatmulCustomTilingData)
TILING_DATA_FIELD_DEF_STRUCT(TCubeTiling, cubeTilingData);
TILING_DATA_FIELD_DEF(uint32_t, mTile); // L2上M切分次数
TILING_DATA_FIELD_DEF(uint32_t, nTile); // L2上N切分次数
TILING_DATA_FIELD_DEF(uint32_t, mTileBlock); // 单次切分运算的baseM个数
TILING_DATA_FIELD_DEF(uint32_t, nTileBlock); // 单次切分运算的baseN个数
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(MatmulCustom, MatmulCustomTilingData)
} // namespace optiling
#endif