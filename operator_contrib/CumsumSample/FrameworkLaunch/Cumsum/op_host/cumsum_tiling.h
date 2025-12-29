/**
* @file cumsum_tiling.h
*
* Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#ifndef CUMSUM_TILING_H
#define CUMSUM_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(CumsumTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, size);
  TILING_DATA_FIELD_DEF_ARR(int32_t, 10, x_ndarray);
  TILING_DATA_FIELD_DEF(int32_t, x_dimensional);
  TILING_DATA_FIELD_DEF(bool, exclusive);
  TILING_DATA_FIELD_DEF(bool, reverse);
  TILING_DATA_FIELD_DEF(int32_t, tileDataMaxNum);
  
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Cumsum, CumsumTilingData)
}

#endif // TILINGDATA_BASE_H