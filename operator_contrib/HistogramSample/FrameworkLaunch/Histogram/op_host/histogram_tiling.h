/**
* @file histogram_tiling.cpp
*
* Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(HistogramTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, CoreDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, finalTileNum);
  TILING_DATA_FIELD_DEF(uint32_t, tileDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, TailDataNum);
  TILING_DATA_FIELD_DEF(int, bins);
  TILING_DATA_FIELD_DEF(float, min);
  TILING_DATA_FIELD_DEF(float, max);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(Histogram, HistogramTilingData)
}
