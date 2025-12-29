/**
* @file three_nn_tiling.h
*
* Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(ThreeNNTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, CoreDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, finalTileNum);
  TILING_DATA_FIELD_DEF(uint32_t, tileDataNum);
  TILING_DATA_FIELD_DEF(uint32_t, TailDataNum);

  TILING_DATA_FIELD_DEF(uint32_t, shapeB);
  TILING_DATA_FIELD_DEF(uint32_t, shapeN);
  TILING_DATA_FIELD_DEF(uint32_t, shapeM);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(ThreeNN, ThreeNNTilingData)
}
