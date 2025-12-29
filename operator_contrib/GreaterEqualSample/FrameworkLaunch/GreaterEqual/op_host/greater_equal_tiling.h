/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 */
#ifndef GREATEREQUAL_TILING_H
#define GREATEREQUAL_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(TilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileNumMean);
  TILING_DATA_FIELD_DEF(uint32_t, tileNumEnd);
  TILING_DATA_FIELD_DEF(uint32_t, tileLengthMean);
  TILING_DATA_FIELD_DEF(uint32_t, tileLengthEnd);
  TILING_DATA_FIELD_DEF(uint32_t, blockLengthMean);
  TILING_DATA_FIELD_DEF(uint32_t, blockLengthEnd);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(GreaterEqual, TilingData)
}
#endif // GREATEREQUAL_TILING_H