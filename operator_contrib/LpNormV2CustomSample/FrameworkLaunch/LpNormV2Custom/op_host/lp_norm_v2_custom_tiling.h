
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 */
#ifndef LPNORMV2_CUSTOM_TILING_H
#define LPNORMV2_CUSTOM_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(LpNormV2CustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, blockLength);
  TILING_DATA_FIELD_DEF(uint32_t, tileNum);
  TILING_DATA_FIELD_DEF(uint32_t, tileLength);
  TILING_DATA_FIELD_DEF(uint32_t, lasttileLength);
  TILING_DATA_FIELD_DEF(uint32_t, typeKey);
  TILING_DATA_FIELD_DEF(uint32_t, pType);
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, stepSize);
  TILING_DATA_FIELD_DEF(uint32_t, unitCount);
  TILING_DATA_FIELD_DEF(float, pValue);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(LpNormV2Custom, LpNormV2CustomTilingData)
}
#endif // LPNORMV2_CUSTOM_TILING_H