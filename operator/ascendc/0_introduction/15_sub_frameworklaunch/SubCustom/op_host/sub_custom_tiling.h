/**
 * @file sub_custom_tiling.h
 *
 * Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef SUB_CUSTOM_TILING_H
#define SUB_CUSTOM_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(SubCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, formerNum);     // number of cores allocated to a larger amount of data, i.e., large blocks
  TILING_DATA_FIELD_DEF(uint32_t, tailNum);       // number of cores allocated to a smaller amount of data, i.e., small blocks
  TILING_DATA_FIELD_DEF(uint32_t, formerLength);  // length of the large block
  TILING_DATA_FIELD_DEF(uint32_t, tailLength);    // length of the small block
  TILING_DATA_FIELD_DEF(uint32_t, alignNum);      // minimum data amount that needs to be aligned
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SubCustom, SubCustomTilingData)
} // namespace optiling
#endif // SUB_CUSTOM_TILING_H
