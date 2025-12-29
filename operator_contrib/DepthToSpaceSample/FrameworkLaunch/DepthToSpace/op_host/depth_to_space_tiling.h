/* Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved. 
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.
 * You may not use this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef DEPTH_TO_SPACE_TILING_H
#define DEPTH_TO_SPACE_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {

enum class RNNTilingKey : int64_t {
  NHWC_C1 = 10000001,
  NCHW_C1,
  NHWC_NO_C1
};

BEGIN_TILING_DATA_DEF(DepthToSpaceTilingData)
  TILING_DATA_FIELD_DEF(int64_t, tilingKey);
  TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
  TILING_DATA_FIELD_DEF(uint64_t, UBSize);
  TILING_DATA_FIELD_DEF(uint32_t, nSize);
  TILING_DATA_FIELD_DEF(uint32_t, cSize);
  TILING_DATA_FIELD_DEF(uint32_t, hSize);
  TILING_DATA_FIELD_DEF(uint32_t, wSize);
  TILING_DATA_FIELD_DEF(uint16_t, blockSize);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(DepthToSpace, DepthToSpaceTilingData)
}
#endif
