/**
* @file as_strided_tiling.h
*
* Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
 
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(AsStridedTilingData)
  TILING_DATA_FIELD_DEF(int32_t, size_dimensional);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(AsStrided, AsStridedTilingData)
}
