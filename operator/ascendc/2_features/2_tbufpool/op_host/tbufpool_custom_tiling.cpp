/**
 * @file tbufpool_custom_tiling.cpp
 *
 * Copyright (C) 2024-2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

#include "tiling/tiling_api.h"
#include "tbufpool_custom_tiling.h"


void GenerateTilingData(uint32_t totalLength, uint8_t* tilingBuf)
{
    TbufPoolTilingData *tiling = reinterpret_cast<TbufPoolTilingData *>(tilingBuf);
    tiling->totalLength = totalLength;
}