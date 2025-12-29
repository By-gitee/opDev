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

#ifndef _BALL_QUARY_TILING_H_
#define _BALL_QUARY_TILING_H_

#include "register/tilingdata_base.h"
#include "ball_query_tiling.h"

namespace optiling
{
  enum class BallQueryTilingKey : int64_t {
    INPUT_FP16_NORM = 10000001,
    INPUT_FP16_STACK,
    INPUT_FP32_NORM,
    INPUT_FP32_STACK,
  };

  BEGIN_TILING_DATA_DEF(BallQueryTilingData)
  TILING_DATA_FIELD_DEF(int64_t, tilingKey);
  TILING_DATA_FIELD_DEF(uint32_t, B);
  TILING_DATA_FIELD_DEF(uint32_t, N);
  TILING_DATA_FIELD_DEF(uint32_t, M);
  TILING_DATA_FIELD_DEF(float, minr);     // min radius
  TILING_DATA_FIELD_DEF(float, maxr);     // max radius
  TILING_DATA_FIELD_DEF(uint32_t, sn);    // sample number
  TILING_DATA_FIELD_DEF(uint32_t, bn);    // batch number
  TILING_DATA_FIELD_DEF(uint32_t, usedCoreNum);
  TILING_DATA_FIELD_DEF(uint64_t, UBSize);
  END_TILING_DATA_DEF;

  REGISTER_TILING_DATA_CLASS(BallQuery, BallQueryTilingData)
}

#endif    // _BALL_QUARY_TILING_H_