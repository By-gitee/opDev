/**
 * @file adds_custom_tiling.h
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef ADDS_CUSTOM_TILING_H
#define ADDS_CUSTOM_TILING_H
#include <cstdint>

class AddsCustomTilingData {
public:
    uint32_t m;
    uint32_t n;
    uint32_t tileM;
    uint32_t tileN;
    uint32_t loopOneCore;
};
#endif // ADDS_CUSTOM_TILING_H
