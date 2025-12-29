/**
 * @file add_custom_tiling.h
 *
 * Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef ADD_CUSTOM_TILING_H
#define ADD_CUSTOM_TILING_H
#include <cstdint>

namespace optiling {
class TilingData{
public:
    uint32_t totalLength;
};

class TilingDataFp{
public:
    uint32_t totalLength;
};

class TilingDataFp16{
public:
    uint32_t totalLength;
};
} // namespace optiling
#endif // ADD_CUSTOM_TILING_H
