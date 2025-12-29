/**
 * @file add_custom_tiling.h
 *
 * Copyright (C) 2025. Huawei Technologies Co. Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef ADD_CUSTOM_TILING_H
#define ADD_CUSTOM_TILING_H
#include <cstdint>

struct AddCustomTilingData {
    uint32_t xLen;
    uint32_t yLen;
    uint32_t coef;
    uint32_t axis;
    uint32_t dataType;

    uint32_t isEvenCore;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t lastTileLength;

    uint32_t formerNum;
    uint32_t formerLength;
    uint32_t formerTileNum;
    uint32_t formerTileLength;
    uint32_t formerLastTileLength;

    uint32_t tailNum; 
    uint32_t tailLength;
    uint32_t tailTileNum;
    uint32_t tailTileLength;
    uint32_t tailLastTileLength;
};
#endif  // ADD_CUSTOM_TILING_H