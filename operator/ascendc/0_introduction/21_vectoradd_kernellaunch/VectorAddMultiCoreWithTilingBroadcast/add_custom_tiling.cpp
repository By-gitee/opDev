/**
 * @file add_custom_tiling.cpp
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

#include <cassert>
#include "add_custom_tiling.h"

// bfloat16, float16, float, int8, int16, int32
constexpr uint32_t DATA_TYPE_SIZE[] = {2, 2, 4, 1, 2, 4};
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t UB_BLOCK_NUM = 100;  // UB最大可以使用的block数量
constexpr uint32_t MAX_AVAILABLE_UB_BLOCK_NUM = UB_BLOCK_NUM / BUFFER_NUM * BUFFER_NUM;
void TilingParamsCalc(uint32_t length, uint32_t ubBlockNum,
    uint32_t& tileNum, uint32_t& tileLength, uint32_t& lastTileLength)
{
    assert(ubBlockNum != 0U);
    tileNum = length / ubBlockNum;
    if (length % ubBlockNum == 0U || tileNum == 0U) {
        if (tileNum == 0U) {
            tileNum = 1U;
        }
        if (length < ubBlockNum) {
            tileLength = length;
            lastTileLength = tileLength;
        } else {
            tileLength = ubBlockNum;
            lastTileLength = tileLength;
        }
    } else {
        tileNum++;
        tileLength = ubBlockNum;
        lastTileLength = (uint32_t)(length - (tileNum - 1) * tileLength);
    }
}

void GenerateTilingData(uint8_t* tilingBuf, uint32_t blockDim)
{
    uint32_t xLen;
    uint32_t yLen;
    uint32_t totalLength;
    uint32_t dataTypeSize;

    AddCustomTilingData *tiling = reinterpret_cast<AddCustomTilingData *>(tilingBuf);
    xLen = tiling->xLen;
    yLen = tiling->yLen;
    assert((xLen != 0U) && (yLen != 0U));
    dataTypeSize = DATA_TYPE_SIZE[tiling->dataType];
    totalLength = (xLen > yLen)? xLen : yLen;

    uint32_t alignNum = BLOCK_SIZE / dataTypeSize;
    assert((alignNum != 0U) && (blockDim != 0U));
    uint32_t shorterAxisLen = (xLen < yLen)? xLen : yLen;
    uint32_t alignCoef = (tiling->axis == 0U)? shorterAxisLen : totalLength / shorterAxisLen;
    uint32_t divDimCoef = (tiling->axis == 0U)? totalLength / shorterAxisLen : shorterAxisLen;
    uint32_t ubBlockAligned =
        (MAX_AVAILABLE_UB_BLOCK_NUM * alignNum / (alignCoef * BUFFER_NUM) * (alignCoef * BUFFER_NUM) == 0U)?
        MAX_AVAILABLE_UB_BLOCK_NUM : MAX_AVAILABLE_UB_BLOCK_NUM * alignNum / (alignCoef * BUFFER_NUM) * (alignCoef * BUFFER_NUM);

    if (divDimCoef % blockDim == 0U) {
        uint32_t blockLength = divDimCoef / blockDim * alignCoef;
        uint32_t tileNum = 0;
        uint32_t tileLength = 0;
        uint32_t lastTileLength = 0;
        if (tiling->axis == 0U) {
            tileNum = blockLength / shorterAxisLen;
            tileLength = shorterAxisLen;
            lastTileLength = tileLength;
        } else {
            TilingParamsCalc(blockLength, ubBlockAligned, tileNum, tileLength, lastTileLength);
        }

        tiling->blockLength = blockLength;
        tiling->tileNum = tileNum;
        tiling->tileLength = tileLength;
        tiling->lastTileLength = lastTileLength;
        tiling->isEvenCore = 1U;
    } else {
        uint32_t formerNum;
        uint32_t tailNum;

        uint32_t formerLength;
        uint32_t tailLength;

        uint32_t formerTileNum;
        uint32_t formerTileLength;
        uint32_t formerLastTileLength;

        uint32_t tailTileNum;
        uint32_t tailTileLength;
        uint32_t tailLastTileLength;
        if (tiling->axis == 0) {
            formerNum = divDimCoef % blockDim;
            tailNum = blockDim - formerNum;

            formerLength = (divDimCoef / blockDim + 1U)  * alignCoef;
            tailLength = divDimCoef / blockDim * alignCoef;

            formerTileNum = formerLength / shorterAxisLen;
            formerTileLength = shorterAxisLen;
            formerLastTileLength = shorterAxisLen;

            tailTileNum = tailLength / shorterAxisLen;
            tailTileLength = shorterAxisLen;
            tailLastTileLength = shorterAxisLen;
        } else {
            formerNum = (divDimCoef / BUFFER_NUM) % blockDim;
            tailNum = blockDim - formerNum;

            formerLength = (((divDimCoef / BUFFER_NUM) / blockDim) + 1U) * BUFFER_NUM * alignCoef;
            tailLength = ((divDimCoef / BUFFER_NUM) / blockDim) * BUFFER_NUM * alignCoef;

            TilingParamsCalc(formerLength, ubBlockAligned,
                formerTileNum, formerTileLength, formerLastTileLength);
            TilingParamsCalc(tailLength, ubBlockAligned,
                tailTileNum, tailTileLength, tailLastTileLength);
        }

        tiling->formerNum = formerNum;
        tiling->formerLength = formerLength;
        tiling->formerTileNum = formerTileNum;
        tiling->formerTileLength = formerTileLength;
        tiling->formerLastTileLength = formerLastTileLength;

        tiling->tailNum = tailNum;
        tiling->tailLength = tailLength;
        tiling->tailTileNum = tailTileNum;
        tiling->tailTileLength = tailTileLength;
        tiling->tailLastTileLength = tailLastTileLength;
        tiling->isEvenCore = 0U;
    }
}
