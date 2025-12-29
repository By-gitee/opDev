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
#include <cstddef>
#include "add_custom_tiling.h"

// bfloat16, float16, float, int8, int16, int32
constexpr uint32_t DATA_TYPE_SIZE[] = {2, 2, 4, 1, 2, 4};
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t UB_BLOCK_NUM = 100;  // UB最大可以使用的block数量
constexpr uint32_t MAX_AVAILABLE_UB_BLOCK_NUM = UB_BLOCK_NUM / BUFFER_NUM * BUFFER_NUM;

namespace {
// tiling参数计算函数
void TilingParamsCalc(uint32_t length, uint32_t alignNum, uint32_t& tileNum, uint32_t& tileLength,
    uint32_t& lastTileLength)
{
    assert(alignNum != 0U);
    tileNum = length / (alignNum * MAX_AVAILABLE_UB_BLOCK_NUM);

    // 单核需要计算的长度 < 单核UB最大一次可计算长度 -> 仅有尾块
    if (tileNum == 0U) {
        tileLength = 0U;
        lastTileLength = static_cast<uint32_t>(((length + alignNum - 1) / alignNum) * alignNum);
    } else if (static_cast<uint32_t>(length / alignNum) % MAX_AVAILABLE_UB_BLOCK_NUM == 0U) {
        // 单核需要计算的长度 = 单核UB最大一次可计算长度 的整数倍 -> 仅有整块
        tileLength = MAX_AVAILABLE_UB_BLOCK_NUM * alignNum;
        lastTileLength = 0U;
    } else {
        // 有整块 + 尾块
        tileLength = MAX_AVAILABLE_UB_BLOCK_NUM * alignNum;
        lastTileLength = static_cast<uint32_t>(length - tileNum* tileLength);
    }
}
}

void GenerateTilingData(uint8_t* tilingBuf, uint32_t blockDim)
{
    uint32_t totalLength;           // 总共要计算的元素个数
    uint32_t dataTypeSize;
    uint32_t blockLength;
    uint32_t totalLengthAligned;

    AddCustomTilingData *tiling = reinterpret_cast<AddCustomTilingData *>(tilingBuf);
    totalLength = tiling->totalLength;
    dataTypeSize = DATA_TYPE_SIZE[tiling->dataType];

    uint32_t alignNum = BLOCK_SIZE / dataTypeSize;     // 一个block中的元素个数
    assert((alignNum != 0U) && (blockDim != 0U));
    /** 计算使用的核数 **/
    /* 如果传入数据的长度非32B对齐, 计算对齐后的长度*/
    totalLengthAligned = (totalLength % alignNum == 0U)?
        static_cast<uint32_t>(totalLength) :
        ((static_cast<uint32_t>(totalLength) + alignNum - 1) / alignNum) * alignNum;

    /* 核间可均分场景 */
    if ((totalLengthAligned / alignNum) % blockDim == 0U) {
        uint32_t tileNum = 0;
        uint32_t tileLength = 0;
        uint32_t lastTileLength = 0;
        blockLength = totalLengthAligned / blockDim;
        TilingParamsCalc(blockLength, alignNum, tileNum, tileLength, lastTileLength);

        tiling->blockLength = blockLength;
        tiling->tileNum = tileNum;
        tiling->tileLength = tileLength;
        tiling->lastTileLength = lastTileLength;
        tiling->isEvenCore = 1U;
    } else {  // 核间不可均分
        uint32_t formerNum = (totalLengthAligned / alignNum) % blockDim;
        uint32_t tailNum = blockDim - formerNum;
        // 计算整块和尾块的数据量
        uint32_t formerLength =
            static_cast<uint32_t>(((totalLengthAligned + blockDim - 1) / blockDim + alignNum - 1) / alignNum) * alignNum;
        uint32_t tailLength = (totalLengthAligned / blockDim / alignNum) * alignNum;

        uint32_t formerTileNum;
        uint32_t formerTileLength;
        uint32_t formerLastTileLength;

        uint32_t tailTileNum;
        uint32_t tailTileLength;
        uint32_t tailLastTileLength;

        TilingParamsCalc(formerLength, alignNum, formerTileNum, formerTileLength, formerLastTileLength);
        TilingParamsCalc(tailLength, alignNum, tailTileNum, tailTileLength, tailLastTileLength);

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