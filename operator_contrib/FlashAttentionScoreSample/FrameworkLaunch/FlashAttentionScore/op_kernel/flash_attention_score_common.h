/**
 * Copyright (c) 2023-2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file flash_attention_score_common.h
 * \brief
 */

#ifndef FLASH_ATTENTION_SCORE_COMMON_H
#define FLASH_ATTENTION_SCORE_COMMON_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"
#include "stdarg.h"

using AscendC::LocalTensor;
using AscendC::GlobalTensor;
using AscendC::DataFormat;
using AscendC::ShapeInfo;
using AscendC::DataCopyParams;
using AscendC::DataCopyPadParams;
using AscendC::HardEvent;
using AscendC::SetFlag;
using AscendC::WaitFlag;
using AscendC::BinaryRepeatParams;
using AscendC::Cast;
using AscendC::Div;
using AscendC::Duplicate;
using AscendC::GetBlockIdx;
using AscendC::RoundMode;
using AscendC::Select;
using AscendC::SelectWithBytesMaskShapeInfo;
using AscendC::SoftmaxFlashV2;
using AscendC::SoftMaxShapeInfo;
using AscendC::TBuf;
using AscendC::TPipe;
using AscendC::TPosition;

constexpr MatmulConfig CFG_EXCEED = GetNormalConfig(true);
constexpr static uint64_t BLOCK_BYTE = 32;
constexpr static int32_t SOFTMAX_M_ALIGNED_SIZE = 8;
constexpr static int32_t SOFTMAX_K_ALIGNED_SIZE = 64;
constexpr int32_t blockBytes = 32;
constexpr static int32_t blockSize = blockBytes / 4; // 4 means sizeof(T)
constexpr static int32_t repeatMaxBytes = 256;
constexpr static int32_t repeatMaxSize = repeatMaxBytes / 4; // 4 means sizeof(T)

// 0级接口的block间隔范围需要满足32B对齐
constexpr static int32_t fp32BaseSize = 8;

namespace math {
template <typename T> __aicore__ inline T Ceil(T a, T b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

template <typename T> __aicore__ inline T Align(T a, T b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b * b;
}
}

template <typename T1, typename T2>
__aicore__ inline T1 CeilDiv(T1 a, T2 b)
{
    if (b == 0) {
        return 0;
    }
    return (a + b - 1) / b;
}

template <typename T1, typename T2>
__aicore__ inline T1 Min(T1 a, T2 b)
{
    return (a > b) ? (b) : (a);
}

__aicore__ inline int32_t Align(int32_t shape)
{
    int32_t alignFactor = 16;
    int32_t alignedSize = CeilDiv(shape, alignFactor) * alignFactor;
    return alignedSize;
}

__aicore__ inline bool IsBasicBlockInSoftMax(int32_t srcM, int32_t srcK)
{
    return srcM % SOFTMAX_M_ALIGNED_SIZE == 0 && srcK % SOFTMAX_K_ALIGNED_SIZE == 0;
}

template <typename T>
__aicore__ inline void DataCopy2D(const LocalTensor<T> &dstLocal, const GlobalTensor<T> &srcGlobal, const uint32_t d0,
                                  const uint32_t d1, const uint32_t orgD1, uint64_t paddingValue = 0)
{
    if (d1 % (BLOCK_BYTE / sizeof(T)) == 0 && orgD1 % (BLOCK_BYTE / sizeof(T)) == 0) {
        auto d1Blocks = math::Ceil(d1 * sizeof(T), BLOCK_BYTE);
        auto orgD1Blocks = math::Ceil(orgD1 * sizeof(T), BLOCK_BYTE);
        DataCopyParams copyParams(d0, d1Blocks, orgD1Blocks - d1Blocks, 0);
        DataCopy(dstLocal, srcGlobal, copyParams);
    } else {
        auto d1Bytes = d1 * sizeof(T);
        auto d1Aligned = math::Align(static_cast<int64_t>(d1), static_cast<int64_t>(BLOCK_BYTE / sizeof(T)));
        DataCopyParams copyParams(static_cast<uint16_t>(d0), static_cast<uint16_t>(d1Bytes),
                                  orgD1 * sizeof(T) - d1Bytes, 0);
        DataCopyPadParams padParams(true, 0, static_cast<uint8_t>(d1Aligned - d1), paddingValue);
        DataCopyPad(dstLocal, srcGlobal, copyParams, padParams);
    }
}

#endif // FLASH_ATTENTION_SCORE_COMMON_H

