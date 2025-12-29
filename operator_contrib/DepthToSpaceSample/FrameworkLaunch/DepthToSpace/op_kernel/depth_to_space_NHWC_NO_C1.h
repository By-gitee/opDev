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
#ifndef _ASCENDC_DEPTHTOSPACENHWC_NO_C1_H_
#define _ASCENDC_DEPTHTOSPACENHWC_NO_C1_H_
#include "kernel_operator.h"
using namespace AscendC;


template <typename T>
class DepthToSpaceNHWC_NO_C1 {
public:
    __aicore__ inline DepthToSpaceNHWC_NO_C1() = default;
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const DepthToSpaceTilingData* __restrict tiling_data);
    __aicore__ inline void Process();
            
protected:
    __aicore__ inline void Compute(uint32_t curcNums, uint32_t copyInNumOffset, uint32_t copyIn);

protected:
    TPipe pipe;
    GlobalTensor<T> xGm;
    GlobalTensor<T> outYGm;
    // Queue
    static constexpr int DB = 2;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, DB> xVecIn;
    const DepthToSpaceTilingData* __restrict tiling;
    uint32_t coutSize;
    uint32_t coutLen;
    uint32_t formerNumPerCore;
    uint32_t tailNumPerCore;
    uint32_t formerCoreNum;
    uint32_t alignblockSize;
    uint32_t aligncoutSize;
    uint32_t oneNcoutNum;
    uint32_t oneHcoutNum;
    uint32_t oneWcoutNum;
    uint32_t oneCopyIn;
    uint32_t CopyInNum;
    uint32_t oneCopyInLast;
    uint32_t srcStride;
};
#endif