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
#ifndef _ASCENDC_DEPTHTOSPACENCHW_H_
#define _ASCENDC_DEPTHTOSPACENCHW_H_
#include "kernel_operator.h"
using namespace AscendC;

template <typename T>
class DepthToSpaceNCHW {
public:
    __aicore__ inline DepthToSpaceNCHW() = default;
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, 
                                const DepthToSpaceTilingData* __restrict tiling_data);
    __aicore__ inline void Process();
            
protected:
    __aicore__ inline void copyIn(GlobalTensor<T>& gm, uint32_t copyInGmOffset);
    __aicore__ inline void Compute(uint32_t curHW);
    __aicore__ inline void CopyOut(GlobalTensor<T>& gm, uint32_t copyOutGmOffset);

protected:
    TPipe pipe;
    GlobalTensor<T> xGm;
    GlobalTensor<T> outYGm;
    // Queue
    static constexpr int DB = 2;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, DB> xVecIn;
    const DepthToSpaceTilingData* __restrict tiling;
    uint32_t onebatchSize;
    uint32_t formerNumPerCore;
    uint32_t tailNumPerCore;
    uint32_t formerCoreNum;
    uint32_t alignblockSize;
    uint32_t alignwSize;
};
#endif