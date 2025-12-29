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
#include "depth_to_space_NHWC.h"

using namespace AscendC;

template <typename T>
__aicore__ inline void DepthToSpaceNHWC<T>::Init(GM_ADDR x, GM_ADDR y, const DepthToSpaceTilingData* __restrict tiling_data)
{
    tiling = tiling_data;
    wcSize = tiling->wSize * tiling->cSize;
    uint32_t nhNum = tiling->nSize * tiling->hSize;
    formerNumPerCore = (nhNum + tiling->usedCoreNum - 1) / tiling->usedCoreNum;  //一个大核上的group数量
    tailNumPerCore = nhNum / tiling->usedCoreNum; //一个小核上的group数量
    formerCoreNum = nhNum % tiling->usedCoreNum;//大核个数
    uint32_t align = 32 / sizeof(T);
    uint32_t gmOffset, coreLen;
    if (GetBlockIdx() < formerCoreNum) {
        gmOffset = formerNumPerCore * wcSize * GetBlockIdx();
        coreLen = formerNumPerCore * wcSize;
    } else if (tailNumPerCore != 0) {
        gmOffset = formerNumPerCore * wcSize * formerCoreNum + tailNumPerCore * wcSize * (GetBlockIdx() - formerCoreNum);
        coreLen = tailNumPerCore * wcSize;
    }
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + gmOffset, coreLen);                    
    outYGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + gmOffset, coreLen);

    aligncSize = (tiling->cSize + align - 1) / align * align;
    alignblockSize =  (tiling->blockSize + align - 1) / align * align;
    pipe.InitBuffer(xVecIn, 2, alignblockSize *  (tiling->cSize / tiling->blockSize) * sizeof(T)); 
}

template <typename T>
__aicore__ inline void DepthToSpaceNHWC<T>::Process()
{
    if (GetBlockIdx() < formerCoreNum) {
        for (uint32_t i = 0; i < formerNumPerCore; i++) {
            Compute(i);
        }
    } else if (tailNumPerCore != 0) {
        for (uint32_t i = 0; i < tailNumPerCore; i++) {
            Compute(i);
        }
    }
}
        
template <typename T>
__aicore__ inline void DepthToSpaceNHWC<T>::copyIn(GlobalTensor<T>& gm, uint32_t copyInGmOffset)
{
    LocalTensor<T> xLocal = xVecIn.AllocTensor<T>();
    DataCopyPad(xLocal, gm[copyInGmOffset], 
        {(uint16_t) (tiling->cSize / tiling->blockSize), (uint32_t)(tiling->blockSize * sizeof(T)), 0, 0, 0},
        {true, 0, (uint8_t)(alignblockSize - tiling->blockSize), 0});
    xVecIn.EnQue(xLocal);
}


template <typename T>
__aicore__ inline void DepthToSpaceNHWC<T>::Compute(uint32_t curWC)
{
    uint32_t offset = curWC * wcSize;
    for (uint32_t j = 0; j < tiling->wSize; j++) {
        copyIn(xGm, (offset + j * tiling->cSize));
        CopyOut(outYGm, (offset + j * tiling->blockSize));
    }
}

template <typename T>
__aicore__ inline void DepthToSpaceNHWC<T>::CopyOut(GlobalTensor<T>& gm, uint32_t copyOutGmOffset) 
{
    LocalTensor<T> xLocal = xVecIn.DeQue<T>();
    for (uint32_t k = 0; k < (tiling->cSize / tiling->blockSize); k++) {
        DataCopyPad(gm[copyOutGmOffset + k * (tiling->cSize / tiling->blockSize) * tiling->wSize], xLocal[k * alignblockSize],
            {1, (uint32_t)(tiling->blockSize * sizeof(T)), 0, 0, 0});
    }
    xVecIn.FreeTensor(xLocal);
}
