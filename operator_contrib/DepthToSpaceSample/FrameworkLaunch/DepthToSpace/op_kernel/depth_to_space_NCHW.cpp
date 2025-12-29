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
#include "depth_to_space_NCHW.h"

using namespace AscendC;

template <typename T>
__aicore__ inline void DepthToSpaceNCHW<T>::Init(GM_ADDR x, GM_ADDR y, const DepthToSpaceTilingData* __restrict tiling_data)
{
    tiling = tiling_data;
    onebatchSize = tiling->wSize * tiling->hSize * tiling->cSize;
    // uint32_t hwNum = tiling->nSize * (tiling->cSize / tiling->blockSize);
    formerNumPerCore = (tiling->nSize + tiling->usedCoreNum - 1) / tiling->usedCoreNum;  //一个大核上的group数量
    tailNumPerCore = tiling->nSize / tiling->usedCoreNum; //一个小核上的group数量
    formerCoreNum = tiling->nSize % tiling->usedCoreNum;//大核个数
    uint32_t align = 32 / sizeof(T);
    uint32_t gmOffset, coreLen;
    if (GetBlockIdx() < formerCoreNum) {
        gmOffset = formerNumPerCore * onebatchSize * GetBlockIdx();
        coreLen = formerNumPerCore * onebatchSize;
    } else if (tailNumPerCore != 0) {
        gmOffset = formerNumPerCore * onebatchSize * formerCoreNum + tailNumPerCore * onebatchSize * (GetBlockIdx() - formerCoreNum);
        coreLen = tailNumPerCore * onebatchSize;
    }
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + gmOffset, coreLen);                    
    outYGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + gmOffset, coreLen);

    alignwSize = (tiling->wSize + align - 1) / align * align;
    alignblockSize =  (tiling->blockSize + align - 1) / align * align;
    pipe.InitBuffer(xVecIn, 2, alignblockSize *  tiling->wSize * sizeof(T)); 
}

template <typename T>
__aicore__ inline void DepthToSpaceNCHW<T>::Process(){
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
__aicore__ inline void DepthToSpaceNCHW<T>::copyIn(GlobalTensor<T>& gm, uint32_t copyInGmOffset)
{
    LocalTensor<T> xLocal = xVecIn.AllocTensor<T>();
    DataCopyPad(xLocal, gm[copyInGmOffset], 
        {(uint16_t)tiling->wSize, (uint32_t)(sizeof(T)), 0, 0, 0},
        {true, 0, (uint8_t)(alignblockSize - 1), 0});
    xVecIn.EnQue(xLocal);
}


template <typename T>
__aicore__ inline void DepthToSpaceNCHW<T>::Compute(uint32_t curbatch)
{
    uint32_t inoffset = curbatch * onebatchSize, outoffset = curbatch * onebatchSize;
    for(uint32_t n = 0; n < tiling->cSize / tiling->blockSize; n ++) {
        for (uint32_t j = 0; j < tiling->blockSize * tiling->hSize; j++) {
            copyIn(xGm, (inoffset + j * tiling->wSize));
            CopyOut(outYGm, (outoffset + (j / tiling->hSize)  + (j % tiling->hSize) * tiling->cSize * tiling->wSize));
        }
        inoffset += tiling->blockSize * tiling->hSize * tiling->wSize;
        outoffset += tiling->blockSize * tiling->wSize;
    }    
}

template <typename T>
__aicore__ inline void DepthToSpaceNCHW<T>::CopyOut(GlobalTensor<T>& gm, uint32_t copyOutGmOffset) 
{
    LocalTensor<T> xLocal = xVecIn.DeQue<T>();
    for (uint32_t k = 0; k < tiling->wSize; k++) {
        DataCopyPad(gm[copyOutGmOffset + k * tiling->blockSize], xLocal[k * alignblockSize],
            {1, (uint32_t)sizeof(T), 0, 0, 0});
    }
    xVecIn.FreeTensor(xLocal);
}
