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
#include "depth_to_space_NHWC_NO_C1.h"

using namespace AscendC;

template <typename T>
__aicore__ inline void DepthToSpaceNHWC_NO_C1<T>::Init(GM_ADDR x, GM_ADDR y, const DepthToSpaceTilingData* __restrict tiling_data)
{
    tiling = tiling_data;
    uint32_t totalNum, totalLen, align, totalWNum;
    totalWNum = tiling->nSize * tiling->hSize * tiling->wSize;
    totalNum = totalWNum * tiling->blockSize;
    totalLen = totalWNum * tiling->cSize;
    coutLen = tiling->cSize / tiling->blockSize;
    coutSize = coutLen * sizeof(T);
    oneNcoutNum = tiling->hSize * tiling->wSize * tiling->blockSize;
    oneHcoutNum = tiling->wSize * tiling->blockSize;
    oneWcoutNum = tiling->blockSize;
    formerCoreNum = totalNum % tiling->usedCoreNum;//大核个数32
    formerNumPerCore = 3278;
    tailNumPerCore = 3272;
    oneCopyIn = 48;
    srcStride = coutSize / 32;
    uint32_t ingmOffset, coreLen;
    if (GetBlockIdx() < formerCoreNum) {
        ingmOffset = formerNumPerCore * coutLen * GetBlockIdx();
        coreLen = formerNumPerCore * coutLen;
        CopyInNum = (formerNumPerCore + oneCopyIn - 1) / oneCopyIn;//
        oneCopyInLast = formerNumPerCore - oneCopyIn * (CopyInNum - 1);//
    } else if (tailNumPerCore != 0) {
        ingmOffset = formerNumPerCore * coutLen * formerCoreNum +
                        tailNumPerCore * coutLen * (GetBlockIdx() - formerCoreNum);
        coreLen = tailNumPerCore * coutLen;
        CopyInNum = (tailNumPerCore + oneCopyIn - 1) / oneCopyIn;//
        oneCopyInLast = tailNumPerCore - oneCopyIn * (CopyInNum - 1);//
    }
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + ingmOffset, coreLen);                    
    outYGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y), totalLen);
    pipe.InitBuffer(xVecIn, 2, oneCopyIn * coutSize); 
}

template <typename T>
__aicore__ inline void DepthToSpaceNHWC_NO_C1<T>::Process()
{
    uint32_t curcNum;
    if (GetBlockIdx() < formerCoreNum) {
        curcNum = GetBlockIdx() * formerNumPerCore;
        for (uint32_t i = 0; i < CopyInNum - 1 ; i++) {
            Compute(curcNum, i, oneCopyIn);
            curcNum += oneCopyIn;
        }
        Compute(curcNum, CopyInNum - 1, oneCopyInLast);
        
    } else if (tailNumPerCore != 0) {
        curcNum = formerCoreNum * formerNumPerCore + (GetBlockIdx() - formerCoreNum) * tailNumPerCore;
        for (uint32_t i = 0; i < CopyInNum - 1 ; i++) {
            Compute(curcNum, i, oneCopyIn);
            curcNum += oneCopyIn;
        }
        Compute(curcNum, CopyInNum - 1, oneCopyInLast);
    }
}

template <typename T>
__aicore__ inline void DepthToSpaceNHWC_NO_C1<T>::Compute(uint32_t curcNums, uint32_t copyInNumOffset, uint32_t copyIn)
{
    uint32_t curNOffset, curHOffset, curw, curc, outOffset, inoffset, nextW, oneW, nextOffset;

    inoffset = copyInNumOffset * oneCopyIn * coutLen;
    curNOffset = curcNums / oneNcoutNum * oneNcoutNum;
    curHOffset = (curcNums - curNOffset) / oneHcoutNum * oneHcoutNum;
    curw = (curcNums - curNOffset - curHOffset) / oneWcoutNum;
    outOffset = (curNOffset + curHOffset + curw) * coutLen;
    oneW = oneHcoutNum - curw * oneWcoutNum;
    if (oneW >= copyIn) {
        LocalTensor<T> xLocal = xVecIn.AllocTensor<T>();
        DataCopyPad(xLocal, xGm[inoffset], 
            {(uint16_t)copyIn, coutSize, 0, 0, 0},
            {false, 0, 0, 0});
        xVecIn.EnQue(xLocal);
        xLocal = xVecIn.DeQue<T>();
        DataCopyPad(outYGm[outOffset],
                    xLocal,
                    {(uint16_t)(copyIn / 2), coutSize, srcStride, 0, 0});
        DataCopyPad(outYGm[outOffset + tiling->wSize * coutLen],
                    xLocal[coutLen],
                    {(uint16_t)(copyIn / 2), coutSize, srcStride, 0, 0});
        xVecIn.FreeTensor(xLocal);
    } else {
        nextW = copyIn - oneW;
        nextOffset = (2 * tiling->wSize  - curw ) * coutLen;
        LocalTensor<T> xLocal = xVecIn.AllocTensor<T>();
        DataCopyPad(xLocal, xGm[inoffset], 
            {(uint16_t)copyIn, coutSize, 0, 0, 0},
            {false, 0, 0, 0});
        xVecIn.EnQue(xLocal);
        xLocal = xVecIn.DeQue<T>();
        DataCopyPad(outYGm[outOffset],
                    xLocal,
                    {(uint16_t)(oneW / 2), coutSize, srcStride, 0, 0});
        DataCopyPad(outYGm[outOffset + tiling->wSize * coutLen],
                    xLocal[coutLen],
                    {(uint16_t)(oneW / 2), coutSize, srcStride, 0, 0});
        DataCopyPad(outYGm[outOffset + nextOffset],
                    xLocal[oneW * coutLen],
                    {(uint16_t)(nextW / 2), coutSize, srcStride, 0, 0});
        DataCopyPad(outYGm[outOffset + nextOffset + tiling->wSize * coutLen],
                    xLocal[oneW * coutLen + coutLen],
                    {(uint16_t)(nextW / 2), coutSize, srcStride, 0, 0});
        xVecIn.FreeTensor(xLocal);
    }
}
