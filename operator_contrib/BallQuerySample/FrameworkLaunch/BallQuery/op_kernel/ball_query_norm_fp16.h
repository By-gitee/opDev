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
#ifndef __BALL_QUERY_NORM_FP16_H_
#define __BALL_QUERY_NORM_FP16_H_
#include "kernel_operator.h"
#include "utils.h"
#include "ball_query_norm_fp16.h"

using namespace AscendC;

template<typename T> class BallQueryNormFP16 {
public:
    __aicore__ inline BallQueryNormFP16() {}
    __aicore__ inline void Init(GM_ADDR in_xyz, GM_ADDR in_ctr, GM_ADDR in_idx, const BallQueryTilingData* __restrict tiling) 
    {
        this->B = tiling->B;
        this->N = tiling->N;
        this->M = tiling->M;
        this->minr = tiling->minr * tiling->minr;
        this->maxr = tiling->maxr * tiling->maxr;
        this->sn = tiling->sn;
        this->usedCoreNum = tiling->usedCoreNum;
        alignNum32 = BYTE_PER_BLK/sizeof(T);
        alignNum256 = BYTE_PER_BIG_BLK/sizeof(T);
        align32N = ceil(N, alignNum32);
        align32M = ceil(M, alignNum32);
        align256N = ceil(M, alignNum256);
        sn32 = ceil(sn, BYTE_PER_BLK);
        maxNM = ceil(N > M ? N : M, BYTE_PER_BLK/sizeof(T));
        alignMaskLen = ceil(div_ceil(align256N, INT32_CNT_PER_BLK), BYTE_PER_BLK);

        NumU32OfResBits = M * ceil(div_ceil(sn, INT32_BIT), INT32_CNT_PER_BLK);

        uint32_t formerBthNumPerCore = 0, tailBthNumPerCore = 0, formerCoreNum = 0, tailCoreNum = 0;
        if (usedCoreNum > 0){
            formerBthNumPerCore = div_ceil(B, usedCoreNum), tailBthNumPerCore = B / usedCoreNum;
            formerCoreNum = B % usedCoreNum, tailCoreNum = usedCoreNum - formerCoreNum;
        }
        usedCoreNum = (tailBthNumPerCore > 0 ? formerCoreNum : (formerCoreNum + tailCoreNum));

        if(GetBlockIdx() < formerCoreNum){
            startBthNum = formerBthNumPerCore * GetBlockIdx();
            curBthNum = formerBthNumPerCore;
        }
        else{
            startBthNum = formerBthNumPerCore * formerCoreNum + tailBthNumPerCore * (GetBlockIdx() - formerCoreNum);
            curBthNum = tailBthNumPerCore;
        }
        if(GetBlockIdx() < usedCoreNum){
            InOutInit(in_xyz, in_ctr, in_idx);
            SeqInit();
            MaskInit();
        }
    }

    __aicore__ inline void Process() {
        if(GetBlockIdx() < usedCoreNum){
            for (uint32_t bth = 0; bth < curBthNum; ++bth) { // handle a batch per loop
                CopyIn(bth);
                Compute(bth);
                CopyOut(bth);
            }
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t bth) {           //load a batch xyz and ctr from gm
        DataCopy(xyzLocal, xyzGm[bth * N * dim3], align32N * dim3);
        DataCopy(ctrLocal, ctrGm[bth * M * dim3], align32M * dim3);
        xyzVecIn.EnQue(xyzLocal);
        ctrVecIn.EnQue(ctrLocal);
    }

    __aicore__ inline void Compute(uint32_t bth) {
        uint64_t rsvdCnt = 0;
        xyzLocal = xyzVecIn.DeQue<T>();
        ctrLocal = ctrVecIn.DeQue<T>();
        LocalTensor<T> xyz_x = transLocal;
        LocalTensor<T> xyz_y = xyz_x[align32N];
        LocalTensor<T> xyz_z = xyz_y[align32N];
        LocalTensor<T> tmp_x = xyz_z[align32N];
        LocalTensor<T> tmp_y = tmp_x[align256N];
        LocalTensor<T> tmp_z = tmp_y[align256N];

        Muls(ctrLocal, ctrLocal, (T)-1.0, align32M * dim3);

        LocalTensor<uint8_t> mask1Local = maskLocal;
        LocalTensor<uint8_t> mask2Local = maskLocal[alignMaskLen];

        Gather(xyz_x, xyzLocal, idxSeq0.ReinterpretCast<uint32_t>(), 0, N);    
        Gather(xyz_y, xyzLocal, idxSeq1.ReinterpretCast<uint32_t>(), 0, N);    
        Gather(xyz_z, xyzLocal, idxSeq2.ReinterpretCast<uint32_t>(), 0, N);    

        Duplicate(tmp_x, (T)MAX_HALF, align256N);
        for (int i = 0; i < M; ++i){
            px = (float)ctrLocal(i+i+i);
            py = (float)ctrLocal(i+i+i+1);
            pz = (float)ctrLocal(i+i+i+1+1);

            Adds(tmp_x, xyz_x, px, N);
            Adds(tmp_y, xyz_y, py, N);
            Adds(tmp_z, xyz_z, pz, N);
            Mul(tmp_x, tmp_x, tmp_x, N);
            Mul(tmp_y, tmp_y, tmp_y, N);
            Mul(tmp_z, tmp_z, tmp_z, N);
            Add(tmp_x, tmp_x, tmp_y, N);
            Add(tmp_x, tmp_x, tmp_z, N);
            
            CompareScalar(mask1Local, tmp_x, (T)maxr, CMPMODE::LT, align256N);
            CompareScalar(mask2Local, tmp_x, (T)minr, CMPMODE::GE, align256N);

            And(mask1Local.ReinterpretCast<uint16_t>(), mask1Local.ReinterpretCast<uint16_t>(), mask2Local.ReinterpretCast<uint16_t>(), align256N / (sizeof(uint16_t)/sizeof(uint8_t)));  // align256N/16 = align256N/8/2 2=sizeof(uint16_t)/sizeof(uint8_t)
            
            CompareScalar(mask2Local, tmp_x, (T)0, CMPMODE::EQ, align256N);
            Or(mask1Local.ReinterpretCast<uint16_t>(), mask1Local.ReinterpretCast<uint16_t>(), mask2Local.ReinterpretCast<uint16_t>(), align256N / (sizeof(uint16_t)/sizeof(uint8_t)));
            
            GatherMask(idxTmp1, idxSeqN, mask1Local.ReinterpretCast<uint32_t>(), true, N, {1,1,8,8}, rsvdCnt);
            if(sn < rsvdCnt){
                rsvdCnt = sn;
            }
            else{
                Duplicate(idxTmp2[i * ceil(sn, BYTE_PER_BLK/sizeof(T))], (int32_t)idxTmp1(0), sn);
            }
            MyCopy(idxTmp2[i * sn32], idxTmp1, rsvdCnt);
        }
        GatherMask(idxLocal, idxTmp2, resMaskLocal, true, M * sn32, {1,1,8,8}, rsvdCnt);
        idxVecOut.EnQue(idxLocal);
    }

    __aicore__ inline void  CopyOut(uint32_t bth) {
        idxLocal = idxVecOut.DeQue<int32_t>();
        DataCopyPad(idxGm[bth * M * sn], idxLocal, {1, (uint32_t)(M * sn * sizeof(int32_t)), 1, 1, 0});
    }

    __aicore__ inline void InOutInit(GM_ADDR in_xyz, GM_ADDR in_ctr, GM_ADDR in_idx){
        xyzGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(in_xyz) + startBthNum * N * dim3, curBthNum * align32N * dim3);                    
        ctrGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(in_ctr) + startBthNum * M * dim3, curBthNum * align32M * dim3);                    
        idxGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(in_idx) + startBthNum * M * sn, curBthNum * M * sn);                    

        pipe.InitBuffer(xyzVecIn, BUFF_NUM, ceil(N * dim3, BYTE_PER_BLK/sizeof(T)) * sizeof(T));
        pipe.InitBuffer(ctrVecIn, BUFF_NUM, ceil(M * dim3, BYTE_PER_BLK/sizeof(T)) * sizeof(T));
        pipe.InitBuffer(idxVecOut, BUFF_NUM, M * ceil(sn, BYTE_PER_BLK/sizeof(T)) * sizeof(int32_t));
        
        xyzLocal = xyzVecIn.AllocTensor<T>();
        ctrLocal = ctrVecIn.AllocTensor<T>();
        idxLocal = idxVecOut.AllocTensor<int32_t>();
    }

    __aicore__ inline void SeqInit(){
        pipe.InitBuffer(tmpBuffIdx, (maxNM * dim3 + align32N + align32N + M * sn32) * sizeof(int32_t));
        idxSeq = tmpBuffIdx.Get<int32_t>(maxNM * dim3 + align32N + align32N + M * sn32);
        idxSeq0 = idxSeq;
        idxSeq1 = idxSeq0[maxNM];
        idxSeq2 = idxSeq1[maxNM];
        idxSeqN = idxSeq2[maxNM];
        idxTmp1  = idxSeqN[align32N];
        idxTmp2  = idxTmp1[align32N];

        Arange(idxSeq0, 0, int32_t(dim3 * sizeof(T)), maxNM);
        Arange(idxSeq1, int32_t(sizeof(T)), int32_t(dim3 * sizeof(T)), maxNM);
        Arange(idxSeq2, int32_t(sizeof(T) + sizeof(T)), int32_t(dim3 * sizeof(T)), maxNM);
        Arange(idxSeqN, 0, 1, N);
    }

    __aicore__ inline void MaskInit(){
            pipe.InitBuffer(tmpBuffMask, (alignMaskLen + alignMaskLen) * sizeof(uint8_t));
            pipe.InitBuffer(tmpBuffResMask, NumU32OfResBits * sizeof(uint32_t));
            pipe.InitBuffer(tmpBuffTrans, (align32N * dim3 + align256N * dim3 + align32M * dim3) * sizeof(T));
            maskLocal = tmpBuffMask.Get<uint8_t>(alignMaskLen + alignMaskLen);
            resMaskLocal = tmpBuffResMask.Get<uint32_t>(NumU32OfResBits);
            Duplicate(resMaskLocal, (uint32_t)(0xffffffff >> (UINT32_BIT - sn)), NumU32OfResBits);
            transLocal = tmpBuffTrans.Get<T>(align32N * dim3 + align256N * dim3 + align32M * dim3);
    }

    template<typename CP_T> __aicore__ inline void MyCopy(const LocalTensor<CP_T> dstLocal, const LocalTensor<CP_T>srcLocal, uint32_t calCount){
        uint64_t elePerRepeat = 256/sizeof(CP_T);
        uint64_t alignEle = calCount / elePerRepeat * elePerRepeat;
        uint64_t remainEle = calCount - alignEle;
        uint64_t startEle = 0;
        while (startEle < alignEle / MAX_REPEAT * MAX_REPEAT) {
            Copy(dstLocal[startEle], srcLocal, elePerRepeat, MAX_REPEAT, {1, 1, 8, 8});
            startEle += MAX_REPEAT * elePerRepeat;
        }
        if(startEle < alignEle)
            Copy(dstLocal, srcLocal, elePerRepeat, (uint8_t)(startEle - alignEle), {1, 1, 8, 8});
        if(remainEle > 0)
            Copy(dstLocal, srcLocal, remainEle, (uint8_t)1, {1, 1, 8, 8});
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFF_NUM> xyzVecIn, ctrVecIn;
    TQue<QuePosition::VECOUT, BUFF_NUM>  idxVecOut;
    TBuf<QuePosition::VECCALC> tmpBuffTrans, tmpBuffIdx, tmpBuffMask, tmpBuffResMask;

    //global tensor
    GlobalTensor<T> xyzGm;         // xyz
    GlobalTensor<T> ctrGm;         // center_xyz
    GlobalTensor<int32_t> idxGm;   // return idx
    //VECIN
    LocalTensor<T> xyzLocal;
    LocalTensor<T> ctrLocal;
    //VECOUT
    LocalTensor<int32_t> idxLocal;
    //VECCALC
    LocalTensor<T> transLocal;
    LocalTensor<uint8_t> maskLocal;
    LocalTensor<uint32_t> resMaskLocal;
    LocalTensor<int32_t> idxSeq;

    LocalTensor<int32_t> idxSeq0 ;
    LocalTensor<int32_t> idxSeq1 ;
    LocalTensor<int32_t> idxSeq2 ;
    LocalTensor<int32_t> idxSeqN ;
    LocalTensor<int32_t> idxTmp1 ;
    LocalTensor<int32_t> idxTmp2 ;

private:
    uint32_t B;
    uint32_t N;
    uint32_t M;
    float minr;         // min_radius
    float maxr;         // max_radius
    int32_t sn;         // sample_num
    uint32_t usedCoreNum;
    uint32_t alignNum32; 
    uint32_t alignNum256;
    uint32_t align32N;
    uint32_t align32M;
    uint32_t align256N;
    uint32_t maxNM;
    uint32_t sn32;
    uint32_t startBthNum;
    uint32_t curBthNum;
    uint32_t alignMaskLen;
    uint32_t NumU32OfResBits;
    int32_t dim3 = 3;
    T px, py, pz;
};
#endif