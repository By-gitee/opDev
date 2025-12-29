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
#ifndef __BALL_QUERY_STACK_H_
#define __BALL_QUERY_STACK_H_
#include "kernel_operator.h"
#include "utils.h"
#include "ball_query_stack.h"

using namespace AscendC;


template<typename T> class BallQueryStack {
public:
    __aicore__ inline BallQueryStack() {}
    __aicore__ inline void Init(GM_ADDR in_xyz, GM_ADDR in_ctr, GM_ADDR in_xyz_bth, GM_ADDR in_ctr_bth, GM_ADDR in_idx, const BallQueryTilingData* __restrict tiling) 
    {
        this->B = tiling->B;
        this->N_total = tiling->N;
        this->M_total = tiling->M;
        this->minr = tiling->minr * tiling->minr;
        this->maxr = tiling->maxr * tiling->maxr;
        this->sn = tiling->sn;
        this->bn = tiling->bn;
        this->B = tiling->bn;
        HandleBatch(in_xyz_bth, in_ctr_bth);
        align32N = ceil(N, BYTE_PER_BLK / sizeof(T));
        align32M = ceil(M, BYTE_PER_BLK / sizeof(T));
        align256N = ceil(M, BYTE_PER_BIG_BLK / sizeof(T));
        maxNM = ceil(N > M ? N : M, BYTE_PER_BLK / sizeof(T));
        alignMaskLen = ceil(align256N / UINT8_BIT, INT8_CNT_PER_BLK);

        usedCoreNum = GetBlockNum();   
        uint32_t formerBthNumPerCore = 0, tailBthNumPerCore = 0, formerCoreNum = 0, tailCoreNum = 0;
        if (usedCoreNum > 0){
            formerBthNumPerCore = div_ceil(B, usedCoreNum), tailBthNumPerCore = B / usedCoreNum;
            formerCoreNum = B % usedCoreNum, tailCoreNum = usedCoreNum - formerCoreNum;
        }
        usedCoreNum = (tailBthNumPerCore > 0 ? formerCoreNum : (formerCoreNum + tailCoreNum));

        uint32_t startN, cntN, startM, cntM;
        if(GetBlockIdx() < formerCoreNum){
            startBthNum = formerBthNumPerCore * GetBlockIdx();
            curBthNum = formerBthNumPerCore;
        }else{
            startBthNum = formerBthNumPerCore * formerCoreNum + tailBthNumPerCore * (GetBlockIdx() - formerCoreNum);
            curBthNum = tailBthNumPerCore;
        }

        startN = xyzBthLocal(startBthNum);
        startM = ctrBthLocal(startBthNum);
        cntN = xyzBthLocal(startBthNum + curBthNum) - xyzBthLocal(startBthNum);
        cntM = ctrBthLocal(startBthNum + curBthNum) - ctrBthLocal(startBthNum);

        xyzGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(in_xyz) + startN + startN + startN, cntN + cntN + cntN);                    
        ctrGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(in_ctr) + startM + startM + startM, cntM + cntM + cntM);                    
        idxGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(in_idx) + startM * sn, cntM * sn);                    

        pipe.InitBuffer(xyzVecIn, BUFF_NUM, ceil(cntN + cntN + cntN, BYTE_PER_BLK/sizeof(T)) * sizeof(T));
        pipe.InitBuffer(ctrVecIn, BUFF_NUM, ceil(cntM + cntM + cntM, BYTE_PER_BLK/sizeof(T)) * sizeof(T));
        pipe.InitBuffer(idxVecOut, BUFF_NUM, cntM * ceil(sn, BYTE_PER_BLK/sizeof(T)) * sizeof(int32_t));

        pipe.InitBuffer(tmpBuffTrans, (align32N + align32N + align32N + align256N + align256N + align256N + align32M + align32M + align32M) * sizeof(T));
        pipe.InitBuffer(tmpBuffMask, alignMaskLen + alignMaskLen * sizeof(uint8_t));
        pipe.InitBuffer(tmpBuffIdx, (maxNM + maxNM + maxNM + align32N + align32N) * sizeof(int32_t));

        xyzLocal = xyzVecIn.AllocTensor<T>();
        ctrLocal = ctrVecIn.AllocTensor<T>();
        idxLocal = idxVecOut.AllocTensor<int32_t>();

        transLocal = tmpBuffTrans.Get<T>(align32N + align32N + align32N + align256N + align256N + align256N + align32M + align32M + align32M);
        maskLocal = tmpBuffMask.Get<uint8_t>(alignMaskLen  + alignMaskLen);
        idxSeq = tmpBuffIdx.Get<int32_t>(maxNM + maxNM + maxNM + align32N + align32N);
    }

    __aicore__ inline void Process() {
        if(GetBlockIdx() < usedCoreNum){
            uint32_t xyzStart = 0, xyzLen = 0;
            uint32_t ctrStart = 0, ctrLen = 0;
            for (uint32_t bth = 0; bth < curBthNum; ++bth) { // handle a batch per loop
                xyzStart = xyzBthLocal(startBthNum + bth) - xyzBthLocal(startBthNum);
                xyzLen = xyzBthLocal(startBthNum + bth + 1) - xyzBthLocal(startBthNum + bth);
                ctrStart = ctrBthLocal(startBthNum + bth) - ctrBthLocal(startBthNum);
                ctrLen = ctrBthLocal(startBthNum + bth + 1) - ctrBthLocal(startBthNum + bth);
                CopyIn(bth, xyzStart, xyzLen, ctrStart, ctrLen);
                Compute(bth, xyzStart, xyzLen, ctrStart, ctrLen);
                CopyOut(bth, xyzStart, xyzLen, ctrStart, ctrLen);
            }
            xyzVecIn.FreeTensor(xyzLocal);
            ctrVecIn.FreeTensor(ctrLocal);
            idxVecOut.FreeTensor(idxLocal);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t bth, uint32_t xyzStart, uint32_t xyzLen, uint32_t ctrStart, uint32_t ctrLen) {           //load a batch xyz and ctr from gm
        uint32_t alignXyzLen= ceil(xyzLen, BYTE_PER_BLK/sizeof(T));
        uint32_t alignCtrLen= ceil(ctrLen, BYTE_PER_BLK/sizeof(T));
        DataCopy(xyzLocal, xyzGm[xyzStart + xyzStart + xyzStart], alignXyzLen + alignXyzLen + alignXyzLen);
        DataCopy(ctrLocal, ctrGm[ctrStart + ctrStart + ctrStart], alignCtrLen + alignCtrLen + alignCtrLen);
        xyzVecIn.EnQue(xyzLocal);
        ctrVecIn.EnQue(ctrLocal);
    }

    __aicore__ inline void Compute(uint32_t bth, uint32_t xyzStart, uint32_t xyzLen, uint32_t ctrStart, uint32_t ctrLen) {
        xyzLocal = xyzVecIn.DeQue<T>(), ctrLocal = ctrVecIn.DeQue<T>();
        LocalTensor<T> xyz_x = transLocal;
        LocalTensor<T> xyz_y = xyz_x[align32N];
        LocalTensor<T> xyz_z = xyz_y[align32N];

        LocalTensor<T> tmp_x = xyz_z[align32N];
        LocalTensor<T> tmp_y = tmp_x[align256N];
        LocalTensor<T> tmp_z = tmp_y[align256N];

        LocalTensor<uint8_t> mask1Local = maskLocal;
        LocalTensor<uint8_t> mask2Local = maskLocal[alignMaskLen];

        LocalTensor<int32_t> idxSeq0 = idxSeq;
        LocalTensor<int32_t> idxSeq1 = idxSeq0[maxNM];
        LocalTensor<int32_t> idxSeq2 = idxSeq1[maxNM];
        LocalTensor<int32_t> idxSeqN = idxSeq2[maxNM];
        LocalTensor<int32_t> idxTmp1 = idxSeqN[align32N];

        Arange(idxSeq0, 0, int32_t(sizeof(T) + sizeof(T) + sizeof(T)), maxNM);
        Arange(idxSeq1, int32_t(sizeof(T)), int32_t(sizeof(T) + sizeof(T) + sizeof(T)), maxNM);
        Arange(idxSeq2, int32_t(sizeof(T) + sizeof(T)), int32_t(sizeof(T) + sizeof(T) + sizeof(T)), maxNM);
        Arange(idxSeqN, 0, 1, N);

        Gather(xyz_x, xyzLocal, idxSeq0.ReinterpretCast<uint32_t>(), 0, N);    
        Gather(xyz_y, xyzLocal, idxSeq1.ReinterpretCast<uint32_t>(), 0, N);    
        Gather(xyz_z, xyzLocal, idxSeq2.ReinterpretCast<uint32_t>(), 0, N);    

        Duplicate(tmp_x, (T)MAX_HALF, align256N);
        for (int i = 0; i < ctrLen; ++i){
            T px = -1 * (float)ctrLocal(i+i+i);
            T py = -1 * (float)ctrLocal(i+i+i+1);
            T pz = -1 * (float)ctrLocal(i+i+i+1+1);

            Adds(tmp_x, xyz_x, px, N);
            Adds(tmp_y, xyz_y, py, N);
            Adds(tmp_z, xyz_z, pz, N);
            
            Mul(tmp_x, tmp_x, tmp_x, N);
            Mul(tmp_y, tmp_y, tmp_y, N);
            Mul(tmp_z, tmp_z, tmp_z, N);

            Add(tmp_x, tmp_x, tmp_y, N);
            Add(tmp_x, tmp_x, tmp_z, N);
            
            CompareScalar(mask1Local, tmp_x, (T)maxr, CMPMODE::LT, align256N);
            
            uint64_t rsvdCnt = 0;
            GatherMask(idxTmp1, idxSeqN, mask1Local.ReinterpretCast<uint32_t>(), true, xyzLen, {1,1,8,8}, rsvdCnt);
            if(sn < rsvdCnt){
                rsvdCnt = sn;
            }
            else{
                Duplicate(idxLocal[i * ceil(sn, BYTE_PER_BLK / sizeof(T))], (int32_t)idxTmp1(0), sn);
            }
            MyCopy(idxLocal[i * ceil(sn, BYTE_PER_BLK / sizeof(T))], idxTmp1, rsvdCnt);
        }
        idxVecOut.EnQue(idxLocal);
    }

    __aicore__ inline void  CopyOut(uint32_t bth, uint32_t xyzStart, uint32_t xyzLen, uint32_t ctrStart, uint32_t ctrLen) {
        idxLocal = idxVecOut.DeQue<int32_t>();
        for(int i = 0; i < ctrLen; ++i){
            DataCopyPad(idxGm[(ctrStart + i) * sn], idxLocal[i * ceil(sn, BYTE_PER_BLK/sizeof(T))], {1, (uint16_t)(sn * sizeof(int32_t)), 1, 1});
        }
    }

    template<typename CP_T> __aicore__ inline void MyCopy(const LocalTensor<CP_T> dstLocal, const LocalTensor<CP_T>srcLocal, uint32_t calCount){
        uint64_t elePerRepeat = BYTE_PER_BIG_BLK / sizeof(CP_T);
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

    __aicore__ inline void HandleBatch(GM_ADDR in_xyz_bth,  GM_ADDR in_ctr_bth){
        GlobalTensor<int32_t> xyzBthGm;      // xyz batch
        GlobalTensor<int32_t> ctrBthGm;      // center_xyz batch
        int32_t tmp_bn = 1;
        tmp_bn += bn;
        xyzBthGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(in_xyz_bth), ceil(tmp_bn, BYTE_PER_BLK/sizeof(T)));                    
        ctrBthGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(in_ctr_bth), ceil(tmp_bn, BYTE_PER_BLK/sizeof(T)));   

        pipe.InitBuffer(xyzBthVecIn, BUFF_NUM, ceil(tmp_bn, BYTE_PER_BLK/sizeof(T)) * sizeof(int32_t));                 
        pipe.InitBuffer(ctrBthVecIn, BUFF_NUM, ceil(tmp_bn, BYTE_PER_BLK/sizeof(T)) * sizeof(int32_t));      

        xyzBthLocal = xyzBthVecIn.AllocTensor<int32_t>();           
        ctrBthLocal = ctrBthVecIn.AllocTensor<int32_t>(); 

        DataCopy(xyzBthLocal, xyzBthGm, ceil(tmp_bn, BYTE_PER_BLK/sizeof(T)));          
        DataCopy(ctrBthLocal, ctrBthGm, ceil(tmp_bn, BYTE_PER_BLK/sizeof(T)));         

        N =  xyzBthLocal(0) > xyzBthLocal(1) ? xyzBthLocal(0) : xyzBthLocal(1);
        M =  ctrBthLocal(0) > ctrBthLocal(1) ? ctrBthLocal(0) : ctrBthLocal(1);

        int32_t xyzcur = 0, xyzlast = xyzBthLocal(1);
        int32_t ctrcur = 0, ctrlast = ctrBthLocal(1);
        int32_t start = 2;
        for(int i = start; i <= bn; ++i){
            xyzcur = xyzBthLocal(i);
            ctrcur = ctrBthLocal(i);
            xyzBthLocal(i) = xyzlast + xyzBthLocal(i-start);
            ctrBthLocal(i) = ctrlast + ctrBthLocal(i-start);
            xyzlast = xyzcur;
            ctrlast = ctrcur;
            if(xyzcur > N)  {N = xyzcur;}
            if(xyzcur > M)  {M = xyzcur;}
        }
        xyzBthLocal(1) = xyzBthLocal(0);
        ctrBthLocal(1) = ctrBthLocal(0);
        xyzBthLocal(0) = 0;
        ctrBthLocal(0) = 0;
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFF_NUM> xyzVecIn, ctrVecIn, xyzBthVecIn, ctrBthVecIn;
    TQue<QuePosition::VECOUT, BUFF_NUM>  idxVecOut;
    TBuf<QuePosition::VECCALC> tmpBuffTrans, tmpBuffIdx, tmpBuffMask;

    //global tensor
    GlobalTensor<T> xyzGm;         // xyz
    GlobalTensor<T> ctrGm;         // center_xyz
    GlobalTensor<int32_t> idxGm;   // return idx
    //VECIN
    LocalTensor<T> xyzLocal;
    LocalTensor<T> ctrLocal;
    LocalTensor<int32_t> xyzBthLocal;
    LocalTensor<int32_t> ctrBthLocal;
    //VECOUT
    LocalTensor<int32_t> idxLocal;
    //VECCALC
    LocalTensor<T> transLocal;
    LocalTensor<uint8_t> maskLocal;
    LocalTensor<int32_t> idxSeq;

private:
    uint32_t B;
    uint32_t N;
    uint32_t M;
    uint32_t N_total;
    uint32_t M_total;
    float minr;         // min_radius
    float maxr;         // max_radius
    int32_t sn;         // sample_num
    int32_t bn;         // batch number
    uint32_t align32N;
    uint32_t align32M;
    uint32_t align256N;
    uint32_t maxNM;
    uint32_t startBthNum;
    uint32_t curBthNum;
    uint32_t alignMaskLen;
    uint32_t usedCoreNum;
    int32_t dim3 = 3;
};

#endif