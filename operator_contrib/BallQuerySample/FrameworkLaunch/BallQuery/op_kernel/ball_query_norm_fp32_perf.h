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
#ifndef __BALL_QUERY_NORM_FP32_PERF_H_
#define __BALL_QUERY_NORM_FP32_PERF_H_

#include "kernel_operator.h"
#include "utils.h"
#include "ball_query_norm_fp32_perf.h"

using namespace AscendC;

template<typename T> class BallQueryNormFP32Perf {
public:
    __aicore__ inline BallQueryNormFP32Perf() {}
    __aicore__ inline void Init(GM_ADDR in_xyz, GM_ADDR in_ctr, GM_ADDR in_idx, const BallQueryTilingData* __restrict tiling) 
    {
        this->B = tiling->B;
        this->N = tiling->N;
        this->M = tiling->M;
        this->minr = (half)(tiling->minr * tiling->minr);
        this->maxr = (half)(tiling->maxr * tiling->maxr);
        this->sn = tiling->sn;
        this->usedCoreNum = GetBlockNum();
        alignFP32N = ceil(N, FP32_CNT_PER_BLK);     
        alignFP16N = ceil(N, FP16_CNT_PER_BLK);

        align256N = ceil(N, FP16_CNT_PER_BIG_BLK);
        alignINT32sn = ceil(sn, INT32_CNT_PER_BLK);  
        sn32 = ceil(sn, BYTE_PER_BLK);                     // resMask is uint32, a number represent 32 indexs, so sn is need to align to 32             
        alignMaskLen = div_ceil(align256N, INT32_CNT_PER_BLK);            //align256N bits require alignMaskLen uint8 element represent
        alignMaskLen = ceil(alignMaskLen, BYTE_PER_BLK);   // 32Bytes align,  32Byte/sizeof(uint8)

        NumU32OfResBits = ceil(div_ceil(M * sn32, BYTE_PER_BLK), INT32_CNT_PER_BLK);
        uint32_t totalCtr = B * M;
        uint32_t formerCtrNumPerCore = div_ceil(totalCtr, usedCoreNum);
        uint32_t tailCtrNumPerCore = totalCtr / usedCoreNum;
        uint32_t formerCoreNum = totalCtr % usedCoreNum;
        if(tailCtrNumPerCore == 0){
            usedCoreNum = formerCoreNum;
        }

        if(GetBlockIdx() < formerCoreNum){
            startCtrIdx = formerCtrNumPerCore * GetBlockIdx();
            curCtrNum = formerCtrNumPerCore;
        }
        else{
            startCtrIdx = tailCtrNumPerCore * GetBlockIdx() + formerCoreNum;
            curCtrNum = tailCtrNumPerCore;
        }
        startBthIdx = startCtrIdx / M;
        curBthNum = (startCtrIdx % M > 0 ? 1 : 0) + div_ceil(startCtrIdx + curCtrNum - ceil(startCtrIdx, M), M);
        if(GetBlockIdx() < usedCoreNum){
            InOutInit(in_xyz, in_ctr, in_idx);
            SeqInit();
            MaskInit();
        }
    }

    __aicore__ inline void Process() {
        if(GetBlockIdx() < usedCoreNum){
            uint32_t startIdx = 0, curNum = 0;
            for (uint32_t bth = 0; bth < curBthNum; ++bth) { // handle a batch per loop
                curNum = floor(startCtrIdx + startIdx, M) + M - (startCtrIdx + startIdx);
                if(startIdx + curNum > curCtrNum){
                    curNum = curCtrNum - startIdx;
                }
                CopyIn(bth, startIdx, curNum);
                Compute(bth, startIdx, curNum);
                CopyOut(bth, startIdx, curNum);
                startIdx += curNum;
            }
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t bth, uint32_t startIdx, uint32_t curNum) {           //load a batch xyz and ctr from gm
        DataCopyPad(xyzLocal, xyzGm[bth * N * dim3], {1, static_cast<uint32_t>(N * dim3 * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
        DataCopyPad(ctrLocal, ctrGm[startIdx * dim3], {1, static_cast<uint32_t>(curNum * dim3 * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
        xyzVecIn.EnQue(xyzLocal);
        ctrVecIn.EnQue(ctrLocal);
    }

    __aicore__ inline void Compute(uint32_t bth, uint32_t startIdx, uint32_t curNum) {
        uint64_t rsvdCnt = 0;
        xyzLocal = xyzVecIn.DeQue<T>();
        ctrLocal = ctrVecIn.DeQue<T>();
        Cast<half, float>(xyzLocalFP16, xyzLocal, RoundMode::CAST_NONE, N * dim3);
        Cast<half, float>(ctrLocalFP16, ctrLocal, RoundMode::CAST_NONE, curNum * dim3);
        PipeBarrier<PIPE_ALL>();
        Muls(ctrLocalFP16, ctrLocalFP16, (half)-1.0, curNum * dim3);
        Gather(xyz_x, xyzLocalFP16, idxSeq0.ReinterpretCast<uint32_t>(), 0, N);    
        Gather(xyz_y, xyzLocalFP16, idxSeq1.ReinterpretCast<uint32_t>(), 0, N);    
        Gather(xyz_z, xyzLocalFP16, idxSeq2.ReinterpretCast<uint32_t>(), 0, N); 
                
        uint32_t remain = curNum % m_repeat;
        uint32_t loops = (remain > 0 ? 1 : 0) + curNum / m_repeat;
        uint32_t tmp_repeat = m_repeat;
        uint32_t repeatLen = align256N * m_repeat;
        for (int i = 0; i < loops; ++i){
            if(i + 1 == loops && remain > 0){
                tmp_repeat = remain;
                repeatLen = align256N * remain;
            }
            int tmp = 0;
            for(int m = 0; m < tmp_repeat; ++m){
                tmp = dim3 * (i * m_repeat + m);
                Adds(tmp_x[align256N * m], xyz_x, ctrLocalFP16.GetValue(tmp), N);
                Adds(tmp_y[align256N * m], xyz_y, ctrLocalFP16.GetValue(tmp+1), N);
                Adds(tmp_z[align256N * m], xyz_z, ctrLocalFP16.GetValue(tmp+1+1), N);
            }
            
            Mul(tmp_x, tmp_x, tmp_x, repeatLen);
            Mul(tmp_y, tmp_y, tmp_y, repeatLen);
            Mul(tmp_z, tmp_z, tmp_z, repeatLen);
            Add(tmp_x, tmp_x, tmp_y, repeatLen);
            Add(tmp_x, tmp_x, tmp_z, repeatLen);

            CompareScalar(mask1Local, tmp_x, maxr, CMPMODE::LT, repeatLen);
            CompareScalar(mask2Local, tmp_x, minr, CMPMODE::GE, repeatLen);
            And(mask1Local.ReinterpretCast<uint16_t>(), mask1Local.ReinterpretCast<uint16_t>(), mask2Local.ReinterpretCast<uint16_t>(), (alignMaskLen / sizeof(uint16_t)) * tmp_repeat);  // uint8_t --> uint16_t  alignMaskLen / 2 
            int tmp2 = i * m_repeat;
            for(int m = 0; m < tmp_repeat; ++m){
                GatherMask(idxTmp2[(tmp2 + m) * sn32], idxSeqN, mask1Local[alignMaskLen * m].ReinterpretCast<uint32_t>(), true, N, {1,1,8,8}, rsvdCnt);
            }
        }
        
        GatherMask(idxLocal, idxTmp2, resMaskLocal, true, curNum * sn32, {1,1,8,8}, rsvdCnt);
        idxVecOut.EnQue<int32_t>(idxLocal);
    }

    __aicore__ inline void  CopyOut(uint32_t bth, uint32_t startIdx, uint32_t curNum) {
        idxLocal = idxVecOut.DeQue<int32_t>();
        DataCopyPad(idxGm[startIdx * sn], idxLocal, {1, (uint32_t)(curNum * sn * sizeof(int32_t)), 0, 0, 0});
    }

    __aicore__ inline void InOutInit(GM_ADDR in_xyz, GM_ADDR in_ctr, GM_ADDR in_idx){
        // input output Init
        xyzGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(in_xyz) + startBthIdx * N * dim3, curBthNum * N * dim3);                    
        ctrGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(in_ctr) + startCtrIdx * dim3, curCtrNum * dim3);                    
        idxGm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(in_idx) + startCtrIdx * sn, curCtrNum * sn);
        // VecIn/VecOut Init
        pipe.InitBuffer(xyzVecIn, BUFF_NUM, ceil(N * dim3, FP32_CNT_PER_BLK) * sizeof(T));     //sizeof(T) == 4
        pipe.InitBuffer(ctrVecIn, BUFF_NUM, ceil(M * dim3, FP32_CNT_PER_BLK) * sizeof(T));
        pipe.InitBuffer(idxVecOut, BUFF_NUM, M * alignINT32sn * sizeof(int32_t));    //* sizeof(int32_t) == *4 == <<2
        xyzLocal = xyzVecIn.AllocTensor<T>();
        ctrLocal = ctrVecIn.AllocTensor<T>();
        idxLocal = idxVecOut.AllocTensor<int32_t>();
    }

    __aicore__ inline void SeqInit(){
        uint32_t tmpBuffIdxSize = alignFP32N+ alignFP32N + alignFP32N + alignFP32N + alignFP32N + sn32 * M + N;
        pipe.InitBuffer(tmpBuffIdx, tmpBuffIdxSize * sizeof(int32_t));   //* sizeof(int32_t) == *4 == <<2
        idxSeq0 = tmpBuffIdx.Get<int32_t>(tmpBuffIdxSize);
        idxSeq1 = idxSeq0[alignFP32N];
        idxSeq2 = idxSeq1[alignFP32N];                  
        idxSeqN = idxSeq2[alignFP32N];
        idxTmp1 = idxSeqN[alignFP32N];                  
        idxTmp2 = idxTmp1[alignFP32N];

        Arange<int32_t>(idxSeq0, 0, dim3 * sizeof(half), N);
        Arange<int32_t>(idxSeq1, sizeof(half), dim3 * sizeof(half), N);
        Arange<int32_t>(idxSeq2, sizeof(half) + sizeof(half), dim3 * sizeof(half), N);
        Arange<int32_t>(idxSeqN, 0, 1, N);
    }

    __aicore__ inline void MaskInit(){
        pipe.InitBuffer(tmpBuffResMask, NumU32OfResBits * sizeof(uint32_t));
        resMaskLocal = tmpBuffResMask.Get<uint32_t>(NumU32OfResBits);

        pipe.InitBuffer(tmpBuffMask, (alignMaskLen + alignMaskLen) * m_repeat * sizeof(uint8_t));
        mask1Local = tmpBuffMask.Get<uint8_t>((alignMaskLen + alignMaskLen) * m_repeat);
        mask2Local = mask1Local[alignMaskLen * m_repeat];
        uint32_t ceilN3_16 = ceil(N * dim3, FP16_CNT_PER_BLK);
        uint32_t ceilM3_16 = ceil(M * dim3, FP16_CNT_PER_BLK);
        uint32_t tmpBuffTransSize = ceilN3_16 + ceilM3_16 + alignFP16N * dim3 + align256N * m_repeat * dim3;
        pipe.InitBuffer(tmpBuffTrans, tmpBuffTransSize * sizeof(half));
        xyzLocalFP16 = tmpBuffTrans.Get<half>(tmpBuffTransSize);
        ctrLocalFP16 = xyzLocalFP16[ceilN3_16];
        xyz_x = ctrLocalFP16[ceilM3_16];
        xyz_y = xyz_x[alignFP16N];
        xyz_z = xyz_y[alignFP16N];
        tmp_x = xyz_z[alignFP16N];
        tmp_y = tmp_x[align256N * m_repeat];
        tmp_z = tmp_y[align256N * m_repeat];
    
        Duplicate(resMaskLocal, (uint32_t)(0xffffffff >> (UINT32_BIT - sn)), NumU32OfResBits);
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
    LocalTensor<uint32_t> resMaskLocal;

    LocalTensor<int32_t> idxSeq0 ;
    LocalTensor<int32_t> idxSeq1 ;
    LocalTensor<int32_t> idxSeq2 ;
    LocalTensor<int32_t> idxSeqN ;
    LocalTensor<int32_t> idxTmp1 ;
    LocalTensor<int32_t> idxTmp2 ;

    LocalTensor<half> xyzLocalFP16;
    LocalTensor<half> ctrLocalFP16;
    LocalTensor<half> xyz_x;
    LocalTensor<half> xyz_y;
    LocalTensor<half> xyz_z;
    LocalTensor<half> tmp_x;
    LocalTensor<half> tmp_y;
    LocalTensor<half> tmp_z;

    LocalTensor<uint8_t> mask1Local;
    LocalTensor<uint8_t> mask2Local;

  private:
    uint32_t B;
    uint32_t N;
    uint32_t M;
    half minr;         // min_radius
    half maxr;         // max_radius
    int32_t sn;         // sample_num
    uint32_t usedCoreNum;
    uint32_t alignFP16N;
    uint32_t alignFP32N;
    uint32_t alignFP32M;
    uint32_t align256N;
    uint32_t maxNM;
    uint32_t alignINT32sn;
    uint32_t sn32;
    uint32_t NumU32OfResBits;

    uint32_t startCtrIdx;
    uint32_t curCtrNum;
    uint32_t startBthIdx;
    uint32_t curBthNum;
    uint32_t alignMaskLen;
    uint32_t m_repeat = 56;
    int32_t dim3 = 3;
};
#endif