/**
* @file three_nn.cpp
*
* Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#include <type_traits>
#include "kernel_operator.h"

#define K_MAX_SHAPE_DIM 0

using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;                                   

class KernelFastGeluGrad {
public:
    __aicore__ inline KernelFastGeluGrad() {}
    __aicore__ inline void Init(GM_ADDR xyz1, GM_ADDR xyz2, GM_ADDR dist, GM_ADDR indices,
                                uint32_t CoreDataNum, uint32_t finalTileNum, uint32_t tileDataNum, uint32_t TailDataNum,
                                uint32_t shapeB, uint32_t shapeN, uint32_t shapeM) {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");

        this->coreDataNum = CoreDataNum;
        this->tileNum = finalTileNum;
        this->tileDataNum = tileDataNum;
        this->tailDataNum = TailDataNum;

        this->shapeB = shapeB;
        this->shapeN = shapeN;
        this->shapeM = shapeM;

        xyz1Gm.SetGlobalBuffer((__gm__ DTYPE_XYZ1*)xyz1, this->coreDataNum);
        xyz2Gm.SetGlobalBuffer((__gm__ DTYPE_XYZ2*)xyz2, this->coreDataNum);
        distGm.SetGlobalBuffer((__gm__ DTYPE_DIST*)dist, this->coreDataNum);
        indicesGm.SetGlobalBuffer((__gm__ DTYPE_INDICES*)indices, this->coreDataNum);
        
        pipe.InitBuffer(inQueueXYZ2, BUFFER_NUM, this->tileDataNum * sizeof(DTYPE_XYZ2)+128);
        pipe.InitBuffer(QueueTmp1, this->tileDataNum * sizeof(DTYPE_XYZ2)+128);
        pipe.InitBuffer(QueueTmp2, this->tileDataNum * sizeof(DTYPE_XYZ2)+128);
        pipe.InitBuffer(QueueTmp3, this->tileDataNum * sizeof(DTYPE_XYZ2)+128);

        pipe.InitBuffer(QueueTmp10, 32 * sizeof(DTYPE_XYZ2));

        pipe.InitBuffer(QueueTmpX, this->tileDataNum * sizeof(DTYPE_XYZ2)+128);
        pipe.InitBuffer(QueueTmpY, this->tileDataNum * sizeof(DTYPE_XYZ2)+128);
        pipe.InitBuffer(QueueTmpZ, this->tileDataNum * sizeof(DTYPE_XYZ2)+128);

        pipe.InitBuffer(QueueMask, sizeof(uint32_t)*(128+36));
    }

    __aicore__ inline void Process() {
        int32_t loopCount;
        int32_t b=0; 
        
        LocalTensor<uint32_t> mask_scl = QueueMask.Get<uint32_t>();
        auto tmpx = QueueTmpX.Get<float>();
        auto tmpy = QueueTmpY.Get<float>();
        auto tmpz = QueueTmpZ.Get<float>();

        uint64_t mask1[2] = {10540996613548315209, 5270498306774157604};
        uint64_t mask2[2] = {2635249153387078802, 10540996613548315209};
        uint64_t mask3[2] = {5270498306774157604, 2635249153387078802};
        uint32_t scalar = 1227133513;
        Duplicate(mask_scl, scalar, mask1, 2, 1, 8 );

        scalar = 2454267026;
        Duplicate(mask_scl, scalar, mask2, 2, 1, 8 );

        scalar = 613566756;
        Duplicate(mask_scl, scalar, mask3, 2, 1, 8 );

        this->processDataNum = this->tailDataNum;
        for(int32_t b=0; b<this->shapeB; b++)
        {
            CopyIn(b, 0);
            Compute(b, 0, mask_scl, tmpx, tmpy, tmpz);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t b, int32_t progress)
    {
        LocalTensor<DTYPE_XYZ2> xyz2Local = inQueueXYZ2.AllocTensor<DTYPE_XYZ2>();
        DataCopy(xyz2Local, xyz2Gm[3*b*this->shapeM], (3*this->shapeM+31)/32*32);
        inQueueXYZ2.EnQue(xyz2Local);
    }
    __aicore__ inline void Compute(int32_t b, int32_t progress, LocalTensor<uint32_t> mask_scl, LocalTensor<float> tmpx, LocalTensor<float> tmpy, LocalTensor<float> tmpz)
    {
        LocalTensor<DTYPE_XYZ2> xyz2Local = inQueueXYZ2.DeQue<DTYPE_XYZ2>();
        auto tmp1 = QueueTmp1.Get<float>();
        auto tmp2 = QueueTmp2.Get<float>();
        auto tmp3 = QueueTmp3.Get<float>();
        auto tmp10 = QueueTmp10.Get<float>();

        float ux,uy,uz;
        int besti1, besti2, besti3;
        uint32_t calculate_num = this->shapeM;
        uint32_t mask = this->shapeM*3;
        uint64_t rsvdCnt = 0;
        GatherMask (tmpx, xyz2Local, mask_scl, true, mask, { 1, 1, 8, 8 }, rsvdCnt);
        GatherMask (tmpy, xyz2Local, mask_scl[16], true, mask, { 1, 1, 8, 8 }, rsvdCnt);
        GatherMask (tmpz, xyz2Local, mask_scl[8], true, mask, { 1, 1, 8, 8 }, rsvdCnt);

        uint32_t b_xyz1_index = 3*(b*this->shapeN);
        for(uint32_t i=0; i < this->shapeN; i++)
        {
            ux = -xyz1Gm.GetValue(b_xyz1_index);
            uy = -xyz1Gm.GetValue(b_xyz1_index+1);
            uz = -xyz1Gm.GetValue(b_xyz1_index+2);

            Adds(tmp3, tmpx, ux, calculate_num);
            Mul(tmp2, tmp3, tmp3, calculate_num);
            
            Adds(tmp3, tmpy, uy, calculate_num);
            Mul(tmp1, tmp3, tmp3, calculate_num);
            
            Add(tmp3, tmp2, tmp1, calculate_num);

            Adds(tmp2, tmpz, uz, calculate_num);
            Mul(tmp1, tmp2, tmp2, calculate_num);

            Add(tmp3, tmp3, tmp1, calculate_num);

            ReduceMin(tmp2, tmp3, tmp1, calculate_num, true);
            besti1 = (tmp2.ReinterpretCast<int32_t>()).GetValue(1);
            tmp3.SetValue(besti1, 3.402823466e+38F);

            ReduceMin(tmp2[8], tmp3, tmp1, calculate_num, true);
            besti2 = (tmp2.ReinterpretCast<int32_t>()).GetValue(9);
            tmp3.SetValue(besti2, 3.402823466e+38F);

            ReduceMin(tmp2[16], tmp3, tmp1, calculate_num, true);
            besti3 = (tmp2.ReinterpretCast<int32_t>()).GetValue(17);

            indicesGm.SetValue(b_xyz1_index, besti1);
            indicesGm.SetValue(b_xyz1_index+1, besti2);
            indicesGm.SetValue(b_xyz1_index+2, besti3);

            distGm.SetValue(b_xyz1_index, tmp2.GetValue(0));
            distGm.SetValue(b_xyz1_index+1, tmp2.GetValue(8));
            distGm.SetValue(b_xyz1_index+2, tmp2.GetValue(16));
            b_xyz1_index+=3;
        }
        inQueueXYZ2.FreeTensor(xyz2Local);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueXYZ2;
    TBuf<QuePosition::VECCALC> QueueTmp1, QueueTmp2, QueueTmp3, QueueTmpX, QueueTmpY, QueueTmpZ, QueueMask, QueueTmp10;
    GlobalTensor<DTYPE_XYZ1> xyz1Gm;
    GlobalTensor<DTYPE_XYZ2> xyz2Gm;
    GlobalTensor<DTYPE_DIST> distGm;
    GlobalTensor<DTYPE_INDICES> indicesGm;
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
    uint32_t xyz2DataNum;
    uint32_t shapeB;
    uint32_t shapeN;
    uint32_t shapeM;
};

extern "C" __global__ __aicore__ void three_nn(GM_ADDR xyz1, GM_ADDR xyz2, GM_ADDR dist, GM_ADDR indices, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);

    KernelFastGeluGrad op;
    op.Init(xyz1, xyz2, dist, indices, 
            tiling_data.CoreDataNum, tiling_data.finalTileNum, tiling_data.tileDataNum, tiling_data.TailDataNum, 
            tiling_data.shapeB, tiling_data.shapeN, tiling_data.shapeM);  
    op.Process();
}
