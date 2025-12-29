/**
 * @file add_custom.cpp
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "add_custom_tiling.h"
#include "kernel_operator.h"

constexpr uint32_t BUFFER_NUM = 2; // tensor num for each queue

constexpr uint32_t ADD_BFLOAT16 = 0;
constexpr uint32_t ADD_FLOAT16 = 1;
constexpr uint32_t ADD_FLOAT32 = 2;
constexpr uint32_t ADD_INT8 = 3;
constexpr uint32_t ADD_INT16 = 4;
constexpr uint32_t ADD_INT32 = 5;

constexpr uint32_t LAST_TWO_TILE = 2;

template <typename dataType> class KernelAdd;
template <> class KernelAdd <bfloat16_t> {
public:
    __aicore__ inline KernelAdd() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, AddCustomTilingData tiling)
    {
        if (tiling.isEvenCore) {
            this->blockLength = tiling.blockLength;
            this->tileNum = tiling.tileNum;
            this->tileLength = tiling.tileLength / BUFFER_NUM;
            this->lastTileLength = tiling.lastTileLength;

            uint64_t offset = this->blockLength * AscendC::GetBlockIdx();
            xGm.SetGlobalBuffer((__gm__ bfloat16_t *)x + offset, this->blockLength);
            yGm.SetGlobalBuffer((__gm__ bfloat16_t *)y + offset, this->blockLength);
            zGm.SetGlobalBuffer((__gm__ bfloat16_t *)z + offset, this->blockLength);
        } else {
            if (AscendC::GetBlockIdx() < tiling.formerNum) {
                this->tileNum = tiling.formerTileNum;
                this->tileLength = tiling.formerTileLength / BUFFER_NUM;
                this->lastTileLength = tiling.formerLastTileLength;

                uint64_t offset = tiling.formerLength * AscendC::GetBlockIdx();
                xGm.SetGlobalBuffer((__gm__ bfloat16_t *)x + offset, tiling.formerLength);
                yGm.SetGlobalBuffer((__gm__ bfloat16_t *)y + offset, tiling.formerLength);
                zGm.SetGlobalBuffer((__gm__ bfloat16_t *)z + offset, tiling.formerLength);
            } else {
                this->tileNum = tiling.tailTileNum;
                this->tileLength = tiling.tailTileLength / BUFFER_NUM;
                this->lastTileLength = tiling.tailLastTileLength;

                uint64_t offset = tiling.formerLength * tiling.formerNum + tiling.tailLength *
                    (AscendC::GetBlockIdx() - tiling.formerNum);
                xGm.SetGlobalBuffer((__gm__ bfloat16_t *)x + offset, tiling.tailLength);
                yGm.SetGlobalBuffer((__gm__ bfloat16_t *)y + offset, tiling.tailLength);
                zGm.SetGlobalBuffer((__gm__ bfloat16_t *)z + offset, tiling.tailLength);
            }
        }

        this->initBufferLength = AscendC::Std::max(this->tileLength, this->lastTileLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->initBufferLength * sizeof(bfloat16_t));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->initBufferLength * sizeof(bfloat16_t));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->initBufferLength * sizeof(bfloat16_t));

        pipe.InitBuffer(tmpBuf0, this->initBufferLength * sizeof(float));
        pipe.InitBuffer(tmpBuf1, this->initBufferLength * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        // 整块进行double buffer计算
        uint32_t loopCount = this->tileNum * BUFFER_NUM;
        for (uint32_t i = 0; i < loopCount; i++) {
            CopyIn(i, this->tileLength);
            Compute(i, this->tileLength);
            CopyOut(i, this->tileLength);
        }

        // 进行尾块计算, 不做double buffer操作
        if (this->lastTileLength > 0) {
            CopyIn(loopCount, this->lastTileLength);
            Compute(loopCount, this->lastTileLength);
            CopyOut(loopCount, this->lastTileLength);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress, uint32_t tileLength)
    {
        AscendC::LocalTensor<bfloat16_t> xLocal = inQueueX.AllocTensor<bfloat16_t>();
        AscendC::LocalTensor<bfloat16_t> yLocal = inQueueY.AllocTensor<bfloat16_t>();

        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], tileLength);
        AscendC::DataCopy(yLocal, yGm[progress * this->tileLength], tileLength);

        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute(uint32_t progress, uint32_t tileLength)
    {
        AscendC::LocalTensor<bfloat16_t> xLocal = inQueueX.DeQue<bfloat16_t>();
        AscendC::LocalTensor<bfloat16_t> yLocal = inQueueY.DeQue<bfloat16_t>();
        AscendC::LocalTensor<bfloat16_t> zLocal = outQueueZ.AllocTensor<bfloat16_t>();

        AscendC::LocalTensor<float> tmpTensor0 = tmpBuf0.Get<float>();
        AscendC::LocalTensor<float> tmpTensor1 = tmpBuf1.Get<float>();

        AscendC::Cast(tmpTensor0, xLocal, AscendC::RoundMode::CAST_NONE, tileLength);
        AscendC::Cast(tmpTensor1, yLocal, AscendC::RoundMode::CAST_NONE, tileLength);

        AscendC::Add(tmpTensor0, tmpTensor0, tmpTensor1, tileLength);
        AscendC::Cast(zLocal, tmpTensor0, AscendC::RoundMode::CAST_RINT, tileLength);

        outQueueZ.EnQue<bfloat16_t>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(uint32_t progress, uint32_t tileLength)
    {
        AscendC::LocalTensor<bfloat16_t> zLocal = outQueueZ.DeQue<bfloat16_t>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf0;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf1;

    AscendC::GlobalTensor<bfloat16_t> xGm;
    AscendC::GlobalTensor<bfloat16_t> yGm;
    AscendC::GlobalTensor<bfloat16_t> zGm;

    uint32_t initBufferLength;     // initBuffer所用长度
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t lastTileLength;
};

template <> class KernelAdd <int8_t> {
public:
    __aicore__ inline KernelAdd() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, AddCustomTilingData tiling)
    {
        if (tiling.isEvenCore) {
            this->blockLength = tiling.blockLength;
            this->tileNum = tiling.tileNum;
            this->tileLength = tiling.tileLength / BUFFER_NUM;
            this->lastTileLength = tiling.lastTileLength;

            uint64_t offset = this->blockLength * AscendC::GetBlockIdx();
            xGm.SetGlobalBuffer((__gm__ int8_t *)x + offset, this->blockLength);
            yGm.SetGlobalBuffer((__gm__ int8_t *)y + offset, this->blockLength);
            zGm.SetGlobalBuffer((__gm__ int8_t *)z + offset, this->blockLength);
        } else {
            if (AscendC::GetBlockIdx() < tiling.formerNum) {
                this->tileNum = tiling.formerTileNum;
                this->tileLength = tiling.formerTileLength / BUFFER_NUM;
                this->lastTileLength = tiling.formerLastTileLength;

                uint64_t offset = tiling.formerLength * AscendC::GetBlockIdx();
                xGm.SetGlobalBuffer((__gm__ int8_t *)x + offset, tiling.formerLength);
                yGm.SetGlobalBuffer((__gm__ int8_t *)y + offset, tiling.formerLength);
                zGm.SetGlobalBuffer((__gm__ int8_t *)z + offset, tiling.formerLength);
            } else {
                this->tileNum = tiling.tailTileNum;
                this->tileLength = tiling.tailTileLength / BUFFER_NUM;
                this->lastTileLength = tiling.tailLastTileLength;

                uint64_t offset = tiling.formerLength * tiling.formerNum + tiling.tailLength *
                    (AscendC::GetBlockIdx() - tiling.formerNum);
                xGm.SetGlobalBuffer((__gm__ int8_t *)x + offset, tiling.tailLength);
                yGm.SetGlobalBuffer((__gm__ int8_t *)y + offset, tiling.tailLength);
                zGm.SetGlobalBuffer((__gm__ int8_t *)z + offset, tiling.tailLength);
            }
        }

        this->initBufferLength = AscendC::Std::max(this->tileLength, this->lastTileLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->initBufferLength * sizeof(int8_t));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->initBufferLength * sizeof(int8_t));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->initBufferLength * sizeof(int8_t));

        pipe.InitBuffer(tmpBuf0, this->initBufferLength * sizeof(half));
        pipe.InitBuffer(tmpBuf1, this->initBufferLength * sizeof(half));
    }
    __aicore__ inline void Process()
    {
        // 整块进行double buffer计算
        uint32_t loopCount = this->tileNum * BUFFER_NUM;
        for (uint32_t i = 0; i < loopCount; i++) {
            CopyIn(i, this->tileLength);
            Compute(i, this->tileLength);
            CopyOut(i, this->tileLength);
        }

        // 进行尾块计算, 不做double buffer操作
        if (this->lastTileLength > 0U) {
            CopyIn(loopCount, this->lastTileLength);
            Compute(loopCount, this->lastTileLength);
            CopyOut(loopCount, this->lastTileLength);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress, uint32_t tileLength)
    {
        AscendC::LocalTensor<int8_t> xLocal = inQueueX.AllocTensor<int8_t>();
        AscendC::LocalTensor<int8_t> yLocal = inQueueY.AllocTensor<int8_t>();

        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], tileLength);
        AscendC::DataCopy(yLocal, yGm[progress * this->tileLength], tileLength);

        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute(uint32_t progress, uint32_t tileLength)
    {
        AscendC::LocalTensor<int8_t> xLocal = inQueueX.DeQue<int8_t>();
        AscendC::LocalTensor<int8_t> yLocal = inQueueY.DeQue<int8_t>();
        AscendC::LocalTensor<int8_t> zLocal = outQueueZ.AllocTensor<int8_t>();

        AscendC::LocalTensor<half> tmpTensor0 = tmpBuf0.Get<half>();
        AscendC::LocalTensor<half> tmpTensor1 = tmpBuf1.Get<half>();

        AscendC::Cast(tmpTensor0, xLocal, AscendC::RoundMode::CAST_NONE, tileLength);
        AscendC::Cast(tmpTensor1, yLocal, AscendC::RoundMode::CAST_NONE, tileLength);

        AscendC::Add(tmpTensor0, tmpTensor0, tmpTensor1, tileLength);
        AscendC::Cast(zLocal, tmpTensor0, AscendC::RoundMode::CAST_NONE, tileLength);

        outQueueZ.EnQue<int8_t>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(uint32_t progress, uint32_t tileLength)
    {
        AscendC::LocalTensor<int8_t> zLocal = outQueueZ.DeQue<int8_t>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf0;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf1;

    AscendC::GlobalTensor<int8_t> xGm;
    AscendC::GlobalTensor<int8_t> yGm;
    AscendC::GlobalTensor<int8_t> zGm;

    uint32_t initBufferLength;     // initBuffer所用长度
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t lastTileLength;
};

template <typename dataType> class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, AddCustomTilingData tiling)
    {
        if (tiling.isEvenCore) {
            this->blockLength = tiling.blockLength;
            this->tileNum = tiling.tileNum;
            this->tileLength = tiling.tileLength / BUFFER_NUM;
            this->lastTileLength = tiling.lastTileLength;

            uint64_t offset = this->blockLength * AscendC::GetBlockIdx();
            xGm.SetGlobalBuffer((__gm__ dataType *)x + offset, this->blockLength);
            yGm.SetGlobalBuffer((__gm__ dataType *)y + offset, this->blockLength);
            zGm.SetGlobalBuffer((__gm__ dataType *)z + offset, this->blockLength);
        } else {
            if (AscendC::GetBlockIdx() < tiling.formerNum) {
                this->tileNum = tiling.formerTileNum;
                this->tileLength = tiling.formerTileLength / BUFFER_NUM;
                this->lastTileLength = tiling.formerLastTileLength;

                uint64_t offset = tiling.formerLength * AscendC::GetBlockIdx();
                xGm.SetGlobalBuffer((__gm__ dataType *)x + offset, tiling.formerLength);
                yGm.SetGlobalBuffer((__gm__ dataType *)y + offset, tiling.formerLength);
                zGm.SetGlobalBuffer((__gm__ dataType *)z + offset, tiling.formerLength);
            } else {
                this->tileNum = tiling.tailTileNum;
                this->tileLength = tiling.tailTileLength / BUFFER_NUM;
                this->lastTileLength = tiling.tailLastTileLength;

                uint64_t offset = tiling.formerLength * tiling.formerNum + tiling.tailLength *
                    (AscendC::GetBlockIdx() - tiling.formerNum);
                xGm.SetGlobalBuffer((__gm__ dataType *)x + offset, tiling.tailLength);
                yGm.SetGlobalBuffer((__gm__ dataType *)y + offset, tiling.tailLength);
                zGm.SetGlobalBuffer((__gm__ dataType *)z + offset, tiling.tailLength);
            }
        }
        this->initBufferLength = AscendC::Std::max(this->tileLength, this->lastTileLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->initBufferLength * sizeof(dataType));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->initBufferLength * sizeof(dataType));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->initBufferLength * sizeof(dataType));
    }
    __aicore__ inline void Process()
    {
        // 整块进行double buffer计算
        uint32_t loopCount = this->tileNum * BUFFER_NUM;
        for (uint32_t i = 0; i < loopCount; i++) {
            CopyIn(i, this->tileLength);
            Compute(i, this->tileLength);
            CopyOut(i, this->tileLength);
        }

        // 进行尾块计算, 不做double buffer操作
        if (this->lastTileLength > 0) {
            CopyIn(loopCount, this->lastTileLength);
            Compute(loopCount, this->lastTileLength);
            CopyOut(loopCount, this->lastTileLength);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress, uint32_t tileLength)
    {
        AscendC::LocalTensor<dataType> xLocal = inQueueX.AllocTensor<dataType>();
        AscendC::LocalTensor<dataType> yLocal = inQueueY.AllocTensor<dataType>();

        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], tileLength);
        AscendC::DataCopy(yLocal, yGm[progress * this->tileLength], tileLength);

        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute(uint32_t progress, uint32_t tileLength)
    {
        AscendC::LocalTensor<dataType> xLocal = inQueueX.DeQue<dataType>();
        AscendC::LocalTensor<dataType> yLocal = inQueueY.DeQue<dataType>();
        AscendC::LocalTensor<dataType> zLocal = outQueueZ.AllocTensor<dataType>();

        AscendC::Add(zLocal, xLocal, yLocal, tileLength);

        outQueueZ.EnQue<dataType>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(uint32_t progress, uint32_t tileLength)
    {
        AscendC::LocalTensor<dataType> zLocal = outQueueZ.DeQue<dataType>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;

    AscendC::GlobalTensor<dataType> xGm;
    AscendC::GlobalTensor<dataType> yGm;
    AscendC::GlobalTensor<dataType> zGm;

    uint32_t initBufferLength;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t lastTileLength;
};

extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, AddCustomTilingData tiling)
{
    if (tiling.dataType == ADD_BFLOAT16) {
        KernelAdd<bfloat16_t> op;
        op.Init(x, y, z, tiling);
        op.Process();
    } else if (tiling.dataType == ADD_FLOAT16) {
        KernelAdd<half> op;
        op.Init(x, y, z, tiling);
        op.Process();
    } else if (tiling.dataType == ADD_FLOAT32) {
        KernelAdd<float> op;
        op.Init(x, y, z, tiling);
        op.Process();
    } else if (tiling.dataType == ADD_INT8) {
        KernelAdd<int8_t> op;
        op.Init(x, y, z, tiling);
        op.Process();
    } else if (tiling.dataType == ADD_INT16) {
        KernelAdd<int16_t> op;
        op.Init(x, y, z, tiling);
        op.Process();
    } else if (tiling.dataType == ADD_INT32) {
        KernelAdd<int32_t> op;
        op.Init(x, y, z, tiling);
        op.Process();
    } else {
        return;
    }
}
