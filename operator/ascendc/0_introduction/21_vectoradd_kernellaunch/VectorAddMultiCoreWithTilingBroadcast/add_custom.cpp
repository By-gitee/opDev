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

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

constexpr uint32_t ADD_BFLOAT16 = 0;
constexpr uint32_t ADD_FLOAT16 = 1;
constexpr uint32_t ADD_FLOAT32 = 2;
constexpr uint32_t ADD_INT8 = 3;
constexpr uint32_t ADD_INT16 = 4;
constexpr uint32_t ADD_INT32 = 5;

constexpr uint32_t BROADCAST_DIM = 2;
constexpr uint32_t BROADCAST_AXIS_ZERO = 0;
constexpr uint32_t BROADCAST_AXIS_ONE = 1;
constexpr uint32_t LAST_TWO_TILE = 2;
template <typename dataType, uint32_t axis> class KernelAdd;

// 针对axis = 0的场景
template <> class KernelAdd<bfloat16_t, 0> {
public:
    __aicore__ inline KernelAdd() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, AddCustomTilingData tiling)
    {
        GM_ADDR longerInputPtr;
        GM_ADDR shorterInputPtr;
        if (tiling.xLen > tiling.yLen) {
            longerInputPtr = x;
            shorterInputPtr = y;
            this->shorterAxisLen = tiling.yLen;
        } else {
            longerInputPtr = y;
            shorterInputPtr = x;
            this->shorterAxisLen = tiling.xLen;
        }
        this->coef = tiling.coef;
        if (tiling.isEvenCore) {
            this->tileNum = tiling.tileNum;
            this->tileLength = tiling.tileLength / BUFFER_NUM;
            this->lastTileLength = tiling.lastTileLength;

            xGm.SetGlobalBuffer((__gm__ bfloat16_t *)longerInputPtr + tiling.blockLength * AscendC::GetBlockIdx(), tiling.blockLength);
            yGm.SetGlobalBuffer((__gm__ bfloat16_t *)shorterInputPtr, this->shorterAxisLen);
            zGm.SetGlobalBuffer((__gm__ bfloat16_t *)z + tiling.blockLength * AscendC::GetBlockIdx(), tiling.blockLength);
        } else {
            if (AscendC::GetBlockIdx() < tiling.formerNum) {
                this->tileNum = tiling.formerTileNum;
                this->tileLength = tiling.formerTileLength / BUFFER_NUM;
                this->lastTileLength = tiling.formerLastTileLength;

                xGm.SetGlobalBuffer((__gm__ bfloat16_t *)longerInputPtr + tiling.formerLength * AscendC::GetBlockIdx(), tiling.formerLength);
                yGm.SetGlobalBuffer((__gm__ bfloat16_t *)shorterInputPtr, this->shorterAxisLen);
                zGm.SetGlobalBuffer((__gm__ bfloat16_t *)z + tiling.formerLength * AscendC::GetBlockIdx(), tiling.formerLength);
            } else {
                this->tileNum = tiling.tailTileNum;
                this->tileLength = tiling.tailTileLength / BUFFER_NUM;
                this->lastTileLength = tiling.tailLastTileLength;

                xGm.SetGlobalBuffer((__gm__ bfloat16_t *)longerInputPtr + tiling.formerLength * tiling.formerNum +
                    tiling.tailLength * (AscendC::GetBlockIdx() - tiling.formerNum), tiling.tailLength);
                yGm.SetGlobalBuffer((__gm__ bfloat16_t *)shorterInputPtr, this->shorterAxisLen);
                zGm.SetGlobalBuffer((__gm__ bfloat16_t *)z + tiling.formerLength * tiling.formerNum +
                    tiling.tailLength * (AscendC::GetBlockIdx() - tiling.formerNum), tiling.tailLength);
            }
        }
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(bfloat16_t));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(bfloat16_t));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(bfloat16_t));

        pipe.InitBuffer(tmpBuf0, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmpBuf1, this->tileLength * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<bfloat16_t> xLocal = inQueueX.AllocTensor<bfloat16_t>();
        AscendC::LocalTensor<bfloat16_t> yLocal = inQueueY.AllocTensor<bfloat16_t>();

        AscendC::DataCopyExtParams copyParams = {1, (uint32_t)(this->tileLength * sizeof(bfloat16_t)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<bfloat16_t> padParams = {false, 0, 0, 0};

        AscendC::DataCopyPad<bfloat16_t>(xLocal, xGm[progress * this->tileLength], copyParams, padParams);
        AscendC::DataCopyPad<bfloat16_t>(yLocal, yGm[(progress % BUFFER_NUM) * this->tileLength], copyParams, padParams);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<bfloat16_t> xLocal = inQueueX.DeQue<bfloat16_t>();
        AscendC::LocalTensor<bfloat16_t> yLocal = inQueueY.DeQue<bfloat16_t>();
        AscendC::LocalTensor<bfloat16_t> zLocal = outQueueZ.AllocTensor<bfloat16_t>();

        AscendC::LocalTensor<float> tmpTensor0 = tmpBuf0.Get<float>();
        AscendC::LocalTensor<float> tmpTensor1 = tmpBuf1.Get<float>();

        AscendC::Cast(tmpTensor0, xLocal, AscendC::RoundMode::CAST_NONE, this->tileLength);
        AscendC::Cast(tmpTensor1, yLocal, AscendC::RoundMode::CAST_NONE, this->tileLength);

        AscendC::Add(tmpTensor0, tmpTensor0, tmpTensor1, this->tileLength);
        AscendC::Cast(zLocal, tmpTensor0, AscendC::RoundMode::CAST_RINT, this->tileLength);


        outQueueZ.EnQue<bfloat16_t>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<bfloat16_t> zLocal = outQueueZ.DeQue<bfloat16_t>();
        AscendC::DataCopyExtParams copyParams = {1, (uint32_t)(this->tileLength * sizeof(bfloat16_t)), 0, 0, 0};

        AscendC::DataCopyPad<bfloat16_t>(zGm[progress * this->tileLength], zLocal, copyParams);
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

    uint32_t coef;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t lastTileLength;
    uint32_t shorterAxisLen;
};

template <> class KernelAdd<int8_t, 0> {
public:
    __aicore__ inline KernelAdd() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, AddCustomTilingData tiling)
    {
        GM_ADDR longerInputPtr;
        GM_ADDR shorterInputPtr;
        if (tiling.xLen > tiling.yLen) {
            longerInputPtr = x;
            shorterInputPtr = y;
            this->shorterAxisLen = tiling.yLen;
        } else {
            longerInputPtr = y;
            shorterInputPtr = x;
            this->shorterAxisLen = tiling.xLen;
        }
        this->coef = tiling.coef;
        if (tiling.isEvenCore) {
            this->tileNum = tiling.tileNum;
            this->tileLength = tiling.tileLength / BUFFER_NUM;
            this->lastTileLength = tiling.lastTileLength;

            xGm.SetGlobalBuffer((__gm__ int8_t *)longerInputPtr + tiling.blockLength * AscendC::GetBlockIdx(), tiling.blockLength);
            yGm.SetGlobalBuffer((__gm__ int8_t *)shorterInputPtr, this->shorterAxisLen);
            zGm.SetGlobalBuffer((__gm__ int8_t *)z + tiling.blockLength * AscendC::GetBlockIdx(), tiling.blockLength);
        } else {
            if (AscendC::GetBlockIdx() < tiling.formerNum) {
                this->tileNum = tiling.formerTileNum;
                this->tileLength = tiling.formerTileLength / BUFFER_NUM;
                this->lastTileLength = tiling.formerLastTileLength;

                xGm.SetGlobalBuffer((__gm__ int8_t *)longerInputPtr + tiling.formerLength * AscendC::GetBlockIdx(), tiling.formerLength);
                yGm.SetGlobalBuffer((__gm__ int8_t *)shorterInputPtr, this->shorterAxisLen);
                zGm.SetGlobalBuffer((__gm__ int8_t *)z + tiling.formerLength * AscendC::GetBlockIdx(), tiling.formerLength);
            } else {
                this->tileNum = tiling.tailTileNum;
                this->tileLength = tiling.tailTileLength / BUFFER_NUM;
                this->lastTileLength = tiling.tailLastTileLength;

                xGm.SetGlobalBuffer((__gm__ int8_t *)longerInputPtr + tiling.formerLength * tiling.formerNum +
                    tiling.tailLength * (AscendC::GetBlockIdx() - tiling.formerNum), tiling.tailLength);
                yGm.SetGlobalBuffer((__gm__ int8_t *)shorterInputPtr, this->shorterAxisLen);
                zGm.SetGlobalBuffer((__gm__ int8_t *)z + tiling.formerLength * tiling.formerNum +
                    tiling.tailLength * (AscendC::GetBlockIdx() - tiling.formerNum), tiling.tailLength);
            }
        }
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(int8_t));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(int8_t));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(int8_t));

        pipe.InitBuffer(tmpBuf0, this->tileLength * sizeof(half));
        pipe.InitBuffer(tmpBuf1, this->tileLength * sizeof(half));
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<int8_t> xLocal = inQueueX.AllocTensor<int8_t>();
        AscendC::LocalTensor<int8_t> yLocal = inQueueY.AllocTensor<int8_t>();

        AscendC::DataCopyExtParams copyParams = {1, (uint32_t)(this->tileLength * sizeof(int8_t)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<int8_t> padParams = {false, 0, 0, 0};

        AscendC::DataCopyPad<int8_t>(xLocal, xGm[progress * this->tileLength], copyParams, padParams);
        AscendC::DataCopyPad<int8_t>(yLocal, yGm[(progress % BUFFER_NUM) * this->tileLength], copyParams, padParams);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<int8_t> xLocal = inQueueX.DeQue<int8_t>();
        AscendC::LocalTensor<int8_t> yLocal = inQueueY.DeQue<int8_t>();
        AscendC::LocalTensor<int8_t> zLocal = outQueueZ.AllocTensor<int8_t>();

        AscendC::LocalTensor<half> tmpTensor0 = tmpBuf0.Get<half>();
        AscendC::LocalTensor<half> tmpTensor1 = tmpBuf1.Get<half>();

        AscendC::Cast(tmpTensor0, xLocal, AscendC::RoundMode::CAST_NONE, this->tileLength);
        AscendC::Cast(tmpTensor1, yLocal, AscendC::RoundMode::CAST_NONE, this->tileLength);

        AscendC::Add(tmpTensor0, tmpTensor0, tmpTensor1, this->tileLength);
        AscendC::Cast(zLocal, tmpTensor0, AscendC::RoundMode::CAST_NONE, this->tileLength);


        outQueueZ.EnQue<int8_t>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<int8_t> zLocal = outQueueZ.DeQue<int8_t>();
        AscendC::DataCopyExtParams copyParams = {1, (uint32_t)(this->tileLength * sizeof(int8_t)), 0, 0, 0};

        AscendC::DataCopyPad<int8_t>(zGm[progress * this->tileLength], zLocal, copyParams);
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

    uint32_t coef;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t lastTileLength;
    uint32_t shorterAxisLen;
};

template <typename dataType> class KernelAdd<dataType, 0> {
public:
    __aicore__ inline KernelAdd() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, AddCustomTilingData tiling)
    {
        GM_ADDR longerInputPtr;
        GM_ADDR shorterInputPtr;
        if (tiling.xLen > tiling.yLen) {
            longerInputPtr = x;
            shorterInputPtr = y;
            this->shorterAxisLen = tiling.yLen;
        } else {
            longerInputPtr = y;
            shorterInputPtr = x;
            this->shorterAxisLen = tiling.xLen;
        }
        this->coef = tiling.coef;
        if (tiling.isEvenCore) {
            this->tileNum = tiling.tileNum;
            this->tileLength = tiling.tileLength / BUFFER_NUM;
            this->lastTileLength = tiling.lastTileLength;

            xGm.SetGlobalBuffer((__gm__ dataType *)longerInputPtr + tiling.blockLength * AscendC::GetBlockIdx(), tiling.blockLength);
            yGm.SetGlobalBuffer((__gm__ dataType *)shorterInputPtr, this->shorterAxisLen);
            zGm.SetGlobalBuffer((__gm__ dataType *)z + tiling.blockLength * AscendC::GetBlockIdx(), tiling.blockLength);
        } else {
            if (AscendC::GetBlockIdx() < tiling.formerNum) {
                this->tileNum = tiling.formerTileNum;
                this->tileLength = tiling.formerTileLength / BUFFER_NUM;
                this->lastTileLength = tiling.formerLastTileLength;

                xGm.SetGlobalBuffer((__gm__ dataType *)longerInputPtr + tiling.formerLength * AscendC::GetBlockIdx(), tiling.formerLength);
                yGm.SetGlobalBuffer((__gm__ dataType *)shorterInputPtr, this->shorterAxisLen);
                zGm.SetGlobalBuffer((__gm__ dataType *)z + tiling.formerLength * AscendC::GetBlockIdx(), tiling.formerLength);
            } else {
                this->tileNum = tiling.tailTileNum;
                this->tileLength = tiling.tailTileLength / BUFFER_NUM;
                this->lastTileLength = tiling.tailLastTileLength;

                xGm.SetGlobalBuffer((__gm__ dataType *)longerInputPtr + tiling.formerLength * tiling.formerNum +
                    tiling.tailLength * (AscendC::GetBlockIdx() - tiling.formerNum), tiling.tailLength);
                yGm.SetGlobalBuffer((__gm__ dataType *)shorterInputPtr, this->shorterAxisLen);
                zGm.SetGlobalBuffer((__gm__ dataType *)z + tiling.formerLength * tiling.formerNum +
                    tiling.tailLength * (AscendC::GetBlockIdx() - tiling.formerNum), tiling.tailLength);
            }
        }
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(dataType));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(dataType));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(dataType));
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<dataType> xLocal = inQueueX.AllocTensor<dataType>();
        AscendC::LocalTensor<dataType> yLocal = inQueueY.AllocTensor<dataType>();

        AscendC::DataCopyExtParams copyParams = {1, (uint32_t)(this->tileLength * sizeof(dataType)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<dataType> padParams = {false, 0, 0, 0};

        AscendC::DataCopyPad<dataType>(xLocal, xGm[progress * this->tileLength], copyParams, padParams);
        AscendC::DataCopyPad<dataType>(yLocal, yGm[(progress % BUFFER_NUM) * this->tileLength], copyParams, padParams);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<dataType> xLocal = inQueueX.DeQue<dataType>();
        AscendC::LocalTensor<dataType> yLocal = inQueueY.DeQue<dataType>();
        AscendC::LocalTensor<dataType> zLocal = outQueueZ.AllocTensor<dataType>();

        AscendC::Add(zLocal, xLocal, yLocal, this->tileLength);

        outQueueZ.EnQue<dataType>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<dataType> zLocal = outQueueZ.DeQue<dataType>();
        AscendC::DataCopyExtParams copyParams = {1, (uint32_t)(this->tileLength * sizeof(dataType)), 0, 0, 0};

        AscendC::DataCopyPad<dataType>(zGm[progress * this->tileLength], zLocal, copyParams);
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

    uint32_t coef;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t lastTileLength;
    uint32_t shorterAxisLen;
};

// 针对axis = 1的场景
template <> class KernelAdd<bfloat16_t, 1> {
public:
    __aicore__ inline KernelAdd() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, AddCustomTilingData tiling)
    {
        GM_ADDR longerInputPtr;
        GM_ADDR shorterInputPtr;
        if (tiling.xLen > tiling.yLen) {
            longerInputPtr = x;
            shorterInputPtr = y;
        } else {
            longerInputPtr = y;
            shorterInputPtr = x;
        }
        this->coef = tiling.coef;
        if (tiling.isEvenCore) {
            this->tileNum = tiling.tileNum;
            this->tileLength = tiling.tileLength / BUFFER_NUM;
            this->lastTileLength = tiling.lastTileLength;

            xGm.SetGlobalBuffer((__gm__ bfloat16_t *)longerInputPtr + tiling.blockLength * AscendC::GetBlockIdx(), tiling.blockLength);
            yGm.SetGlobalBuffer((__gm__ bfloat16_t *)shorterInputPtr + tiling.blockLength * AscendC::GetBlockIdx() / this->coef, tiling.blockLength / this->coef);
            zGm.SetGlobalBuffer((__gm__ bfloat16_t *)z + tiling.blockLength * AscendC::GetBlockIdx(), tiling.blockLength);
        } else {
            if (AscendC::GetBlockIdx() < tiling.formerNum) {
                this->tileNum = tiling.formerTileNum;
                this->tileLength = tiling.formerTileLength / BUFFER_NUM;
                this->lastTileLength = tiling.formerLastTileLength;

                xGm.SetGlobalBuffer((__gm__ bfloat16_t *)longerInputPtr + tiling.formerLength * AscendC::GetBlockIdx(), tiling.formerLength);
                yGm.SetGlobalBuffer((__gm__ bfloat16_t *)shorterInputPtr + tiling.formerLength * AscendC::GetBlockIdx() / this->coef, tiling.formerLength / this->coef);
                zGm.SetGlobalBuffer((__gm__ bfloat16_t *)z + tiling.formerLength * AscendC::GetBlockIdx(), tiling.formerLength);
            } else {
                this->tileNum = tiling.tailTileNum;
                this->tileLength = tiling.tailTileLength / BUFFER_NUM;
                this->lastTileLength = tiling.tailLastTileLength;

                xGm.SetGlobalBuffer((__gm__ bfloat16_t *)longerInputPtr + tiling.formerLength * tiling.formerNum +
                    tiling.tailLength * (AscendC::GetBlockIdx() - tiling.formerNum), tiling.tailLength);
                yGm.SetGlobalBuffer((__gm__ bfloat16_t *)shorterInputPtr + tiling.formerLength * tiling.formerNum / this->coef +
                tiling.tailLength * (AscendC::GetBlockIdx() - tiling.formerNum) / this->coef, tiling.tailLength / this->coef);
                zGm.SetGlobalBuffer((__gm__ bfloat16_t *)z + tiling.formerLength * tiling.formerNum +
                    tiling.tailLength * (AscendC::GetBlockIdx() - tiling.formerNum), tiling.tailLength);
            }
        }
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(bfloat16_t));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->coef * sizeof(bfloat16_t));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(bfloat16_t));

        pipe.InitBuffer(tmpBuf2, this->tileLength * sizeof(bfloat16_t));
        pipe.InitBuffer(tmpBuf0, this->tileLength * sizeof(float));
        pipe.InitBuffer(tmpBuf1, this->tileLength * sizeof(float));
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<bfloat16_t> xLocal = inQueueX.AllocTensor<bfloat16_t>();
        AscendC::LocalTensor<bfloat16_t> yLocal = inQueueY.AllocTensor<bfloat16_t>();

        AscendC::DataCopyExtParams copyXParams = {1, (uint32_t)(this->tileLength * sizeof(bfloat16_t)), 0, 0, 0};
        AscendC::DataCopyExtParams copyYParams = {1, (uint32_t)(this->tileLength * sizeof(bfloat16_t) / this->coef), 0, 0, 0};
        AscendC::DataCopyPadExtParams<bfloat16_t> padParams = {false, 0, 0, 0};

        if (progress == (this->tileNum * BUFFER_NUM - 1)) {
            AscendC::DataCopyPad<bfloat16_t>(xLocal, xGm[(progress - LAST_TWO_TILE) * this->tileLength + this->lastTileLength],
                copyXParams, padParams);
            AscendC::DataCopyPad<bfloat16_t>(yLocal, yGm[((progress - LAST_TWO_TILE) * this->tileLength + this->lastTileLength) / this->coef],
                copyYParams, padParams);
        } else {
            AscendC::DataCopyPad<bfloat16_t>(xLocal, xGm[progress * this->tileLength], copyXParams, padParams);
            AscendC::DataCopyPad<bfloat16_t>(yLocal, yGm[progress * this->tileLength / this->coef], copyYParams, padParams);
        }
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<bfloat16_t> xLocal = inQueueX.DeQue<bfloat16_t>();
        AscendC::LocalTensor<bfloat16_t> yLocal = inQueueY.DeQue<bfloat16_t>();
        AscendC::LocalTensor<bfloat16_t> zLocal = outQueueZ.AllocTensor<bfloat16_t>();

        AscendC::LocalTensor<bfloat16_t> broadcastTmpTensor = tmpBuf2.Get<bfloat16_t>();
        uint32_t dstShape[] = {this->tileLength / this->coef, this->coef};
        uint32_t srcShape[] = {this->tileLength / this->coef, 1};
        AscendC::Broadcast<bfloat16_t, BROADCAST_DIM, BROADCAST_AXIS_ONE>(broadcastTmpTensor, yLocal, dstShape, srcShape);

        AscendC::LocalTensor<float> tmpTensor0 = tmpBuf0.Get<float>();
        AscendC::LocalTensor<float> tmpTensor1 = tmpBuf1.Get<float>();

        AscendC::Cast(tmpTensor0, xLocal, AscendC::RoundMode::CAST_NONE, this->tileLength);
        AscendC::Cast(tmpTensor1, broadcastTmpTensor, AscendC::RoundMode::CAST_NONE, this->tileLength);

        AscendC::Add(tmpTensor0, tmpTensor0, tmpTensor1, this->tileLength);
        AscendC::Cast(zLocal, tmpTensor0, AscendC::RoundMode::CAST_RINT, this->tileLength);

        outQueueZ.EnQue<bfloat16_t>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<bfloat16_t> zLocal = outQueueZ.DeQue<bfloat16_t>();
        AscendC::DataCopyExtParams copyParams = {1, (uint32_t)(this->tileLength * sizeof(bfloat16_t)), 0, 0, 0};
        if (progress == (this->tileNum * BUFFER_NUM - 1)) {
            AscendC::DataCopyPad<bfloat16_t>(zGm[(progress - LAST_TWO_TILE) * this->tileLength + this->lastTileLength], zLocal, copyParams);
        } else {
            AscendC::DataCopyPad<bfloat16_t>(zGm[progress * this->tileLength], zLocal, copyParams);
        }
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;

    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf0;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf1;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf2;

    AscendC::GlobalTensor<bfloat16_t> xGm;
    AscendC::GlobalTensor<bfloat16_t> yGm;
    AscendC::GlobalTensor<bfloat16_t> zGm;

    uint32_t coef;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t lastTileLength;
};

template <> class KernelAdd<int8_t, 1> {
public:
    __aicore__ inline KernelAdd() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, AddCustomTilingData tiling)
    {
        GM_ADDR longerInputPtr;
        GM_ADDR shorterInputPtr;
        if (tiling.xLen > tiling.yLen) {
            longerInputPtr = x;
            shorterInputPtr = y;
        } else {
            longerInputPtr = y;
            shorterInputPtr = x;
        }
        this->coef = tiling.coef;
        if (tiling.isEvenCore) {
            this->tileNum = tiling.tileNum;
            this->tileLength = tiling.tileLength / BUFFER_NUM;
            this->lastTileLength = tiling.lastTileLength;

            xGm.SetGlobalBuffer((__gm__ int8_t *)longerInputPtr + tiling.blockLength * AscendC::GetBlockIdx(), tiling.blockLength);
            yGm.SetGlobalBuffer((__gm__ int8_t *)shorterInputPtr + tiling.blockLength * AscendC::GetBlockIdx() / this->coef, tiling.blockLength / this->coef);
            zGm.SetGlobalBuffer((__gm__ int8_t *)z + tiling.blockLength * AscendC::GetBlockIdx(), tiling.blockLength);
        } else {
            if (AscendC::GetBlockIdx() < tiling.formerNum) {
                this->tileNum = tiling.formerTileNum;
                this->tileLength = tiling.formerTileLength / BUFFER_NUM;
                this->lastTileLength = tiling.formerLastTileLength;

                xGm.SetGlobalBuffer((__gm__ int8_t *)longerInputPtr + tiling.formerLength * AscendC::GetBlockIdx(), tiling.formerLength);
                yGm.SetGlobalBuffer((__gm__ int8_t *)shorterInputPtr + tiling.formerLength * AscendC::GetBlockIdx() / this->coef, tiling.formerLength / this->coef);
                zGm.SetGlobalBuffer((__gm__ int8_t *)z + tiling.formerLength * AscendC::GetBlockIdx(), tiling.formerLength);
            } else {
                this->tileNum = tiling.tailTileNum;
                this->tileLength = tiling.tailTileLength / BUFFER_NUM;
                this->lastTileLength = tiling.tailLastTileLength;

                xGm.SetGlobalBuffer((__gm__ int8_t *)longerInputPtr + tiling.formerLength * tiling.formerNum +
                    tiling.tailLength * (AscendC::GetBlockIdx() - tiling.formerNum), tiling.tailLength);
                yGm.SetGlobalBuffer((__gm__ int8_t *)shorterInputPtr + tiling.formerLength * tiling.formerNum / this->coef +
                tiling.tailLength * (AscendC::GetBlockIdx() - tiling.formerNum) / this->coef, tiling.tailLength / this->coef);
                zGm.SetGlobalBuffer((__gm__ int8_t *)z + tiling.formerLength * tiling.formerNum +
                    tiling.tailLength * (AscendC::GetBlockIdx() - tiling.formerNum), tiling.tailLength);
            }
        }
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(int8_t));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->coef * sizeof(int8_t));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(int8_t));

        pipe.InitBuffer(tmpBuf2, this->tileLength * sizeof(int8_t));

        pipe.InitBuffer(tmpBuf0, this->tileLength * sizeof(half));
        pipe.InitBuffer(tmpBuf1, this->tileLength * sizeof(half));
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<int8_t> xLocal = inQueueX.AllocTensor<int8_t>();
        AscendC::LocalTensor<int8_t> yLocal = inQueueY.AllocTensor<int8_t>();

        AscendC::DataCopyExtParams copyXParams = {1, (uint32_t)(this->tileLength * sizeof(int8_t)), 0, 0, 0};
        AscendC::DataCopyExtParams copyYParams = {1, (uint32_t)(this->tileLength * sizeof(int8_t) / this->coef), 0, 0, 0};
        AscendC::DataCopyPadExtParams<int8_t> padParams = {false, 0, 0, 0};

        if (progress == (this->tileNum * BUFFER_NUM - 1)) {
            AscendC::DataCopyPad<int8_t>(xLocal, xGm[(progress - LAST_TWO_TILE) * this->tileLength + this->lastTileLength],
                copyXParams, padParams);
            AscendC::DataCopyPad<int8_t>(yLocal, yGm[((progress - LAST_TWO_TILE) * this->tileLength + this->lastTileLength) / this->coef],
                copyYParams, padParams);
        } else {
            AscendC::DataCopyPad<int8_t>(xLocal, xGm[progress * this->tileLength], copyXParams, padParams);
            AscendC::DataCopyPad<int8_t>(yLocal, yGm[progress * this->tileLength / this->coef], copyYParams, padParams);
        }
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<int8_t> xLocal = inQueueX.DeQue<int8_t>();
        AscendC::LocalTensor<int8_t> yLocal = inQueueY.DeQue<int8_t>();
        AscendC::LocalTensor<int8_t> zLocal = outQueueZ.AllocTensor<int8_t>();

        AscendC::LocalTensor<int8_t> broadcastTmpTensor = tmpBuf2.Get<int8_t>();
        uint32_t dstShape[] = {this->tileLength / this->coef, this->coef};
        uint32_t srcShape[] = {this->tileLength / this->coef, 1};
        AscendC::Broadcast<int8_t, BROADCAST_DIM, BROADCAST_AXIS_ONE>(broadcastTmpTensor, yLocal, dstShape, srcShape);

        AscendC::LocalTensor<half> tmpTensor0 = tmpBuf0.Get<half>();
        AscendC::LocalTensor<half> tmpTensor1 = tmpBuf1.Get<half>();

        AscendC::Cast(tmpTensor0, xLocal, AscendC::RoundMode::CAST_NONE, this->tileLength);
        AscendC::Cast(tmpTensor1, broadcastTmpTensor, AscendC::RoundMode::CAST_NONE, this->tileLength);

        AscendC::Add(tmpTensor0, tmpTensor0, tmpTensor1, this->tileLength);
        AscendC::Cast(zLocal, tmpTensor0, AscendC::RoundMode::CAST_NONE, this->tileLength);

        outQueueZ.EnQue<int8_t>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<int8_t> zLocal = outQueueZ.DeQue<int8_t>();
        AscendC::DataCopyExtParams copyParams = {1, (uint32_t)(this->tileLength * sizeof(int8_t)), 0, 0, 0};
        if (progress == (this->tileNum * BUFFER_NUM - 1)) {
            AscendC::DataCopyPad<int8_t>(zGm[(progress - LAST_TWO_TILE) * this->tileLength + this->lastTileLength], zLocal, copyParams);
        } else {
            AscendC::DataCopyPad<int8_t>(zGm[progress * this->tileLength], zLocal, copyParams);
        }
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;

    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf0;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf1;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf2;

    AscendC::GlobalTensor<int8_t> xGm;
    AscendC::GlobalTensor<int8_t> yGm;
    AscendC::GlobalTensor<int8_t> zGm;

    uint32_t coef;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t lastTileLength;
};

template <typename dataType> class KernelAdd<dataType, 1> {
public:
    __aicore__ inline KernelAdd() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, AddCustomTilingData tiling)
    {
        GM_ADDR longerInputPtr;
        GM_ADDR shorterInputPtr;
        if (tiling.xLen > tiling.yLen) {
            longerInputPtr = x;
            shorterInputPtr = y;
        } else {
            longerInputPtr = y;
            shorterInputPtr = x;
        }
        this->coef = tiling.coef;
        if (tiling.isEvenCore) {
            this->tileNum = tiling.tileNum;
            this->tileLength = tiling.tileLength / BUFFER_NUM;
            this->lastTileLength = tiling.lastTileLength;

            xGm.SetGlobalBuffer((__gm__ dataType *)longerInputPtr + tiling.blockLength * AscendC::GetBlockIdx(), tiling.blockLength);
            yGm.SetGlobalBuffer((__gm__ dataType *)shorterInputPtr + tiling.blockLength * AscendC::GetBlockIdx() / this->coef, tiling.blockLength / this->coef);
            zGm.SetGlobalBuffer((__gm__ dataType *)z + tiling.blockLength * AscendC::GetBlockIdx(), tiling.blockLength);
        } else {
            if (AscendC::GetBlockIdx() < tiling.formerNum) {
                this->tileNum = tiling.formerTileNum;
                this->tileLength = tiling.formerTileLength / BUFFER_NUM;
                this->lastTileLength = tiling.formerLastTileLength;

                xGm.SetGlobalBuffer((__gm__ dataType *)longerInputPtr + tiling.formerLength * AscendC::GetBlockIdx(), tiling.formerLength);
                yGm.SetGlobalBuffer((__gm__ dataType *)shorterInputPtr + tiling.formerLength * AscendC::GetBlockIdx() / this->coef, tiling.formerLength / this->coef);
                zGm.SetGlobalBuffer((__gm__ dataType *)z + tiling.formerLength * AscendC::GetBlockIdx(), tiling.formerLength);
            } else {
                this->tileNum = tiling.tailTileNum;
                this->tileLength = tiling.tailTileLength / BUFFER_NUM;
                this->lastTileLength = tiling.tailLastTileLength;

                xGm.SetGlobalBuffer((__gm__ dataType *)longerInputPtr + tiling.formerLength * tiling.formerNum +
                    tiling.tailLength * (AscendC::GetBlockIdx() - tiling.formerNum), tiling.tailLength);
                yGm.SetGlobalBuffer((__gm__ dataType *)shorterInputPtr + tiling.formerLength * tiling.formerNum / this->coef +
                tiling.tailLength * (AscendC::GetBlockIdx() - tiling.formerNum) / this->coef, tiling.tailLength / this->coef);
                zGm.SetGlobalBuffer((__gm__ dataType *)z + tiling.formerLength * tiling.formerNum +
                    tiling.tailLength * (AscendC::GetBlockIdx() - tiling.formerNum), tiling.tailLength);
            }
        }
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(dataType));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->coef * sizeof(dataType));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(dataType));

        pipe.InitBuffer(tmpBuf2, this->tileLength * sizeof(dataType));
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<dataType> xLocal = inQueueX.AllocTensor<dataType>();
        AscendC::LocalTensor<dataType> yLocal = inQueueY.AllocTensor<dataType>();

        AscendC::DataCopyExtParams copyXParams = {1, (uint32_t)(this->tileLength * sizeof(dataType)), 0, 0, 0};
        AscendC::DataCopyExtParams copyYParams = {1, (uint32_t)(this->tileLength * sizeof(dataType) / this->coef), 0, 0, 0};
        AscendC::DataCopyPadExtParams<dataType> padParams = {false, 0, 0, 0};

        if (progress == (this->tileNum * BUFFER_NUM - 1)) {
            AscendC::DataCopyPad<dataType>(xLocal, xGm[(progress - LAST_TWO_TILE) * this->tileLength + this->lastTileLength],
                copyXParams, padParams);
            AscendC::DataCopyPad<dataType>(yLocal, yGm[((progress - LAST_TWO_TILE) * this->tileLength + this->lastTileLength) / this->coef],
                copyYParams, padParams);
        } else {
            AscendC::DataCopyPad<dataType>(xLocal, xGm[progress * this->tileLength], copyXParams, padParams);
            AscendC::DataCopyPad<dataType>(yLocal, yGm[progress * this->tileLength / this->coef], copyYParams, padParams);
        }
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<dataType> xLocal = inQueueX.DeQue<dataType>();
        AscendC::LocalTensor<dataType> yLocal = inQueueY.DeQue<dataType>();
        AscendC::LocalTensor<dataType> zLocal = outQueueZ.AllocTensor<dataType>();

        AscendC::LocalTensor<dataType> broadcastTmpTensor = tmpBuf2.Get<dataType>();
        uint32_t dstShape[] = {this->tileLength / this->coef, this->coef};
        uint32_t srcShape[] = {this->tileLength / this->coef, 1};
        AscendC::Broadcast<dataType, BROADCAST_DIM, BROADCAST_AXIS_ONE>(broadcastTmpTensor, yLocal, dstShape, srcShape);

        AscendC::Add(zLocal, xLocal, broadcastTmpTensor, this->tileLength);

        outQueueZ.EnQue<dataType>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<dataType> zLocal = outQueueZ.DeQue<dataType>();
        AscendC::DataCopyExtParams copyParams = {1, (uint32_t)(this->tileLength * sizeof(dataType)), 0, 0, 0};
        if (progress == (this->tileNum * BUFFER_NUM - 1)) {
            AscendC::DataCopyPad<dataType>(zGm[(progress - LAST_TWO_TILE) * this->tileLength + this->lastTileLength], zLocal, copyParams);
        } else {
            AscendC::DataCopyPad<dataType>(zGm[progress * this->tileLength], zLocal, copyParams);
        }
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;

    AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf2;

    AscendC::GlobalTensor<dataType> xGm;
    AscendC::GlobalTensor<dataType> yGm;
    AscendC::GlobalTensor<dataType> zGm;

    uint32_t coef;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t lastTileLength;
};

extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, AddCustomTilingData tiling)
{
    if (tiling.axis == 0) {
        if (tiling.dataType == ADD_BFLOAT16) {
            KernelAdd<bfloat16_t, 0> op;
            op.Init(x, y, z, tiling);
            op.Process();
        } else if (tiling.dataType == ADD_FLOAT16) {
            KernelAdd<half, 0> op;
            op.Init(x, y, z, tiling);
            op.Process();
        } else if (tiling.dataType == ADD_FLOAT32) {
            KernelAdd<float, 0> op;
            op.Init(x, y, z, tiling);
            op.Process();
        } else if (tiling.dataType == ADD_INT8) {
            KernelAdd<int8_t, 0> op;
            op.Init(x, y, z, tiling);
            op.Process();
        } else if (tiling.dataType == ADD_INT16) {
            KernelAdd<int16_t, 0> op;
            op.Init(x, y, z, tiling);
            op.Process();
        } else if (tiling.dataType == ADD_INT32) {
            KernelAdd<int32_t, 0> op;
            op.Init(x, y, z, tiling);
            op.Process();
        } else {
            return;
        }
    } else if (tiling.axis == 1) {
        if (tiling.dataType == ADD_BFLOAT16) {
            KernelAdd<bfloat16_t, 1> op;
            op.Init(x, y, z, tiling);
            op.Process();
        } else if (tiling.dataType == ADD_FLOAT16) {
            KernelAdd<half, 1> op;
            op.Init(x, y, z, tiling);
            op.Process();
        } else if (tiling.dataType == ADD_FLOAT32) {
            KernelAdd<float, 1> op;
            op.Init(x, y, z, tiling);
            op.Process();
        } else if (tiling.dataType == ADD_INT8) {
            KernelAdd<int8_t, 1> op;
            op.Init(x, y, z, tiling);
            op.Process();
        } else if (tiling.dataType == ADD_INT16) {
            KernelAdd<int16_t, 1> op;
            op.Init(x, y, z, tiling);
            op.Process();
        } else if (tiling.dataType == ADD_INT32) {
            KernelAdd<int32_t, 1> op;
            op.Init(x, y, z, tiling);
            op.Process();
        } else {
            return;
        }
    } else {
        return;
    }
}
