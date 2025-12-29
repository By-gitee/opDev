#include "kernel_operator.h"

class KernelLpNormV2
{
public:
  __aicore__ inline KernelLpNormV2() {}
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t blockLength,
                              uint32_t tileNum, uint32_t tileLength,
                              uint32_t lasttileLength, uint32_t typeKey,
                              uint32_t pType, float pValue, uint32_t totalLength,
                              uint32_t stepSize, uint32_t unitCount)
  {
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    this->typeKey = typeKey;
    this->pType = pType;
    this->pValue = pValue;
    if (this->typeKey == 0)
    {
      this->yLength = 8;
    }
    else
    {
      this->yLength = 16;
    }
    this->totalLength = totalLength;
    this->stepSize = stepSize;
    this->unitCount = unitCount;

    this->blockLength = blockLength;
    this->tileNum =
        tileNum ASSERT(tileNum != 0 && "tile num can not be zero!");
    this->tileLength = tileLength;
    this->lasttileLength = lasttileLength;
    uint32_t xLen = this->totalLength * this->stepSize;
    xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x,
                        blockLength > xLen ? blockLength : xLen);
    yGm.SetGlobalBuffer((__gm__ DTYPE_X *)y,
                        blockLength > xLen ? blockLength : xLen);
    yGm_f.SetGlobalBuffer((__gm__ float *)y,
                          blockLength > xLen ? blockLength : xLen);
    if (stepSize == 1 && typeKey == 0 && pType == 0)
    // 此类特殊情况，可以适用double buffer处理进行优化
    {
      this->bufferNum = 2;
      this->tileLength = tileLength / this->bufferNum;
      if (this->tileNum * 2 % 8 == 0)
      {
        this->tileNumAlign = this->tileNum * 2;
      }
      else
      {
        this->tileNumAlign = (this->tileNum * 2 + 7) / 8 * 8;
      }
      if (this->lasttileLength % 16 == 0)
      {
        this->lasttileLengthAlign = this->lasttileLength;
      }
      else
      {
        this->lasttileLengthAlign = (this->lasttileLength + 15) / 16 * 16;
      }
      pipe.InitBuffer(inQueueX_DB, this->bufferNum, this->tileLength * sizeof(float));
      pipe.InitBuffer(inQueueT_DB, this->bufferNum, this->tileLength * sizeof(float));
      pipe.InitBuffer(outQueueY_DB, this->bufferNum, this->yLength * sizeof(float));
      pipe.InitBuffer(calcBuf, this->tileLength * sizeof(float));
    }
    else
    {
      this->bufferNum = 1;
      pipe.InitBuffer(inQueueX, this->bufferNum, this->tileLength * sizeof(DTYPE_X));
      pipe.InitBuffer(inQueueT, this->bufferNum, this->yLength * sizeof(float));
      pipe.InitBuffer(outQueueY, this->bufferNum, this->yLength * sizeof(DTYPE_X));
      pipe.InitBuffer(outQueueY_F, this->bufferNum, this->yLength * sizeof(float));
      pipe.InitBuffer(calcBuf, this->bufferNum * this->tileLength * sizeof(float));
      pipe.InitBuffer(xfBuf, this->bufferNum * this->tileLength * sizeof(float));
      pipe.InitBuffer(calcBuf1, this->bufferNum * this->tileLength * sizeof(uint8_t));
      pipe.InitBuffer(calcBuf2, this->bufferNum * this->tileLength * sizeof(float));
    }
  }

  __aicore__ inline void Process()
  {
    if (stepSize != 1)
    {
      // 处理多轴情况
      ProcessAxes();
      return;
    }
    if (typeKey == 0 && pType == 0)
    {
      // 处理可优化情况
      ProcessSqrtSumFloat();
      return;
    }
    // 处理一般情况
    int32_t loopCount = this->tileNum * this->bufferNum;
    for (int32_t i = 0; i < loopCount; i++)
    {
      CopyIn(i);
      Compute(i);
      CopyOut(i);
    }
    if (this->pType == 0)
    {
      // ========CopyIn======
      AscendC::LocalTensor<float> tLocal = inQueueT.AllocTensor<float>();
      AscendC::DataCopy(tLocal, yGm_f[0], this->yLength);
      inQueueT.EnQue(tLocal);
      if (this->typeKey == 1)
      {
        // ========Compute======
        AscendC::LocalTensor<DTYPE_X> yLocal = outQueueY.AllocTensor<DTYPE_X>();
        tLocal = inQueueT.DeQue<float>();
        AscendC::Sqrt(tLocal, tLocal, this->yLength);
        AscendC::Cast(yLocal, tLocal, AscendC::RoundMode::CAST_NONE, this->yLength);
        inQueueT.FreeTensor(tLocal);
        outQueueY.EnQue<DTYPE_X>(yLocal);
        // ========CopyOut======
        yLocal = outQueueY.DeQue<DTYPE_X>();
        AscendC::DataCopy(yGm[0], yLocal, this->yLength);
        outQueueY.FreeTensor(yLocal);
      }
      else
      {
        // ========Compute======
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        tLocal = inQueueT.DeQue<float>();
        AscendC::Sqrt(yLocal, tLocal, this->yLength);
        inQueueT.FreeTensor(tLocal);
        outQueueY.EnQue<float>(yLocal);
        // ========CopyOut======
        yLocal = outQueueY.DeQue<float>();
        AscendC::DataCopy(yGm_f[0], yLocal, this->yLength);
        outQueueY.FreeTensor(yLocal);
      }
    }
    else if (this->pType >= 1 && this->pType <= 4)
    {
      if (this->typeKey == 1)
      {
        // ========CopyIn======
        AscendC::LocalTensor<float> tLocal = inQueueT.AllocTensor<float>();
        AscendC::DataCopy(tLocal, yGm_f[0], this->yLength);
        inQueueT.EnQue(tLocal);
        // ========Compute======
        tLocal = inQueueT.DeQue<float>();
        AscendC::LocalTensor<DTYPE_X> yLocal = outQueueY.AllocTensor<DTYPE_X>();
        AscendC::Cast(yLocal, tLocal, AscendC::RoundMode::CAST_NONE, this->yLength);
        inQueueT.FreeTensor(tLocal);
        outQueueY.EnQue<DTYPE_X>(yLocal);
        // ========CopyOut======
        yLocal = outQueueY.DeQue<DTYPE_X>();
        AscendC::DataCopy(yGm[0], yLocal, this->yLength);
        outQueueY.FreeTensor(yLocal);
      }
      else
      {
        // ========CopyIn======
        AscendC::LocalTensor<float> tLocal = inQueueT.AllocTensor<float>();
        AscendC::DataCopy(tLocal, yGm_f[0], this->yLength);
        inQueueT.EnQue(tLocal);
        // ========Compute======
        tLocal = inQueueT.DeQue<float>();
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        AscendC::Adds(yLocal, tLocal, (float)0.0, this->yLength);
        inQueueT.FreeTensor(tLocal);
        outQueueY.EnQue<float>(yLocal);
        // ========CopyOut======
        yLocal = outQueueY.DeQue<float>();
        AscendC::DataCopy(yGm_f[0], yLocal, this->yLength);
        outQueueY.FreeTensor(yLocal);
      }
    }
    else
    {
      if (this->typeKey == 1)
      {
        // ========CopyIn======
        AscendC::LocalTensor<float> tLocal = inQueueT.AllocTensor<float>();
        AscendC::DataCopy(tLocal, yGm_f[0], this->yLength);
        inQueueT.EnQue(tLocal);
        // ========Compute======
        AscendC::LocalTensor<DTYPE_X> yLocal = outQueueY.AllocTensor<DTYPE_X>();
        tLocal = inQueueT.DeQue<float>();
        AscendC::Ln(tLocal, tLocal, this->tileLength);
        AscendC::Muls(tLocal, tLocal, (float)(1.0f / this->pValue), this->tileLength);
        AscendC::Exp(tLocal, tLocal, this->tileLength);
        AscendC::Cast(yLocal, tLocal, AscendC::RoundMode::CAST_NONE, this->yLength);
        inQueueT.FreeTensor(tLocal);
        outQueueY.EnQue<DTYPE_X>(yLocal);
        // ========CopyOut======
        yLocal = outQueueY.DeQue<DTYPE_X>();
        AscendC::DataCopy(yGm[0], yLocal, this->yLength);
        outQueueY.FreeTensor(yLocal);
      }
      else
      {
        // ========CopyIn======
        AscendC::LocalTensor<float> tLocal = inQueueT.AllocTensor<float>();
        AscendC::DataCopy(tLocal, yGm_f[0], this->yLength);
        inQueueT.EnQue(tLocal);
        // ========Compute======
        AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
        tLocal = inQueueT.DeQue<float>();
        AscendC::Ln(tLocal, tLocal, this->tileLength);
        AscendC::Muls(tLocal, tLocal, (float)(1.0f / this->pValue), this->tileLength);
        AscendC::Exp(yLocal, tLocal, this->tileLength);
        inQueueT.FreeTensor(tLocal);
        outQueueY.EnQue<float>(yLocal);
        // ========CopyOut======
        yLocal = outQueueY.DeQue<float>();
        AscendC::DataCopy(yGm_f[0], yLocal, this->yLength);
        outQueueY.FreeTensor(yLocal);
      }
    }
  }

private:
  __aicore__ inline void ProcessAxes()
  {
    int32_t loopCount = this->tileNum;
    for (int32_t k = 0; k < this->unitCount; k++)
    {
      for (int32_t j = 0; j < this->stepSize; j++)
      {
        for (int32_t i = 0; i < loopCount; i++)
        {
          CopyInAxes(i, j, k);
          Compute(i);
          CopyOutAxes(i, j, k);
        }
        if (this->pType == 0)
        {
          // ========CopyIn======
          AscendC::LocalTensor<float> tLocal = inQueueT.AllocTensor<float>();
          AscendC::DataCopy(tLocal, yGm_f[j + k * this->stepSize], this->yLength);
          inQueueT.EnQue(tLocal);
          if (this->typeKey == 1)
          {
            // ========Compute======
            AscendC::LocalTensor<DTYPE_X> yLocal = outQueueY.AllocTensor<DTYPE_X>();
            tLocal = inQueueT.DeQue<float>();
            AscendC::Sqrt(tLocal, tLocal, this->yLength);
            AscendC::Cast(yLocal, tLocal, AscendC::RoundMode::CAST_NONE, this->yLength);
            inQueueT.FreeTensor(tLocal);
            outQueueY.EnQue<DTYPE_X>(yLocal);
            // ========CopyOut======
            yLocal = outQueueY.DeQue<DTYPE_X>();
            AscendC::DataCopy(yGm[j + k * this->stepSize], yLocal, this->yLength);
            outQueueY.FreeTensor(yLocal);
          }
          else
          {
            // ========Compute======
            AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
            tLocal = inQueueT.DeQue<float>();
            AscendC::Sqrt(yLocal, tLocal, this->yLength);
            inQueueT.FreeTensor(tLocal);
            outQueueY.EnQue<float>(yLocal);
            // ========CopyOut======
            yLocal = outQueueY.DeQue<float>();
            AscendC::DataCopy(yGm_f[j + k * this->stepSize], yLocal, this->yLength);
            outQueueY.FreeTensor(yLocal);
          }
        }
        else if (this->pType >= 1 && this->pType <= 4)
        {
          if (this->typeKey == 1)
          {
            // ========CopyIn======
            AscendC::LocalTensor<float> tLocal = inQueueT.AllocTensor<float>();
            AscendC::DataCopy(tLocal, yGm_f[j + k * this->stepSize], this->yLength);
            inQueueT.EnQue(tLocal);
            // ========Compute======
            tLocal = inQueueT.DeQue<float>();
            AscendC::LocalTensor<DTYPE_X> yLocal = outQueueY.AllocTensor<DTYPE_X>();
            AscendC::Cast(yLocal, tLocal, AscendC::RoundMode::CAST_NONE, this->yLength);
            inQueueT.FreeTensor(tLocal);
            outQueueY.EnQue<DTYPE_X>(yLocal);
            // ========CopyOut======
            yLocal = outQueueY.DeQue<DTYPE_X>();
            AscendC::DataCopy(yGm[j + k * this->stepSize], yLocal, this->yLength);
            outQueueY.FreeTensor(yLocal);
          }
          else
          {
            // ========CopyIn======
            AscendC::LocalTensor<float> tLocal = inQueueT.AllocTensor<float>();
            AscendC::DataCopy(tLocal, yGm_f[j + k * this->stepSize], this->yLength);
            inQueueT.EnQue(tLocal);
            // ========Compute======
            tLocal = inQueueT.DeQue<float>();
            AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
            AscendC::Adds(yLocal, tLocal, (float)0.0, this->yLength);
            inQueueT.FreeTensor(tLocal);
            outQueueY.EnQue<float>(yLocal);
            // ========CopyOut======
            yLocal = outQueueY.DeQue<float>();
            AscendC::DataCopy(yGm_f[j + k * this->stepSize], yLocal, this->yLength);
            outQueueY.FreeTensor(yLocal);
          }
        }
        else
        {
          if (this->typeKey == 1)
          {
            // ========CopyIn======
            AscendC::LocalTensor<float> tLocal = inQueueT.AllocTensor<float>();
            AscendC::DataCopy(tLocal, yGm_f[j + k * this->stepSize], this->yLength);
            inQueueT.EnQue(tLocal);
            // ========Compute======
            AscendC::LocalTensor<DTYPE_X> yLocal = outQueueY.AllocTensor<DTYPE_X>();
            tLocal = inQueueT.DeQue<float>();
            AscendC::Ln(tLocal, tLocal, this->tileLength);
            AscendC::Muls(tLocal, tLocal, (float)(1.0f / this->pValue), this->tileLength);
            AscendC::Exp(tLocal, tLocal, this->tileLength);
            AscendC::Cast(yLocal, tLocal, AscendC::RoundMode::CAST_NONE, this->yLength);
            inQueueT.FreeTensor(tLocal);
            outQueueY.EnQue<DTYPE_X>(yLocal);
            // ========CopyOut======
            yLocal = outQueueY.DeQue<DTYPE_X>();
            AscendC::DataCopy(yGm[j + k * this->stepSize], yLocal, this->yLength);
            outQueueY.FreeTensor(yLocal);
          }
          else
          {
            // ========CopyIn======
            AscendC::LocalTensor<float> tLocal = inQueueT.AllocTensor<float>();
            AscendC::DataCopy(tLocal, yGm_f[j + k * this->stepSize], this->yLength);
            inQueueT.EnQue(tLocal);
            // ========Compute======
            AscendC::LocalTensor<float> yLocal = outQueueY.AllocTensor<float>();
            tLocal = inQueueT.DeQue<float>();
            AscendC::Ln(tLocal, tLocal, this->tileLength);
            AscendC::Muls(tLocal, tLocal, (float)(1.0f / this->pValue), this->tileLength);
            AscendC::Exp(yLocal, tLocal, this->tileLength);
            inQueueT.FreeTensor(tLocal);
            outQueueY.EnQue<float>(yLocal);
            // ========CopyOut======
            yLocal = outQueueY.DeQue<float>();
            AscendC::DataCopy(yGm_f[j + k * this->stepSize], yLocal, this->yLength);
            outQueueY.FreeTensor(yLocal);
          }
        }
      }
    }
  }

  __aicore__ inline void ProcessSqrtSumFloat()
  {
    int32_t loopCount = this->tileNum * 2;
    AscendC::LocalTensor<float> tLocal = inQueueT_DB.AllocTensor<float>();
    inQueueT_DB.EnQue<float>(tLocal);
    for (int32_t progress = 0; progress < loopCount; progress++)
    {
      CopyInSqrtSumFloat(progress);
      ComputeSqrtSumFloat(progress);
      CopyOutSqrtSumFloat(progress);
    }
    // ========Compute======
    tLocal = inQueueT_DB.DeQue<float>();
    AscendC::LocalTensor<float> yLocal = outQueueY_DB.AllocTensor<float>();
    AscendC::LocalTensor<float> wLocal = calcBuf.Get<float>();
    AscendC::ReduceSum(yLocal, tLocal, wLocal, this->tileLength);
    AscendC::Sqrt(yLocal, yLocal, this->yLength);
    inQueueT_DB.FreeTensor(tLocal);
    outQueueY_DB.EnQue(yLocal);
    // ========CopyOut======
    yLocal = outQueueY_DB.DeQue<float>();
    AscendC::DataCopy(yGm_f[0], yLocal, this->yLength);
    outQueueY_DB.FreeTensor(yLocal);
  }

  __aicore__ inline void CopyInSqrtSumFloat(int32_t progress)
  {
    AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX_DB.AllocTensor<DTYPE_X>();
    if (progress == this->tileNum * 2 - 2)
    {
      AscendC::DataCopy(
          xLocal[0],
          xGm[progress * this->tileLength],
          this->lasttileLengthAlign / 2);
    }
    else if (progress == this->tileNum * 2 - 1)
    {
      AscendC::DataCopy(xLocal[0],
               xGm[(progress - 1) * this->tileLength + this->lasttileLengthAlign / 2],
               this->lasttileLengthAlign / 2);
    }
    else
    {
      AscendC::DataCopy(xLocal[0], xGm[progress * this->tileLength],
               this->tileLength);
    }
    inQueueX_DB.EnQue(xLocal);
  }

  __aicore__ inline void ComputeSqrtSumFloat(int32_t progress)
  {
    AscendC::LocalTensor<float> tLocal = inQueueT_DB.DeQue<float>();
    AscendC::LocalTensor<float> xLocal_f = inQueueX_DB.DeQue<float>();
    AscendC::LocalTensor<float> yLocal = outQueueY_DB.AllocTensor<float>();
    if (progress >= this->tileNum * 2 - 2)
    {
      if (progress == 0)
      {
        AscendC::Mul(tLocal, xLocal_f, xLocal_f, this->lasttileLengthAlign / 2);
      }
      else
      {
        AscendC::Mul(xLocal_f, xLocal_f, xLocal_f, this->lasttileLengthAlign / 2);
        AscendC::Add(tLocal, tLocal, xLocal_f, this->lasttileLengthAlign / 2);
      }
    }
    else
    {
      if (progress == 0)
      {
        AscendC::Mul(tLocal, xLocal_f, xLocal_f, this->tileLength);
      }
      else
      {
        AscendC::Mul(xLocal_f, xLocal_f, xLocal_f, this->tileLength);
        AscendC::Add(tLocal, tLocal, xLocal_f, this->tileLength);
      }
    }
    inQueueT_DB.EnQue<float>(tLocal);
    outQueueY_DB.EnQue(yLocal);
    inQueueX_DB.FreeTensor(xLocal_f);
  }

  __aicore__ inline void CopyOutSqrtSumFloat(int32_t progress)
  {
    AscendC::LocalTensor<float> yLocal = outQueueY_DB.DeQue<float>();
    outQueueY_DB.FreeTensor(yLocal);
  }

  __aicore__ inline void CopyIn(int32_t progress)
  {
    AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
    AscendC::LocalTensor<float> tLocal = inQueueT.AllocTensor<float>();
    if (progress == this->tileNum - 1)
    {
      if (progress == 0)
      {
        AscendC::DataCopy(xLocal[0], xGm[0], this->tileLength);
        for (int i = this->totalLength; i < this->tileLength; i++)
        {
          if (pType == 0 || pType == 3 || pType == 4 || pType == 5)
          {
            xLocal.SetValue(i, (DTYPE_X)0);
          }
          else
          {
            xLocal.SetValue(i, xLocal.GetValue(0));
          }
        }
      }
      else
      {
        AscendC::DataCopy(
            xLocal[0],
            xGm[(progress - 1) * this->tileLength + this->lasttileLength],
            this->tileLength);
        for (int i = 0; i < this->tileLength - this->lasttileLength; i++)
        {
          if (pType == 0 || pType == 3 || pType == 4 || pType == 5)
          {
            xLocal.SetValue(i, (DTYPE_X)0);
          }
          else
          {
            xLocal.SetValue(i, xLocal.GetValue(this->tileLength - this->lasttileLength));
          }
        }
      }
    }
    else
    {
      AscendC::DataCopy(xLocal[0], xGm[progress * this->tileLength],
               this->tileLength);
    }

    AscendC::DataCopy(tLocal, yGm_f[0], this->yLength);
    if (progress == 0)
    {
      tLocal.SetValue(0, (DTYPE_X)0);
    }
    inQueueT.EnQue(tLocal);
    inQueueX.EnQue(xLocal);
  }

  __aicore__ inline void Compute(int32_t progress)
  {
    AscendC::LocalTensor<float> yLocal = outQueueY_F.AllocTensor<float>();
    AscendC::LocalTensor<float> tLocal = inQueueT.DeQue<float>();
    AscendC::LocalTensor<float> wLocal = calcBuf.Get<float>();

    if (this->typeKey == 0)
    {
      AscendC::LocalTensor<float> xLocal = inQueueX.DeQue<float>();
      AscendC::LocalTensor<float> xLocal_f = xLocal;

      if (pType == 0)
      {
        // sqrt(sum(abs(x)^2))
        AscendC::Mul(xLocal_f, xLocal_f, xLocal_f, this->tileLength);
        AscendC::ReduceSum(yLocal, xLocal_f, wLocal, this->tileLength);
        AscendC::Add(yLocal, yLocal, tLocal, this->yLength);
      }
      else if (pType == 1)
      {
        // max(abs(x))
        AscendC::Abs(xLocal_f, xLocal_f, this->tileLength);
        AscendC::ReduceMax(yLocal, xLocal_f, wLocal, this->tileLength, false);
        AscendC::Max(yLocal, yLocal, tLocal, this->yLength);
      }
      else if (pType == 2)
      {
        // min(abs(x))
        if (progress < 1)
        {
          AscendC::Abs(tLocal, xLocal_f, this->yLength);
        }
        AscendC::Abs(xLocal_f, xLocal_f, this->tileLength);
        AscendC::ReduceMin(yLocal, xLocal_f, wLocal, this->tileLength, false);
        AscendC::Min(yLocal, yLocal, tLocal, this->yLength);
      }
      else if (pType == 3)
      {
        // sum(x!=0)
        AscendC::LocalTensor<uint8_t> resLocal = calcBuf1.Get<uint8_t>();
        AscendC::LocalTensor<float> zeroLocal = calcBuf2.Get<float>();

        AscendC::Duplicate(zeroLocal, (float)0.0, this->tileLength);
        AscendC::Compare(resLocal, xLocal_f, zeroLocal, AscendC::CMPMODE::EQ, this->tileLength);
        AscendC::Select(zeroLocal, resLocal, zeroLocal, static_cast<float>(1),
               AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, this->tileLength);
        AscendC::ReduceSum(yLocal, zeroLocal, wLocal, this->tileLength);
        AscendC::Add(yLocal, yLocal, tLocal, this->yLength);
      }
      else if (pType == 4)
      {
        // sum(abs(x))
        AscendC::Abs(xLocal_f, xLocal_f, this->tileLength);
        AscendC::ReduceSum(yLocal, xLocal_f, wLocal, this->tileLength);
        AscendC::Add(yLocal, yLocal, tLocal, this->yLength);
      }
      else if (pType == 5)
      {
        // sum(abs(x)^p)^(1/p)
        AscendC::Abs(xLocal_f, xLocal_f, this->tileLength);
        AscendC::Ln(xLocal_f, xLocal_f, this->tileLength);
        AscendC::Muls(xLocal_f, xLocal_f, (float)(this->pValue), this->tileLength);
        AscendC::Exp(xLocal_f, xLocal_f, this->tileLength);
        AscendC::ReduceSum(yLocal, xLocal_f, wLocal, this->tileLength);
        AscendC::Add(yLocal, yLocal, tLocal, this->yLength);
      }
      outQueueY_F.EnQue<float>(yLocal);
      inQueueX.FreeTensor(xLocal);
      inQueueT.FreeTensor(tLocal);
    }
    else
    {
      AscendC::LocalTensor<half> xLocal = inQueueX.DeQue<half>();
      AscendC::LocalTensor<float> xLocal_f = xfBuf.Get<float>();
      AscendC::Cast(xLocal_f, xLocal, AscendC::RoundMode::CAST_NONE, this->tileLength);
      if (pType == 0)
      {
        // sqrt(sum(abs(x)^2))
        AscendC::Abs(xLocal_f, xLocal_f, this->tileLength);
        AscendC::Mul(xLocal_f, xLocal_f, xLocal_f, this->tileLength);
        AscendC::ReduceSum(yLocal, xLocal_f, wLocal, this->tileLength);
        AscendC::Add(yLocal, yLocal, tLocal, this->yLength);
      }
      else if (pType == 1)
      {
        // max(abs(x))
        AscendC::Abs(xLocal_f, xLocal_f, this->tileLength);
        AscendC::ReduceMax(yLocal, xLocal_f, wLocal, this->tileLength, false);
        AscendC::Max(yLocal, yLocal, tLocal, this->yLength);
      }
      else if (pType == 2)
      {
        // min(abs(x))
        if (progress < 1)
        {
          AscendC::Abs(tLocal, xLocal_f, this->yLength);
        }
        AscendC::Abs(xLocal_f, xLocal_f, this->tileLength);
        AscendC::ReduceMin(yLocal, xLocal_f, wLocal, this->tileLength, false);
        AscendC::Min(yLocal, yLocal, tLocal, this->yLength);
      }
      else if (pType == 3)
      {
        // sum(x!=0)
        AscendC::LocalTensor<uint8_t> resLocal = calcBuf1.Get<uint8_t>();
        AscendC::LocalTensor<float> zeroLocal = calcBuf2.Get<float>();

        AscendC::Duplicate(zeroLocal, (float)0.0, this->tileLength);
        AscendC::Compare(resLocal, xLocal_f, zeroLocal, AscendC::CMPMODE::EQ, this->tileLength);
        AscendC::Select(zeroLocal, resLocal, zeroLocal, static_cast<float>(1),
               AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, this->tileLength);
        AscendC::ReduceSum(yLocal, zeroLocal, wLocal, this->tileLength);
        AscendC::Add(yLocal, yLocal, tLocal, this->yLength);
      }
      else if (pType == 4)
      {
        // sum(abs(x))
        AscendC::Abs(xLocal_f, xLocal_f, this->tileLength);
        AscendC::ReduceSum(yLocal, xLocal_f, wLocal, this->tileLength);
        AscendC::Add(yLocal, yLocal, tLocal, this->yLength);
      }
      else if (pType == 5)
      {
        // sum(abs(x)^p)^(1/p)
        AscendC::Abs(xLocal_f, xLocal_f, this->tileLength);
        AscendC::Ln(xLocal_f, xLocal_f, this->tileLength);
        AscendC::Muls(xLocal_f, xLocal_f, (float)(this->pValue), this->tileLength);
        AscendC::Exp(xLocal_f, xLocal_f, this->tileLength);
        AscendC::ReduceSum(yLocal, xLocal_f, wLocal, this->tileLength);
        AscendC::Add(yLocal, yLocal, tLocal, this->yLength);
      }
      outQueueY_F.EnQue<float>(yLocal);
      inQueueX.FreeTensor(xLocal);
      inQueueT.FreeTensor(tLocal);
    }
  }

  __aicore__ inline void CopyOut(int32_t progress)
  {
    AscendC::LocalTensor<float> yLocal = outQueueY_F.DeQue<float>();
    AscendC::DataCopy(yGm_f[0], yLocal, this->yLength);
    outQueueY_F.FreeTensor(yLocal);
  }

  __aicore__ inline void CopyInAxes(int32_t progress, int j, int k)
  {
    AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
    AscendC::LocalTensor<float> tLocal = inQueueT.AllocTensor<float>();
    uint32_t startIndex = j + k * this->stepSize * this->totalLength +
                          progress * tileLength * this->stepSize;
    if (progress == this->tileNum - 1)
    {
      for (int i = 0; i < lasttileLength; i++)
      {
        xLocal.SetValue(i, xGm.GetValue(startIndex + i * stepSize));
      }
      for (int i = lasttileLength; i < this->tileLength; i++)
      {
        if (pType == 0 || pType == 3 || pType == 4 || pType == 5)
        {
          xLocal.SetValue(i, (DTYPE_X)0);
        }
        else
        {
          xLocal.SetValue(i, xLocal.GetValue(0));
        }
      }
    }
    else
    {
      for (int i = 0; i < tileLength; i++)
      {
        xLocal.SetValue(i, xGm.GetValue(startIndex + i * stepSize));
      }
    }
    AscendC::DataCopy(tLocal, yGm_f[j + k * this->stepSize], this->yLength);
    if (progress == 0)
    {
      tLocal.SetValue(0, (DTYPE_X)0);
    }
    inQueueT.EnQue(tLocal);
    inQueueX.EnQue(xLocal);
  }

  __aicore__ inline void CopyOutAxes(int32_t progress, int j, int k)
  {
    AscendC::LocalTensor<float> yLocal = outQueueY_F.DeQue<float>();
    AscendC::DataCopy(yGm_f[j + k * this->stepSize], yLocal, this->yLength);
    outQueueY_F.FreeTensor(yLocal);
  }

private:
  AscendC::TPipe pipe;
  AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueX, inQueueT;
  AscendC::TQue<AscendC::QuePosition::VECIN, 2> inQueueX_DB, inQueueT_DB;
  AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueueY, outQueueY_F;
  AscendC::TQue<AscendC::QuePosition::VECOUT, 2> outQueueY_DB;
  AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf, calcBuf1, calcBuf2, xfBuf;
  AscendC::GlobalTensor<DTYPE_X> xGm;
  AscendC::GlobalTensor<float> yGm_f;
  AscendC::GlobalTensor<DTYPE_Y> yGm;
  uint32_t blockLength;
  uint32_t tileNum;
  uint32_t tileNumAlign;
  uint32_t tileLength;
  uint32_t totalLength;
  uint32_t stepSize;
  uint32_t unitCount;
  uint32_t lasttileLength;
  uint32_t lasttileLengthAlign; // 64B对齐后的最后一个tile的长度
  uint32_t typeKey;
  uint32_t pType;
  uint32_t yLength; // 输出元素的最小搬运长度(32B)
  float pValue;
  uint32_t bufferNum; // BUFFER_NUM, 为2时开启Double Buffer
};

extern "C" __global__ __aicore__ void lp_norm_v2_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
  GET_TILING_DATA(tiling_data, tiling);

  KernelLpNormV2 op;

  op.Init(x, y, tiling_data.blockLength,
          tiling_data.tileNum, tiling_data.tileLength,
          tiling_data.lasttileLength, tiling_data.typeKey,
          tiling_data.pType, tiling_data.pValue, tiling_data.totalLength,
          tiling_data.stepSize, tiling_data.unitCount);
  op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
void lp_norm_v2_custom_do(uint32_t blockDim, void *l2ctrl, void *stream,
                          uint8_t *x, uint8_t *y,
                          uint8_t *workspace, uint8_t *tiling)
{

  lp_norm_v2_custom<<<blockDim, l2ctrl, stream>>>(x, y, workspace, tiling);
}
#endif