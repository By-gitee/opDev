/**
 * @file matmul_custom.cpp
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace matmul;

__aicore__ inline uint64_t MMLcm(uint64_t m, uint64_t n) {
  if (m == 0 || n == 0) {
    return 0; // 处理输入为0的情况
  }
  uint64_t total = m * n;
  uint64_t tmp = 0;
  while (n != 0) {
    tmp = m % n;
    m = n;
    n = tmp;
  }
  return total / m;
}

template <typename T> __aicore__ inline int64_t DivCeil(int64_t a, int64_t b) {
  if (b == 0) {
    return a;
  }
  return (a + b - 1) / b;
}

class BaseBlock {
public:
  __aicore__ inline BaseBlock() {}

  template <class C_TYPE>
  __aicore__ inline void Init(const TCubeTiling &cubeTiling, uint32_t nTile,
                              uint32_t mTile, uint32_t mTileBlock,
                              uint32_t nTileBlock);

  // 更新当前切块偏移，并更新当前切块内包含结果基础块个数,M和N方向上的块数
  __aicore__ inline void UpdateBlockByCount(const TCubeTiling &cubeTiling,
                                            uint64_t mTileIndex,
                                            uint64_t nTileIndex);

  // 初始化当前切块是否有基础块分配到当前核运算，需要运算的基础块下标
  __aicore__ inline void InitBlockIndex(const TCubeTiling &cubeTiling,
                                        uint64_t processIndex,
                                        uint64_t cube_idx);

  // 错位分核更新基本索引，根据给定的processIndex更新currentBlockIndex
  __aicore__ inline void UpdateBasicIndex(const TCubeTiling &cubeTiling,
                                          uint64_t processIndex,
                                          uint64_t cube_idx);

  // 计算GM偏移量，根据给定的mTile和nTile的索引更新offsetA、offsetB、offsetC
  __aicore__ inline void CalcGMOffset(const TCubeTiling &cubeTiling,
                                      uint64_t mTileIndex, uint64_t nTileIndex);

  // 当前处理的块索引
  uint64_t currentBlockIndex;
  // M方向上每个mTile包含的base块个数
  uint64_t numBaseBlocksPerMTile;
  // N方向上每个nTile包含的base块个数
  uint64_t numBaseBlocksPerNTile;
  // 计算结果基础块总数
  uint64_t totalNumBlocks;
  uint64_t singleCoreM = 0;
  uint64_t singleCoreN = 0;
  // N方向上的base尾块个数
  uint64_t baseTailNumN;
  // M方向上的base尾块大小
  uint64_t baseTailNumM;
  // L2上M切分次数
  uint64_t mTileCntL2;
  // L2上N切分次数
  uint64_t nTileCntL2;
  // M方向上mTile尾块里的base块的个数
  uint64_t numBaseBlocksInMTileTail;
  // N方向上nTile尾块里的base块的个数
  uint64_t numBaseBlocksInNTileTail;
  // M方向上的总块数
  uint64_t totalNumBlocksM;
  // N方向上的总块数
  uint64_t totalNumBlocksN;
  // 每个核处理的轮数
  uint64_t numRoundsPerCore;
  // 当前核实际处理的轮数
  uint64_t actualNumRounds;
  // 尾核处理的块数
  uint64_t numBlocksForTailCore;
  // 当前核实际处理的M方向上的块数
  uint64_t numBlocksUsedM;
  // 当前核实际处理的N方向上的块数
  uint64_t numBlocksUsedN;
  // 行优先或列优先的顺序
  uint32_t rowOrder = 0;
  // 开始运算时，首核的索引
  uint64_t startBlockIndex = 0;
  // 结束运算时，尾核的索引
  uint64_t endBlockIndex = 0;

  // M方向上当前mTile的地址偏移
  uint64_t mTileAddressOffset;
  // N方向上当前nTile的地址偏移
  uint64_t nTileAddressOffset;

  // 索引是否初始化的标志
  bool indexInit_ = false;
  // A矩阵的偏移量
  uint64_t offsetA = 0;
  // B矩阵的偏移量
  uint64_t offsetB = 0;
  // C矩阵的偏移量
  uint64_t offsetC = 0;
  // 偏置的偏移量
  uint64_t offsetBias = 0;
};

template <class C_TYPE>
__aicore__ inline void
BaseBlock::Init(const TCubeTiling &cubeTiling, uint32_t nTile, uint32_t mTile,
                uint32_t mTileBlock, uint32_t nTileBlock) {
  nTileCntL2 = nTile;
  mTileCntL2 = mTile;
  totalNumBlocksM =
      (static_cast<uint64_t>(cubeTiling.M) + cubeTiling.singleCoreM - 1) /
      cubeTiling.singleCoreM; // m切分块数
  totalNumBlocksN =
      (static_cast<uint64_t>(cubeTiling.N) + cubeTiling.singleCoreN - 1) /
      cubeTiling.singleCoreN; // n切分块数
  baseTailNumN =
      static_cast<uint64_t>(cubeTiling.N) -
      (totalNumBlocksN - 1) * cubeTiling.singleCoreN; // n方向上的base尾块
  baseTailNumM =
      static_cast<uint64_t>(cubeTiling.M) -
      (totalNumBlocksM - 1) * cubeTiling.singleCoreM; // m方向上的base尾块
  currentBlockIndex = 0;
  numBaseBlocksPerMTile = (totalNumBlocksM + mTileCntL2 - 1) /
                          mTileCntL2; // 每一份mTile包含的base块个数
  numBaseBlocksPerNTile = (totalNumBlocksN + nTileCntL2 - 1) /
                          nTileCntL2; // 每一份nTile包含的base块个数

  if (mTileBlock > 0 && nTileBlock > 0) {
    numBaseBlocksPerMTile = mTileBlock;
    numBaseBlocksPerNTile = nTileBlock;
  }

  totalNumBlocks = numBaseBlocksPerMTile * numBaseBlocksPerNTile;
  numBaseBlocksInMTileTail =
      totalNumBlocksM -
      (mTileCntL2 - 1) *
          numBaseBlocksPerMTile; // M方向上mTile尾块里的base块的个数
  numBaseBlocksInNTileTail =
      totalNumBlocksN -
      (nTileCntL2 - 1) *
          numBaseBlocksPerNTile; // M方向上nTile尾块里的base块的个数
  numRoundsPerCore =
      (totalNumBlocks + cubeTiling.usedCoreNum - 1) / cubeTiling.usedCoreNum;
  actualNumRounds = 0;
  numBlocksForTailCore = totalNumBlocks % cubeTiling.usedCoreNum;
  numBlocksUsedM = numBaseBlocksPerMTile;
  numBlocksUsedN = numBaseBlocksPerNTile;

  mTileAddressOffset = 0;
  nTileAddressOffset = 0;
}

__aicore__ inline void
BaseBlock::UpdateBlockByCount(const TCubeTiling &cubeTiling,
                              uint64_t mTileIndex, uint64_t nTileIndex) {
  // 计算M方向上当前tile的地址偏移
  mTileAddressOffset =
      mTileIndex * numBaseBlocksPerMTile * cubeTiling.singleCoreM;
  // 计算N方向上当前tile的地址偏移
  nTileAddressOffset =
      nTileIndex * numBaseBlocksPerNTile * cubeTiling.singleCoreN;

  // 判断当前是否为M或者N方向上尾块，并根据是否是尾块设置当前核运算块数
  if ((mTileIndex == (mTileCntL2 - 1)) && (nTileIndex == (nTileCntL2 - 1))) {
    totalNumBlocks = numBaseBlocksInMTileTail * numBaseBlocksInNTileTail;
    numBlocksUsedM = numBaseBlocksInMTileTail;
    numBlocksUsedN = numBaseBlocksInNTileTail;
  } else if (mTileIndex == (mTileCntL2 - 1)) {
    totalNumBlocks = numBaseBlocksInMTileTail * numBaseBlocksPerNTile;
    numBlocksUsedM = numBaseBlocksInMTileTail;
    numBlocksUsedN = numBaseBlocksPerNTile;
  } else if (nTileIndex == (nTileCntL2 - 1)) {
    totalNumBlocks = numBaseBlocksPerMTile * numBaseBlocksInNTileTail;
    numBlocksUsedM = numBaseBlocksPerMTile;
    numBlocksUsedN = numBaseBlocksInNTileTail;
  } else {
    totalNumBlocks = numBaseBlocksPerMTile * numBaseBlocksPerNTile;
    numBlocksUsedM = numBaseBlocksPerMTile;
    numBlocksUsedN = numBaseBlocksPerNTile;
  }
  // 更新实际的轮数和块数
  numRoundsPerCore =
      DivCeil(totalNumBlocks, static_cast<uint64_t>(cubeTiling.usedCoreNum));
  numBlocksForTailCore = totalNumBlocks % cubeTiling.usedCoreNum;
  if (numBlocksForTailCore == 0) {
    numBlocksForTailCore = static_cast<uint64_t>(cubeTiling.usedCoreNum);
  }
}

__aicore__ inline void BaseBlock::InitBlockIndex(const TCubeTiling &cubeTiling,
                                                 uint64_t processIndex,
                                                 uint64_t cube_idx) {
  if (indexInit_) {
    startBlockIndex = endBlockIndex; // 开始运算时，首核的索引
  } else {
    startBlockIndex = processIndex * numBlocksForTailCore %
                      cubeTiling.usedCoreNum; // 开始运算时，首核的索引
    indexInit_ = true;
  }
  endBlockIndex = (startBlockIndex + numBlocksForTailCore) %
                  cubeTiling.usedCoreNum; // 结束运算时，尾核的索引
  uint64_t indexStart = startBlockIndex;
  uint64_t indexEnd = endBlockIndex;

  if (indexStart < indexEnd) {
    // 正常排序, numBlocksForTailCore在整个Cores的中间
    if (cube_idx < indexStart) {
      currentBlockIndex = cube_idx * (numRoundsPerCore - 1);
      actualNumRounds = numRoundsPerCore - 1;
    } else if (cube_idx < indexEnd) {
      currentBlockIndex = indexStart * (numRoundsPerCore - 1) +
                          (cube_idx - indexStart) * numRoundsPerCore;
      actualNumRounds = numRoundsPerCore;
    } else {
      currentBlockIndex = (indexStart * (numRoundsPerCore - 1) +
                           numBlocksForTailCore * numRoundsPerCore +
                           (cube_idx - indexEnd) * (numRoundsPerCore - 1));
      actualNumRounds = numRoundsPerCore - 1;
    }
  } else if (indexEnd < indexStart) {
    // indexEnd会翻转
    if (cube_idx < indexEnd) {
      currentBlockIndex = cube_idx * numRoundsPerCore;
      actualNumRounds = numRoundsPerCore;
    } else if (cube_idx < indexStart) {
      currentBlockIndex = indexEnd * numRoundsPerCore +
                          (cube_idx - indexEnd) * (numRoundsPerCore - 1);
      actualNumRounds = numRoundsPerCore - 1;
    } else {
      currentBlockIndex = (indexEnd * numRoundsPerCore +
                           (indexStart - indexEnd) * (numRoundsPerCore - 1) +
                           (cube_idx - indexStart) * numRoundsPerCore);
      actualNumRounds = numRoundsPerCore;
    }
  } else {
    // 不存在尾核，基本块对齐
    currentBlockIndex = cube_idx * numRoundsPerCore;
    actualNumRounds = numRoundsPerCore;
  }
}

__aicore__ inline void
BaseBlock::UpdateBasicIndex(const TCubeTiling &cubeTiling,
                            uint64_t processIndex, uint64_t cube_idx) {
  uint64_t newBlockIdx = (cube_idx + cubeTiling.usedCoreNum - startBlockIndex) %
                             cubeTiling.usedCoreNum +
                         processIndex * cubeTiling.usedCoreNum;
  uint64_t mIdx = newBlockIdx % numBlocksUsedM;
  uint64_t nIdx = 0;
  if (numBlocksUsedM != 0 && numBlocksUsedN != 0) {
    nIdx = (newBlockIdx + newBlockIdx / MMLcm(numBlocksUsedM, numBlocksUsedN)) %
           numBlocksUsedN;
  }
  currentBlockIndex = mIdx * numBlocksUsedN + nIdx;
}

__aicore__ inline void BaseBlock::CalcGMOffset(const TCubeTiling &cubeTiling,
                                               uint64_t mTileIndex,
                                               uint64_t nTileIndex) {
  uint64_t mCntIndex = currentBlockIndex / numBlocksUsedN;
  uint64_t nCntIndex = currentBlockIndex % numBlocksUsedN;

  offsetA = mCntIndex * cubeTiling.singleCoreM * cubeTiling.Ka +
            mTileAddressOffset * cubeTiling.Ka;

  offsetB = nCntIndex * cubeTiling.singleCoreN + nTileAddressOffset;

  offsetC = (nCntIndex * cubeTiling.singleCoreN +
             mCntIndex * cubeTiling.singleCoreM * cubeTiling.N +
             (mTileAddressOffset * cubeTiling.N + nTileAddressOffset));
}

extern "C" __global__ __aicore__ void
matmul_leakyrelu_custom(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c,
                        GM_ADDR workspace, GM_ADDR tiling) { 
  GET_TILING_DATA(tilingData, tiling);
  using A_T = float;
  using B_T = float;
  using C_T = float;
  using BiasT = float;
  TPipe que;
  TCubeTiling cubeTiling = tilingData.cubeTilingData;

  GlobalTensor<A_T> aGlobal;
  GlobalTensor<B_T> bGlobal;
  GlobalTensor<C_T> cGlobal;

  aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ A_T *>(a),
                          cubeTiling.M * cubeTiling.Ka);
  bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ B_T *>(b),
                          cubeTiling.Kb * cubeTiling.N);
  cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ C_T *>(c),
                          cubeTiling.M * cubeTiling.N);

  cGlobal.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
  typedef MatmulType<AscendC::TPosition::GM, CubeFormat::ND, A_T> aType;
  typedef MatmulType<AscendC::TPosition::GM, CubeFormat::ND, B_T> bType;
  typedef MatmulType<AscendC::TPosition::GM, CubeFormat::ND, C_T> cType;
  typedef MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BiasT> biasType;
  constexpr static MatmulConfig MCFG_MDL =
      GetMDLConfig(false, false, 0, false, false, false, true);
  Matmul<aType, bType, cType, biasType, MCFG_MDL,
         MatmulCallBackFunc<nullptr, nullptr, nullptr>>
      mm;

  AscendC::LocalTensor<C_T> reluOutLocal;
  AscendC::TQue<AscendC::QuePosition::VECIN, 1> reluOutQueue_;
  que.InitBuffer(reluOutQueue_, 1,
                 cubeTiling.baseM * cubeTiling.baseN * sizeof(C_T));
  REGIST_MATMUL_OBJ(&que, GetSysWorkSpacePtr(), mm, &cubeTiling);

  BaseBlock BaseBlock;
  BaseBlock.Init<C_T>(cubeTiling, tilingData.nTile, tilingData.mTile,
                      tilingData.mTileBlock, tilingData.nTileBlock);

  mm.SetHF32(
      false,
      0); // L0A/L0B中的FP32数据将在矩阵乘法之前被CUBE舍入为HF32，可以有效提升性能
  bool reverse = true;
  for (uint64_t mTileIndex = 0; mTileIndex < BaseBlock.mTileCntL2;
       mTileIndex++) {
    reverse = !reverse;
    for (uint64_t nTileIndexTemp = 0; nTileIndexTemp < BaseBlock.nTileCntL2;
         nTileIndexTemp++) {
      uint64_t nTileIndex = reverse
                                ? (BaseBlock.nTileCntL2 - nTileIndexTemp - 1)
                                : nTileIndexTemp;
      BaseBlock.UpdateBlockByCount(cubeTiling, mTileIndex, nTileIndex);

      auto cube_idx = GetBlockIdx();
      // 使用Vector对cube发送运算命令时，2Vector对应1个cube，所以实际上Vector对应的Cube应该是0,20,1,21,2,22,3,23
      if (cube_idx % 2 == 0) {
        cube_idx = cube_idx / 2;
      } else {
        cube_idx = (cube_idx + 41) / 2 - 1;
      }
      BaseBlock.InitBlockIndex(cubeTiling, 0, cube_idx);

      for (uint64_t j = 0; j < BaseBlock.actualNumRounds; j++) {
        if (BaseBlock.rowOrder == 0) {
          BaseBlock.UpdateBasicIndex(cubeTiling, j, cube_idx);
        }
        if (BaseBlock.currentBlockIndex < BaseBlock.totalNumBlocks) {
          reluOutLocal = reluOutQueue_.AllocTensor<C_T>();
          BaseBlock.CalcGMOffset(cubeTiling, mTileIndex, nTileIndex);
          mm.SetSingleShape(BaseBlock.singleCoreM, BaseBlock.singleCoreN,
                            cubeTiling.singleCoreK);
          mm.SetTensorA(aGlobal[BaseBlock.offsetA], false);
          mm.SetTensorB(bGlobal[BaseBlock.offsetB], false);
          mm.Iterate();
          mm.GetTensorC(cGlobal[BaseBlock.offsetC]);
          // //
          // 当前使用mm.GetTensorC(reluOutLocal);会卡死，只能先搬运到全局内存，再搬运回来

          int32_t eventIDMTE3_V = static_cast<int32_t>(
              GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_V));
          AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(eventIDMTE3_V);
          AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(eventIDMTE3_V);

          DataCopyParams copyInParam = {
              (uint16_t)cubeTiling.baseM,
              (uint16_t)(cubeTiling.baseN * sizeof(C_T) / DEFAULT_C0_SIZE),
              (uint16_t)((cubeTiling.N - cubeTiling.baseN) * sizeof(C_T) /
                         DEFAULT_C0_SIZE),
              0};
          DataCopy(reluOutLocal, cGlobal[BaseBlock.offsetC], copyInParam);
          reluOutQueue_.EnQue(reluOutLocal);
          reluOutLocal = reluOutQueue_.DeQue<C_T>();

          int32_t eventIDMTE2_S = static_cast<int32_t>(
              GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_S));
          AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(eventIDMTE2_S);
          AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(eventIDMTE2_S);

          LeakyRelu(reluOutLocal, reluOutLocal, tilingData.alpha,
                    cubeTiling.baseM * cubeTiling.baseN);

          DataCopyParams copyOutParam = {
              (uint16_t)cubeTiling.baseM,
              (uint16_t)(cubeTiling.baseN * sizeof(C_T) / DEFAULT_C0_SIZE), 0,
              (uint16_t)((cubeTiling.N - cubeTiling.baseN) * sizeof(C_T) /
                         DEFAULT_C0_SIZE)};
          DataCopy(cGlobal[BaseBlock.offsetC], reluOutLocal, copyOutParam);

          reluOutQueue_.FreeTensor(reluOutLocal);
        }
      }
    }
  }

  PipeBarrier<PIPE_ALL>();
  SetAtomicNone();
}