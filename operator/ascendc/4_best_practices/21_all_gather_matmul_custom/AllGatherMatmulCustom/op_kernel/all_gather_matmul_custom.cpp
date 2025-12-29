/**
 * @file all_gather_matmul_custom.h
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "all_gather_matmul_custom_tiling.h"
using namespace AscendC;
using MATMUL_TYPE = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>;

__aicore__ inline void MatmulKernel(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM, TCubeTiling &tiling,
                                    MatmulImpl<MATMUL_TYPE, MATMUL_TYPE, MATMUL_TYPE> &mm)
{
    if (GetBlockIdx() >= tiling.usedCoreNum) {
        return;
    }
    GlobalTensor<half> aGlobal, bGlobal, cGlobal;
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(aGM), tiling.M * tiling.Ka);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(bGM), tiling.Ka * tiling.N);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(cGM), tiling.M * tiling.N);

    int mSingleBlocks = (tiling.M + tiling.singleCoreM - 1) / tiling.singleCoreM;
    int mCoreIndex = GetBlockIdx() % mSingleBlocks;
    int nCoreIndex = GetBlockIdx() / mSingleBlocks;
    int offsetA = mCoreIndex * tiling.Ka * tiling.singleCoreM;
    int offsetB = nCoreIndex * tiling.singleCoreN;
    int offsetC = mCoreIndex * tiling.N * tiling.singleCoreM + nCoreIndex * tiling.singleCoreN;
    int tailM = Std::min(tiling.M - mCoreIndex * tiling.singleCoreM, tiling.singleCoreM);
    int tailN = Std::min(tiling.N - nCoreIndex * tiling.singleCoreN, tiling.singleCoreN);

    mm.SetOrgShape(tiling.M, tiling.N, tiling.Ka, tiling.Kb);
    mm.SetTensorA(aGlobal[offsetA]);
    mm.SetTensorB(bGlobal[offsetB]);
    mm.SetTail(tailM, tailN);
    mm.IterateAll(cGlobal[offsetC]);
}

extern "C" __global__ __aicore__ void all_gather_matmul_custom(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGM,
                                                               GM_ADDR gatherOutGM, GM_ADDR workspaceGM,
                                                               GM_ADDR tilingGM)
{
    if ASCEND_IS_AIV {
        return;
    }
    REGISTER_TILING_DEFAULT(AllGatherMatmulCustomTilingData);
    GET_TILING_DATA(tilingData, tilingGM);
    TPipe pipe;

    auto &&localTiling = tilingData.localTiling;
    auto &&tileTiling = tilingData.tileTiling;
    auto &&tailTiling = tilingData.tailTiling;
    const auto tileNum = tilingData.cfg.tileNum;
    const auto tailNum = tilingData.cfg.tailNum;
    const auto aTileEleCnt = tileTiling.M * tileTiling.Ka;
    const auto aTileSize = tileTiling.M * tileTiling.Ka * sizeof(half);
    const auto cTileSize = tileTiling.M * tileTiling.N * sizeof(half);
    const auto aTailEleCnt = tailTiling.M * tailTiling.Ka;
    const auto aRankEleCnt = localTiling.M * localTiling.Ka;
    const auto aRankSize = localTiling.M * localTiling.Ka * sizeof(half);
    const auto cRankSize = localTiling.M * localTiling.N * sizeof(half);

    Hccl hccl;
    GM_ADDR contextGM = GetHcclContext<HCCL_GROUP_ID_0>();
    hccl.InitV2(contextGM, &tilingData);
    hccl.SetCcTilingV2(offsetof(AllGatherMatmulCustomTilingData, mc2CcTiling));
    auto handleId =
        hccl.AllGather<true>(aGM, gatherOutGM, aTileEleCnt, HcclDataType::HCCL_DATA_TYPE_FP16, aRankEleCnt, tileNum);
    auto tailHandleId = hccl.AllGather<true>(aGM + tileNum * aTileSize, gatherOutGM + tileNum * aTileSize, aTailEleCnt,
                                             HcclDataType::HCCL_DATA_TYPE_FP16, aRankEleCnt, tailNum);

    MatmulImpl<MATMUL_TYPE, MATMUL_TYPE, MATMUL_TYPE> mm;
    mm.SetSubBlockIdx(0);
    mm.Init(&localTiling);
    MatmulKernel(aGM, bGM, cGM + hccl.GetRankId() * cRankSize, localTiling, mm);

    auto aAddr = gatherOutGM;
    auto cAddr = cGM;
    mm.Init(&tileTiling);
    for (uint32_t i = 0; i < tileNum; i++) {
        hccl.Wait(handleId);
        for (uint32_t rankId = 0; rankId < hccl.GetRankDim(); rankId++) {
            if (rankId == hccl.GetRankId())
                continue;
            MatmulKernel(aAddr + rankId * aRankSize, bGM, cAddr + rankId * cRankSize, tileTiling, mm);
        }
        aAddr += aTileSize;
        cAddr += cTileSize;
    }

    aAddr = gatherOutGM + tileNum * aTileSize;
    cAddr = cGM + tileNum * cTileSize;
    if (tailNum > 0) {
        mm.Init(&tailTiling);
        hccl.Wait(tailHandleId);
        for (uint32_t rankId = 0; rankId < hccl.GetRankDim(); rankId++) {
            if (rankId == hccl.GetRankId())
                continue;
            MatmulKernel(aAddr + rankId * aRankSize, bGM, cAddr + rankId * cRankSize, tailTiling, mm);
        }
    }

    mm.End();
    hccl.Finalize();
}