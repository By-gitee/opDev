/**
 * @file matmul_reduce_scatter_custom.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

#include "matmul_reduce_scatter_custom.h"
#include "matmul_reduce_scatter_custom_common.h"
#include "matmul_reduce_scatter_custom_tiling.h"
#include "lib/matmul_intf.h"
#include "kernel_operator.h"
#include "kernel_operator_intf.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void matmul_reduce_scatter_custom(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM,
    GM_ADDR cGM, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    if (workspaceGM == nullptr) {
        return;
    }
    if ASCEND_IS_AIV {
        return;
    }
    GM_ADDR userWS = GetUserWorkspace(workspaceGM);
    if (userWS == nullptr) {
        return;
    }

    REGISTER_TILING_DEFAULT(MatmulReduceScatterCustomTilingData);
    auto tiling1 = (__gm__ MatmulReduceScatterCustomTilingData*)tilingGM;
    __gm__ void *mc2InitTiling = (__gm__ void *)(&(tiling1->mc2InitTiling));
    __gm__ void *mc2CcTiling = (__gm__ void *)(&(tiling1->mc2CcTiling));

    GET_TILING_DATA(tilingData, tilingGM);
    auto &&tiling = tilingData.matmulTiling;
    auto &&cfg = tilingData.param;
    auto &&tailTiling = tilingData.tailTiling;

    TPipe pipe;

    Hccl hccl;
    GM_ADDR contextGM = GetHcclContext<HCCL_GROUP_ID_0>();
    hccl.Init(contextGM, mc2InitTiling);
    hccl.SetCcTiling(mc2CcTiling);

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    if (TILING_KEY_IS(1000UL)) {
        using aType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X1>;
        using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_X2>;
        using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, DTYPE_Y>;
        using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>;

        GM_ADDR aAddr = aGM;
        GM_ADDR cAddr = cGM;
        
        GM_ADDR computeResAddrGM = workspaceGM;
        bool determinism = false;
        GM_ADDR computeResAddr = computeResAddrGM;

        uint64_t tileLen = tiling.M * tiling.N;
        AscendC::HcclHandle handleIds[256];
        if (cfg.tileCnt > 0) {
            for (size_t i = 0; i < cfg.tileCnt; i++) {
                handleIds[i] = hccl.ReduceScatter(computeResAddr + (i * 2 * tileLen * cfg.rankDim),
                    cAddr + (i * 2 * tileLen), tileLen, AscendC::HCCL_DATA_TYPE_FP16, HCCL_REDUCE_SUM, 0, 1);
            }
            AscendC::MatMulKernelReduceScatter<aType, bType, cType, biasType>(aAddr, bGM, computeResAddr,
                biasGM, tiling, cfg, hccl, cfg.tileCnt, handleIds);
        }
        AscendC::HcclHandle handleIdTail[1];
        if (cfg.tailM != 0) {
            GM_ADDR tailaAddr = GetTailA(aGM, tiling, cfg.tileCnt);
            GM_ADDR tailcAddr = GetTailC(cGM, tiling, cfg.tileCnt);
            GM_ADDR tailComputeResAddr = GetTailC(computeResAddrGM, tiling, cfg.tileCnt * cfg.rankDim);
            uint64_t tailLen = tailTiling.M * tailTiling.N;
            handleIdTail[0] =  hccl.ReduceScatter(tailComputeResAddr, tailcAddr, tailLen,
                AscendC::HCCL_DATA_TYPE_FP16, AscendC::HCCL_REDUCE_SUM, 0, cfg.tailCnt);
            AscendC::MatMulKernelReduceScatter<aType, bType, cType, biasType>(tailaAddr, bGM, tailComputeResAddr,
                biasGM, tailTiling, cfg, hccl, cfg.tailCnt, handleIdTail);
        }
        for (uint32_t i = 0; i< cfg.tileCnt; i++) {
            hccl.Wait(handleIds[i]);
        }
        if (cfg.tailM != 0) {
            for (uint32_t i = 0; i< cfg.tailCnt; i++) {
                hccl.Wait(handleIdTail[i]);
            }
        }
    }
    CrossCoreSetFlag<0x0, PIPE_FIX>(0x8);
    CrossCoreWaitFlag(0x8);
    hccl.Finalize();
}