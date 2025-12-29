/**
 * @file matmul_all_reduce_custom.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

#include "matmul_all_reduce_custom.h"
#include "matmul_all_reduce_custom_common.h"
#include "matmul_all_reduce_custom_tiling.h"
#include "lib/matmul_intf.h"
#include "kernel_operator.h"
#include "kernel_operator_intf.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void matmul_all_reduce_custom(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM,
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

    REGISTER_TILING_DEFAULT(MatmulAllReduceCustomTilingData);
    auto tiling1 = (__gm__ MatmulAllReduceCustomTilingData*)tilingGM;
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
        
        GM_ADDR computeResAddrGM = cGM;
        bool determinism = false;
        if (cfg.useBufferType == static_cast<uint8_t>(AscendC::MC2_BUFFER_LOCATION::MC2_BUFFER_TYPE_WINDOW_IN) &&
            !determinism) {
                computeResAddrGM = hccl.GetWindowsInAddr(hccl.GetRankId());
        }
        GM_ADDR computeResAddr = computeResAddrGM;

        AscendC::HcclHandle handleId;
        if (cfg.tileCnt > 0) {
            auto tileLen = tiling.M * tiling.N;
            handleId = hccl.AllReduce<false>(computeResAddr, cAddr, tileLen, AscendC::HCCL_DATA_TYPE_FP16,
                HCCL_REDUCE_SUM, cfg.tileCnt);
            AscendC::MatMulKernelAllReduce<aType, bType, cType, biasType>(aAddr, bGM, cAddr, computeResAddr,
                biasGM, tiling, cfg, hccl, cfg.tileCnt, handleId);
        }

        AscendC::HcclHandle handleIdTail = -1;
        if (cfg.tailM != 0) {
            aAddr = GetTailA(aGM, tiling, cfg.tileCnt);
            cAddr = GetTailC(cGM, tiling, cfg.tileCnt);
            computeResAddr = GetTailC(computeResAddrGM, tiling, cfg.tileCnt);
            uint64_t tailLen = tailTiling.M * tailTiling.N;
            handleIdTail =  hccl.AllReduce(computeResAddr, cAddr, tailLen,
                AscendC::HCCL_DATA_TYPE_FP16, AscendC::HCCL_REDUCE_SUM, cfg.tailCnt);
            AscendC::MatMulKernelAllReduce<aType, bType, cType, biasType>(aAddr, bGM, cAddr, computeResAddr,
                biasGM, tailTiling, cfg, hccl, cfg.tailCnt, handleIdTail);
        }
        if (cfg.tileCnt > 0) {
            hccl.Wait(handleId);
        }

        if (cfg.tailM != 0) {
            hccl.Wait(handleIdTail);
        }
    }
    CrossCoreSetFlag<0x0, PIPE_FIX>(0x8);
    CrossCoreWaitFlag(0x8);
    hccl.Finalize();
}