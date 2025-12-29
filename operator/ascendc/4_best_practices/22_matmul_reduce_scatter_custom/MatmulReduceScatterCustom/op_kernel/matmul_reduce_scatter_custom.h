/**
 * @file matmul_reduce_scatter_custom.h
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef MC2_MM_SCATTER_H
#define MC2_MM_SCATTER_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "matmul_reduce_scatter_custom_common.h"
#include "matmul_reduce_scatter_custom_compute.h"

namespace AscendC {

template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void MatMulKernelReduceScatter(GM_ADDR aAddr, GM_ADDR bGM, GM_ADDR computeResAddr,
    GM_ADDR biasGM, TCubeTiling& tiling, ReduceScatterRCSTiling& cfg, AscendC::Hccl<AscendC::HCCL_SERVER_TYPE_AICPU> &hccl,
    uint32_t tileCnt, AscendC::HcclHandle handleId[])
{
    if (GetBlockIdx() >= tiling.usedCoreNum) {
        for (int i=0; i< tileCnt; i++) {
            CrossCoreSetFlag<0x0, PIPE_FIX>(0x8);
            CrossCoreWaitFlag(0x8);
        }
        return;
    }

    using A_T = typename A_TYPE::T;
    using B_T = typename B_TYPE::T;
    using C_T = typename C_TYPE::T;
    using BiasT = typename BIAS_TYPE::T;

    auto aOffset = tiling.M *  tiling.Ka * sizeof(A_T);
    auto cOffset = tiling.M *  tiling.N  * sizeof(C_T);
    auto rankOffset = (sizeof(A_T) * cfg.rankM) / cfg.rankDim * tiling.Ka;
    // 处理带C‘场景
    uint32_t indexC = 0;
    uint8_t enAtomicC = 0;

    MatmulCompute<A_TYPE, B_TYPE, C_TYPE, BIAS_TYPE> mm;
    // AllReduce需要提前计算一次C矩阵的Offset地址
    mm.Init(tiling, cfg);
    mm.InitGlobalBTensor(bGM, biasGM);

    for (uint32_t i = 0; i < tileCnt; i++) {
        for (uint32_t j = 0U; j< cfg.rankDim; j++) {
            auto aWorkAddr = aAddr + j * rankOffset;
            mm.InitGlobalATensor(aWorkAddr, aOffset, computeResAddr, cOffset);
            mm.Compute(indexC, enAtomicC);
            computeResAddr += cOffset;
        }
        CrossCoreSetFlag<0x0, PIPE_FIX>(0x8);
        CrossCoreWaitFlag(0x8);
        hccl.Commit(handleId[i]);
        aAddr += aOffset;
    }
    mm.End();
}
}
#endif // MC2_MM_SCATTER_H
