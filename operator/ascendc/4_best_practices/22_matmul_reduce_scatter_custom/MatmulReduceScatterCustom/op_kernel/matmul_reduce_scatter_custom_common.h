/**
 * @file matmul_reduce_scatter_custom_common.h
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef MC2_ALLREDUCE_COMM_H
#define MC2_ALLREDUCE_COMM_H

#if defined ASCENDC_CPU_DEBUG
#define SET_G_CORE_TYPE_IS_AIV thread_local int g_coreType = 2
#define SET_G_CORE_TYPE_IS_AIC thread_local int g_coreType = 1
#define DTYPE_X1 half
#define DTYPE_X2 half
#define DTYPE_Y half
#else
#define SET_G_CORE_TYPE_IS_AIV
#define SET_G_CORE_TYPE_IS_AIC
#endif

#include "lib/hccl/hccl.h"

namespace AscendC {
// 代码多数据类型支持
using A_DTYPE = DTYPE_X1;
using B_DTYPE = DTYPE_X1;
using C_DTYPE = DTYPE_Y;
using BIAS_DTYPE = DTYPE_Y;

using namespace matmul;
template <class A_TYPE, class B_TYPE, class C_TYPE, class BIAS_TYPE>
__aicore__ inline void CalcGMOffset(int blockIdx, int usedCoreNum, TCubeTiling &param, uint64_t &offsetA, uint64_t &offsetB,
    uint64_t &offsetC, uint64_t &offsetBias, int32_t isTransposeAIn, int32_t isTransposeBIn)
{
    auto temp0 = Ceil(param.M, param.singleCoreM);
    auto temp1 = Ceil(param.N, param.singleCoreN);
    auto temp2 = Ceil(param.Ka, param.singleCoreK); // 不切K， 应该=1

    auto divideKcoreNum = usedCoreNum / temp2;

    auto mCoreIndx = (blockIdx % divideKcoreNum) % temp0; // 必须沿着N 轴方向输出
    auto nCoreIndx = (blockIdx % divideKcoreNum) / temp0;
    auto subKindx = blockIdx / divideKcoreNum; // 缺省为0

    if constexpr (A_TYPE::format == CubeFormat::ND) {
        if (isTransposeAIn > 0) {
            offsetA = mCoreIndx * param.singleCoreM + subKindx * param.M * param.singleCoreK;
        } else {
            offsetA = mCoreIndx * param.Ka * param.singleCoreM + subKindx * param.singleCoreK;
        }
    } else if constexpr (A_TYPE::format == CubeFormat::NZ) {
        offsetA = subKindx * param.singleCoreK * param.M + mCoreIndx * param.singleCoreM * BLOCK_CUBE;
    } else if constexpr (A_TYPE::format == CubeFormat::SCALAR) {
    } else if constexpr (A_TYPE::format == CubeFormat::VECTOR) {
    } else {
        ASSERT(false && "Data format of A matrix should be ND or NZ.");
    }

    if constexpr (B_TYPE::format == CubeFormat::ND) {
        if (isTransposeBIn > 0) {
            offsetB = subKindx * param.singleCoreK + nCoreIndx * param.Ka * param.singleCoreN;
        } else {
            offsetB = subKindx * param.singleCoreK * param.N + nCoreIndx * param.singleCoreN;
        }
    } else if constexpr (B_TYPE::format == CubeFormat::NZ) {
        if (isTransposeBIn > 0) {
            offsetB =  nCoreIndx * param.singleCoreN * 16;
        }
        else {
            offsetB = param.Ka * nCoreIndx * param.singleCoreN + subKindx * param.singleCoreK * BLOCK_CUBE;
        }
    } else {
        ASSERT(false && "Data format of B matrix should be ND or NZ.");
    }

    if constexpr (C_TYPE::format == CubeFormat::ND || C_TYPE::format == CubeFormat::ND_ALIGN) {
        offsetC = mCoreIndx * param.N * param.singleCoreM + nCoreIndx * param.singleCoreN;
    } else if constexpr (C_TYPE::format == CubeFormat::NZ) {
        offsetC = param.M * nCoreIndx * param.singleCoreN + mCoreIndx * param.singleCoreM * BLOCK_CUBE;
    } else {
        ASSERT(false && "Data format of C matrix should be ND or ND_ALIGN or NZ.");
    }

    if constexpr (BIAS_TYPE::format == CubeFormat::ND) {
        offsetBias = nCoreIndx * param.singleCoreN;
    } else {
        ASSERT(false && "Data format of BIAS should be ND.");
    }

    // 尾块M
    int gmUseM = param.M - mCoreIndx * param.singleCoreM;
    param.singleCoreM = gmUseM < param.singleCoreM ? gmUseM : param.singleCoreM;

    // 尾块N
    int gmUseN = param.N - nCoreIndx * param.singleCoreN;
    param.singleCoreN = gmUseN < param.singleCoreN ? gmUseN : param.singleCoreN;

    // 尾块K
    int gmUseK = param.Ka - subKindx * param.singleCoreK;
    param.singleCoreK = gmUseK < param.singleCoreK ? gmUseK : param.singleCoreK;
}

__aicore__ __inline__ GM_ADDR GetTailA(GM_ADDR aGM, TCubeTiling& tiling, uint32_t size)
{
    return aGM + (tiling.M * tiling.Ka) * sizeof(A_DTYPE) * size;
}
__aicore__ __inline__ GM_ADDR GetTailC(GM_ADDR cGM, TCubeTiling& tiling, uint32_t size)
{
    return cGM + (tiling.M * tiling.N) * sizeof(C_DTYPE) * size;
}

}
#endif // MC2_ALLREDUCE_COMM_H
