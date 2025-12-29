/**
 * @file matmul_noABshare_custom.cpp
 *
 * Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace matmul;

__aicore__ inline void CopyTiling(TCubeTiling *tiling, GM_ADDR tilingGM) // copy tiling info
{
    uint32_t *ptr = reinterpret_cast<uint32_t *>(tiling);
    auto tiling32 = reinterpret_cast<__gm__ uint32_t *>(tilingGM);

    for (uint32_t i = 0; i < sizeof(TCubeTiling) / sizeof(uint32_t); i++, ptr++) {
        *ptr = *(tiling32 + i);
    }
    return;
}

template <typename aType, typename bType, typename cType> class MatmutNoABshareKernel {
public:
    __aicore__ inline MatmutNoABshareKernel(){};
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace,
                                const TCubeTiling &tiling, AscendC::TPipe *pipe);
    __aicore__ inline void Process(AscendC::TPipe *pipe);
    __aicore__ inline void CalcOffset(int32_t blockIdx, const TCubeTiling &tiling, int32_t &offsetA, int32_t &offsetB,
                                      int32_t &offsetC);

    Matmul<MatmulType<AscendC::TPosition::GM, CubeFormat::ND, aType>, 
           MatmulType<AscendC::TPosition::GM, CubeFormat::ND, bType>,
           MatmulType<AscendC::TPosition::VECIN, CubeFormat::ND, cType>>
        matmulObj;

    AscendC::GlobalTensor<aType> aGlobal;
    AscendC::GlobalTensor<bType> bGlobal;
    AscendC::GlobalTensor<cType> cGlobal;
    TCubeTiling tiling;
};

template <typename aType, typename bType, typename cType>
__aicore__ inline void MatmutNoABshareKernel<aType, bType, cType>::Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, 
                                                                GM_ADDR workspace,const TCubeTiling &tiling, AscendC::TPipe *pipe)
{
    this->tiling = tiling;
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ aType *>(a), tiling.M * tiling.Ka);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ bType *>(b), tiling.Kb * tiling.N);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ cType *>(c), tiling.M * tiling.N);

    int32_t offsetA, offsetB, offsetC;
    CalcOffset(AscendC::GetBlockIdx(), tiling, offsetA, offsetB, offsetC); // calculate offset
    aGlobal = aGlobal[offsetA];
    bGlobal = bGlobal[offsetB];
    cGlobal = cGlobal[offsetC];
}

template <typename aType, typename bType, typename cType>
__aicore__ inline void MatmutNoABshareKernel<aType, bType, cType>::Process(AscendC::TPipe *pipe)
{
    AscendC::InitOutput<float> (cGlobal, tiling.M * tiling.N, 0); // init output zero
    SyncAll();
    matmulObj.Init(&tiling);
    matmulObj.SetTensorA(aGlobal);
    matmulObj.SetTensorB(bGlobal);
    matmulObj.SetSingleShape(tiling.M, tiling.N, tiling.Ka/2); // set matmul single-core process shape
    matmulObj.IterateAll(cGlobal, 1);
    matmulObj.End();
}

template <typename aType, typename bType, typename cType>
__aicore__ inline void
MatmutNoABshareKernel<aType, bType, cType>::CalcOffset(int32_t blockIdx, const TCubeTiling &tiling,
                                                             int32_t &offsetA, int32_t &offsetB, int32_t &offsetC)
{
    if (blockIdx ==0)
    {
        offsetA = 0;
        offsetB = 0;
    }else if (blockIdx == 1){
        offsetA = tiling.Ka/2; // cut in half according to the k axis
        offsetB = tiling.Ka/2 * tiling.N; // cut in half according to the k axis
    }
    offsetC = 0;
}

extern "C" __global__ __aicore__ void matmul_noABshare_custom(GM_ADDR a, GM_ADDR b, GM_ADDR c,
                                                              GM_ADDR workspace, GM_ADDR tilingGm)
{
    AscendC::TPipe pipe;
    TCubeTiling tiling;
    CopyTiling(&tiling, tilingGm);

    MatmutNoABshareKernel<half, half, float> MatmutNoABshareKernel;
    MatmutNoABshareKernel.Init(a, b, c, workspace, tiling, &pipe);
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), MatmutNoABshareKernel.matmulObj, &MatmutNoABshareKernel.tiling);
    MatmutNoABshareKernel.Process(&pipe);
}