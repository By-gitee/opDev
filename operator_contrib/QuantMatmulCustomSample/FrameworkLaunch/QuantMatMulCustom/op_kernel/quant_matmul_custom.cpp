#include "kernel_operator.h"
#include "lib/matmul/matmul.h"
#include "lib/quantization/ascend_antiquant.h"

using namespace matmul_asc;

template<class aType, class bType, class cType, class biasType, typename Antiq_T, const AntiQuantConfig& Antiq_CFG>
class QuantMatmulCustom {
public:
    __aicore__ inline QuantMatmulCustom() {}
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR offset, GM_ADDR scale, GM_ADDR workspace,
        const QuantMatmulTilingData& tiling, TPipe& pipe)
    {
        tilingInfo = &tiling.customTilingData;

        //结构体初始化
        //Copy A GM->L1 & ND2NZ
        gm2l1Params.SetNdNum(1);
        gm2l1Params.SetNValue(tilingInfo->blockM);
        gm2l1Params.SetDValue(tilingInfo->blockKa);
        gm2l1Params.SetSrcDValue(tilingInfo->blockKa);
        gm2l1Params.SetDstNzC0Stride(tilingInfo->blockM);
        gm2l1Params.SetDstNzNStride(1);
        //Antiq
        antiqShapeInfo = {tilingInfo->blockOffsetK, tilingInfo->blockOffsetN,
            tilingInfo->blockOffsetK, tilingInfo->blockOffsetN};
        // Copy B UB->L1 & ND2NZ
        ub2l1Params.SetNdNum(1);
        ub2l1Params.SetNValue(tilingInfo->blockKb);
        ub2l1Params.SetDValue(tilingInfo->blockN);
        ub2l1Params.SetSrcDValue(tilingInfo->blockN);
        ub2l1Params.SetDstNzC0Stride(tilingInfo->blockKb);
        ub2l1Params.SetDstNzNStride(1);

        globalA.SetGlobalBuffer((__gm__ A_T*)(a), tiling.cubeTilingData.M * tiling.cubeTilingData.Ka);
        globalB.SetGlobalBuffer((__gm__ B_T*)(b), tilingInfo->weightK * tilingInfo->weightN);
        globalOffset.SetGlobalBuffer((__gm__ B_T*)(offset), tilingInfo->offsetK * tilingInfo->offsetN);
        globalScale.SetGlobalBuffer((__gm__ B_T*)(scale), tilingInfo->offsetK * tilingInfo->offsetN);
        globalC.SetGlobalBuffer((__gm__ C_T*)(c), tiling.cubeTilingData.M * tiling.cubeTilingData.N);

        pipe.InitBuffer(queA1, 1, tilingInfo->sizeAExpand);
        pipe.InitBuffer(queAntiqIn, 2, tilingInfo->sizeBlockAntiqIn);
        pipe.InitBuffer(queAntiqOut, 2, tilingInfo->sizeBlockAntiqOut);
        pipe.InitBuffer(queB1, 2, tilingInfo->sizeBlockAntiqOut);

        mm.SetSubBlockIdx(0);
        mm.Init(&tiling.cubeTilingData, &pipe);
    }
    __aicore__ inline void Process()
    {
        CopyInAFull();
        for (uint32_t subN = 0; subN < tilingInfo->tileN; subN++) {
            for (uint32_t subK = 0; subK < tilingInfo->tileK; subK++) {
                Compute(subN, subK);
            }
            CopyOut(subN);
        }
        ComputeEnd();
    }
private:
    __aicore__ inline void CopyInAFull()
    {
        localA1 = queA1.AllocTensor<A_T>();
        DataCopy(localA1, globalA, gm2l1Params);
        queA1.EnQue(localA1);
        localA1 = queA1.DeQue<A_T>();
    }
    __aicore__ inline void Compute(uint32_t subN, uint32_t subK)
    {
        // step1: copy b/scale/offset gm -> ub
        LocalTensor<B_T> localAntiqIn = queAntiqIn.AllocTensor<B_T>();
        DataCopy(localAntiqIn,
            globalB[subN * tilingInfo->blockWeightN * tilingInfo->weightK + subK * tilingInfo->blockWeight],
            tilingInfo->blockWeight);
        DataCopy(localAntiqIn[tilingInfo->blockWeight],
            globalOffset[subN * tilingInfo->blockN * tilingInfo->offsetK + subK * tilingInfo->blockOffset],
            tilingInfo->blockOffset);
        DataCopy(localAntiqIn[tilingInfo->blockWeightAndOffset],
            globalScale[subN * tilingInfo->blockN * tilingInfo->offsetK + subK * tilingInfo->blockOffset],
            tilingInfo->blockOffset);
        queAntiqIn.EnQue(localAntiqIn);

        // step2: antiq compute
        LocalTensor<B_T> localAntiqOut = queAntiqOut.AllocTensor<B_T>();
        localAntiqIn = queAntiqIn.DeQue<B_T>();

        // Antiquant内部会读取weight大小，这里需要还原为实际大小
        localAntiqIn.SetSize(tilingInfo->blockWeight);
        AscendAntiQuant<Antiq_T, B_T, false, Antiq_CFG>(
            localAntiqOut, localAntiqIn.template ReinterpretCast<Antiq_T>(),
            localAntiqIn[tilingInfo->blockWeight], localAntiqIn[tilingInfo->blockWeightAndOffset],
            static_cast<uint32_t>(tilingInfo->blockKb), antiqShapeInfo);

        queAntiqOut.EnQue(localAntiqOut);
        queAntiqIn.FreeTensor(localAntiqIn);

        // step3: copy b ub -> l1
        localAntiqOut = queAntiqOut.DeQue<B_T>();
        LocalTensor<B_T> localB1 = queB1.AllocTensor<B_T>();
        DataCopy(localB1, localAntiqOut, ub2l1Params);
        queAntiqOut.FreeTensor(localAntiqOut);
        queB1.EnQue(localB1);

        // step4: matmul compute
        localB1 = queB1.DeQue<B_T>();
        mm.SetTensorA(localA1[subK * tilingInfo->blockA]);
        mm.SetTensorB(localB1);
        mm.Iterate(subK != 0);
        queB1.FreeTensor(localB1);
    }
    __aicore__ inline void CopyOut(uint32_t subN)
    {
        mm.GetTensorC(globalC[subN * tilingInfo->blockN]);
    }
    __aicore__ inline void ComputeEnd()
    {
        mm.End();
        queA1.FreeTensor(localA1);
    }
private:
    using A_T = typename aType::T;
    using B_T = typename bType::T;
    using C_T = typename cType::T;

    TQue<QuePosition::A1, 1> queA1;
    TQue<QuePosition::VECIN, 1> queAntiqIn;
    TQue<QuePosition::VECOUT, 1> queAntiqOut;
    TQueBind<QuePosition::VECOUT, QuePosition::B1, 1> queB1;

    GlobalTensor<A_T> globalA;
    GlobalTensor<B_T> globalB;
    GlobalTensor<B_T> globalOffset;
    GlobalTensor<B_T> globalScale;
    GlobalTensor<C_T> globalC;
    LocalTensor<A_T> localA1;

    const CustomTilingInfo* tilingInfo;
    Nd2NzParams gm2l1Params;
    AntiQuantShapeInfo antiqShapeInfo;
    Nd2NzParams ub2l1Params;

    MatmulImpl<aType, bType, cType, biasType, CFG_MDL> mm;
};

#define GEMV_A_NORM_B_NORM_2BIT 1000
#define GEMV_A_NORM_B_NORM_4BIT 1001
#define GEMM_A_NORM_B_NORM_2BIT 1010
#define GEMM_A_NORM_B_NORM_4BIT 1011

extern "C" __global__ __aicore__ void quant_matmul_custom(GM_ADDR a, GM_ADDR b, GM_ADDR scale, GM_ADDR offset, GM_ADDR c, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    TPipe pipe;
    typedef MatmulType<AscendC::TPosition::TSCM, CubeFormat::NZ, half> bType;
    typedef MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half> cType;
    typedef MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half> biasType;
    static constexpr AntiQuantConfig Antiq_CFG = {AntiQuantMode::INPUT_INTLV};

    if (TILING_KEY_IS(GEMV_A_NORM_B_NORM_2BIT)) {
        typedef MatmulType<AscendC::TPosition::TSCM, CubeFormat::VECTOR, half> aType;
        QuantMatmulCustom<aType, bType, cType, biasType, uint2b_t, Antiq_CFG> op;
        op.Init(a, b, c, offset, scale, workspace, tilingData, pipe);
        op.Process();
    } else if (TILING_KEY_IS(GEMV_A_NORM_B_NORM_4BIT)) {
        typedef MatmulType<AscendC::TPosition::TSCM, CubeFormat::VECTOR, half> aType;
        QuantMatmulCustom<aType, bType, cType, biasType, int4b_t, Antiq_CFG> op;
        op.Init(a, b, c, offset, scale, workspace, tilingData, pipe);
        op.Process();
    } else if (TILING_KEY_IS(GEMM_A_NORM_B_NORM_2BIT)) {
        typedef MatmulType<AscendC::TPosition::TSCM, CubeFormat::NZ, half> aType;
        QuantMatmulCustom<aType, bType, cType, biasType, uint2b_t, Antiq_CFG> op;
        op.Init(a, b, c, offset, scale, workspace, tilingData, pipe);
        op.Process();
    } else if (TILING_KEY_IS(GEMM_A_NORM_B_NORM_4BIT)) {
        typedef MatmulType<AscendC::TPosition::TSCM, CubeFormat::NZ, half> aType;
        QuantMatmulCustom<aType, bType, cType, biasType, int4b_t, Antiq_CFG> op;
        op.Init(a, b, c, offset, scale, workspace, tilingData, pipe);
        op.Process();
    }
}