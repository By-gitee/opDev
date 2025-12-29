/**
 * @file cube_group_custom.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace matmul;
constexpr uint32_t BOLCKSTART = 6*2; // block start number
constexpr uint32_t BOLCKSIZE= 6*2; // block size
constexpr uint32_t MSGQUEUESIZE = 48; // message numbers
constexpr uint32_t GROUPID = 1;

struct CubeMsgBody
{
    CubeGroupMsgHead head;
    uint8_t funcID;
    uint8_t skipCnt;
    uint32_t value;
    bool isTransA;
    bool isTransB;
    bool isAtomic;
    bool isLast;
    int32_t tailM;
    int32_t tailN;
    int32_t tailK; 
    uint64_t aAddr;
    uint64_t bAddr;
    uint64_t cAddr;
    uint64_t aGap;
    uint64_t bGap;
};

// User-defined Init an Call function
template<class MatmulApiCfg, typename CubeMsgBody>
struct MyCallbackFunc
{
    template<int32_t funcID>
    __aicore__ inline static typename IsEqual<funcID, 0>::Type CubeGroupCallBack(MatmulApiCfg &mm, __gm__ CubeMsgBody *rcvMsg, CubeResGroupHandle<CubeMsgBody> &handle)
    {
        GlobalTensor<int64_t> msgGlobal;
        msgGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *> (rcvMsg) + sizeof(int64_t));
        DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT> (msgGlobal);
        using SrcAT = typename MatmulApiCfg::AType::T;
        auto skipNum = 0;
        for (int i = 0; i < skipNum + 1; ++i)
        {
            auto tmpId = handle.FreeMessage(rcvMsg + i); // msg process is complete
        }
        handle.SetSkipMsg(skipNum);
    }
    template<int32_t funcID>
    __aicore__ inline static typename IsEqual<funcID, 1>::Type CubeGroupCallBack(MatmulApiCfg &mm, __gm__ CubeMsgBody *rcvMsg, CubeResGroupHandle<CubeMsgBody> &handle)
    {
        GlobalTensor<int64_t> msgGlobal;
        msgGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *> (rcvMsg) + sizeof(int64_t));
        DataCacheCleanAndInvalid<int64_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT> (msgGlobal);
        using SrcAT = typename MatmulApiCfg::AType::T;
        LocalTensor<SrcAT> tensor_temp;
        auto skipNum = 3;
        auto tmpId = handle.FreeMessage(rcvMsg, CubeMsgState::VALID);
        for (int i = 1; i < skipNum + 1; ++i)
        {
            auto tmpId = handle.FreeMessage(rcvMsg + i, CubeMsgState::FAKE);
        }
        handle.SetSkipMsg(skipNum); // notify the cube not to process
    }
    __aicore__ inline static void Call(MatmulApiCfg &mm, __gm__ CubeMsgBody *rcvMsg, CubeResGroupHandle<CubeMsgBody> &handle)
    {
        if (rcvMsg->funcID == 0)
        {
            CubeGroupCallBack<0> (mm, rcvMsg, handle);
        }
        else if(rcvMsg->funcID == 1)
        {
            CubeGroupCallBack<1> (mm, rcvMsg, handle);
        }
    }
    __aicore__ inline static void Init(MyCallbackFunc<MatmulApiCfg, CubeMsgBody> &foo, MatmulApiCfg &mm, GM_ADDR tilingGM)
    {
        auto tempTilingGM = (__gm__ uint32_t*)tilingGM;
        auto tempTiling = (uint32_t*)&(foo.tiling);
        for (int i = 0; i < sizeof(TCubeTiling) / sizeof(int32_t); ++i, ++tempTilingGM, ++tempTiling)
        {
            *tempTiling = *tempTilingGM;
        }
        mm.SetSubBlockIdx(0);
        mm.Init(&foo.tiling, GetTPipePtr());
    }
    TCubeTiling tiling;
};


template <typename A_TYPE, typename B_TYPE, typename C_TYPE, typename BIAS_TYPE>
class CubeGroupKernel {
public:
    __aicore__ inline CubeGroupKernel(){};
    __aicore__ inline void Init(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR cGm, GM_ADDR biasGM, 
                                GM_ADDR tilingGM, GM_ADDR workspaceGM, uint32_t isTransposeAIn, uint32_t isTransposeBIn)
    {
        KfcWorkspace desc(workspaceGM);
        using MatmulApiType = MatmulImpl<A_TYPE, B_TYPE, C_TYPE, C_TYPE, CFG_NORM>;
        handle = CreateCubeResGroup<GROUPID, MatmulApiType, MyCallbackFunc, CubeMsgBody> (desc, BOLCKSTART, BOLCKSIZE, MSGQUEUESIZE, tilingGM);
    };

    __aicore__ inline void Process()
    {
        auto queIdx = GetBlockIdx();
        handle.AssignQueue(queIdx); // assign queue
        CubeGroupMsgHead head = {CubeMsgState::VALID, (uint8_t)queIdx};
        CubeMsgBody aCubeMsgBody {head, 0, 0, 0, false, false, false, false, 0, 0, 0, 0, 0, 0, 0, 0};
        CubeMsgBody bCubeMsgBody {head, 1, 0, 0, false, false, false, false, 0, 0, 0, 0, 0, 0, 0, 0};
        auto offset = 0;
        if (GetBlockIdx() == 0)
        {
            auto msgPtr = handle.template AllocMessage(); // alloc for queue space
            offset = handle.template PostMessage(msgPtr, bCubeMsgBody); // post true msg
            bool waitState = handle.template Wait<true> (offset); // wait until the msg is proscessed
        }
        else if (GetBlockIdx() < 4)
        {
            auto msgPtr = handle.AllocMessage();
            offset = handle.PostFakeMsg(msgPtr); // post fake msgPtr
            bool waitState = handle.template Wait<true> (offset); // wait until the msg is proscessed
        }
        else
        {
            auto msgPtr = handle.template AllocMessage();
            offset = handle.template PostMessage(msgPtr, aCubeMsgBody);
            bool waitState = handle.template Wait<true> (offset); // wait until the msg is proscessed
        }
        auto msgPtr = handle.AllocMessage();
        handle.SetQuit(msgPtr); // set quit info
    };

private:
    TPipe pipe;
    CubeResGroupHandle<CubeMsgBody> handle;
};


extern "C" __global__ __aicore__ void cube_group_custom(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c, GM_ADDR workspace,
                                                    GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    typedef MatmulType<TPosition::GM, CubeFormat::ND, half> aType;
    typedef MatmulType<TPosition::GM, CubeFormat::ND, half> bType;
    typedef MatmulType<TPosition::LCM, CubeFormat::ND, float> cType;
    typedef MatmulType<TPosition::GM, CubeFormat::ND, float> biasType;
    CubeGroupKernel<aType, bType, cType, biasType> op;
    op.Init(a, b, c, bias, tiling, workspace, 0, 0);
    op.Process();
}