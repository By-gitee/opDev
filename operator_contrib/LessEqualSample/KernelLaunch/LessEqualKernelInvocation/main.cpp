/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 * This file constains code of cpu debug and npu code.We read data from bin file
 * and write result to file.
 */
#include "less_equal_tiling.h"
#include "data_utils.h"
#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#include "aclrtlaunch_less_equal.h"
#else
#include "tikicpulib.h"
extern "C" __global__ __aicore__ void less_equal(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace, SplitTilingData tiling);
#endif
using T = int8_t;

int32_t main(int32_t argc, char* argv[])
{
    size_t tilingSize = sizeof(SplitTilingData);
    size_t usrWorkSpaceSize = 4096;
    size_t sysWorkSpaceSize = 16 * 1024 * 1024;
    size_t inputByteSize = 16 * sizeof(T);
    size_t outputByteSize = 16 * sizeof(int8_t);
    uint32_t blockDim = 1; //默认核心全开

#ifdef ASCENDC_CPU_DEBUG
    uint8_t* usrWorkSpace = (uint8_t*)AscendC::GmAlloc(usrWorkSpaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    gen_splitTiling<T>(inputByteSize, blockDim, tiling);
    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    ReadFile("./input/input_x1.bin", inputByteSize, x1, inputByteSize);
    ReadFile("./input/input_x2.bin", inputByteSize, x2, inputByteSize);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(less_equal, blockDim, x1, x2, y, usrWorkSpace, *reinterpret_cast<SplitTilingData*>(tiling));
    WriteFile("./output/output_y.bin", y, outputByteSize);

    AscendC::GmFree((void*)x1);
    AscendC::GmFree((void*)x2);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)usrWorkSpace);
    AscendC::GmFree((void*)tiling);
#else
    CHECK_ACL(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));
    SplitTilingData* tiling;
    uint8_t* x1_host, * x2_host, * y_host, * workSpace_host;
    uint8_t* x1_device, * x2_device, * y_device, * workSpace_device;
    CHECK_ACL(aclrtMallocHost((void**)(&tiling), tilingSize));
    gen_splitTiling<T>(inputByteSize, blockDim, tiling);
    CHECK_ACL(aclrtMallocHost((void**)(&x1_host), inputByteSize));
    CHECK_ACL(aclrtMallocHost((void**)(&x2_host), inputByteSize));
    CHECK_ACL(aclrtMallocHost((void**)(&y_host), outputByteSize));
    CHECK_ACL(aclrtMallocHost((void**)(&workSpace_host), sysWorkSpaceSize + usrWorkSpaceSize));
    CHECK_ACL(aclrtMalloc((void**)&x1_device, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void**)&x2_device, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void**)&y_device, outputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void**)&workSpace_device, sysWorkSpaceSize + usrWorkSpaceSize, ACL_MEM_MALLOC_HUGE_FIRST));
    
    ReadFile("./input/input_x2.bin", inputByteSize, x2_host, inputByteSize);
    ReadFile("./input/input_x1.bin", inputByteSize, x1_host, inputByteSize);
    CHECK_ACL(aclrtMemcpy(x2_device, inputByteSize, x2_host, inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(x1_device, inputByteSize, x1_host, inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    ACLRT_LAUNCH_KERNEL(less_equal)(blockDim, stream, x1_device, x2_device, y_device, workSpace_device, tiling);
    CHECK_ACL(aclrtSynchronizeStream(stream));
    CHECK_ACL(aclrtMemcpy(y_host, outputByteSize, y_device, outputByteSize, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile("./output/output_y.bin", y_host, outputByteSize);
    
    CHECK_ACL(aclrtFree(x1_device));
    CHECK_ACL(aclrtFree(x2_device));
    CHECK_ACL(aclrtFree(y_device));
    CHECK_ACL(aclrtFree(workSpace_device));
    CHECK_ACL(aclrtFreeHost(x1_host));
    CHECK_ACL(aclrtFreeHost(x2_host));
    CHECK_ACL(aclrtFreeHost(y_host));
    CHECK_ACL(aclrtFreeHost(tiling));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
#endif
    return 0;
}
