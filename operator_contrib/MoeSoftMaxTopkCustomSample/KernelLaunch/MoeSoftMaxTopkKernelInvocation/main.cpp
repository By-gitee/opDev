/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 * This file constains code of cpu debug and npu code.We read data from bin file
 * and write result to file.
 */
#include "data_utils.h"
#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
extern void moe_soft_max_topk_do(uint32_t coreDim, void* l2ctrl, void* stream, uint8_t* x, uint8_t* y, uint8_t* indices, uint32_t k);
#else
#include "tikicpulib.h"
extern "C" __global__ __aicore__ void moe_soft_max_topk(GM_ADDR x, GM_ADDR y, GM_ADDR indices, uint32_t k);
#endif

std::string caseId = "";
int x_shape = 1024;
int y_shape = 16;
int k = 4;

int32_t main(int32_t argc, char* argv[])
{
    caseId = std::string(argv[1]);
    if(caseId == "case2"){
        x_shape = 2048;
        y_shape = 32;
        k = 6;
    }
    uint32_t blockDim = 48;
    size_t inputSize_x = x_shape * y_shape  * sizeof(float);
    size_t outputSize_y = x_shape * k * sizeof(float);
    size_t outputSize_indices = x_shape * k * sizeof(int32_t);

    
#ifdef ASCENDC_CPU_DEBUG
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputSize_x);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputSize_y);
    uint8_t* indices = (uint8_t*)AscendC::GmAlloc(outputSize_indices);

    ReadFile("./input/input_x.bin", inputSize_x, x, inputSize_x);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(moe_soft_max_topk, blockDim, x, y, indices, k); // use this macro for cpu debug

    WriteFile("./output/output_y.bin", y, outputSize_y);
    WriteFile("./output/output_indices.bin", indices, outputSize_indices);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)indices);
#else
    CHECK_ACL(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t *xHost, *yHost, *indicesHost;
    uint8_t *xDevice, *yDevice, *indicesDevice;

    CHECK_ACL(aclrtMallocHost((void**)(&xHost), inputSize_x));
    CHECK_ACL(aclrtMallocHost((void**)(&yHost), outputSize_y));
    CHECK_ACL(aclrtMallocHost((void**)(&indicesHost), outputSize_indices));

    CHECK_ACL(aclrtMalloc((void**)&xDevice, inputSize_x, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void**)&yDevice, outputSize_y, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void**)&indicesDevice, outputSize_indices, ACL_MEM_MALLOC_HUGE_FIRST));


    ReadFile("./input/input_x.bin", inputSize_x, xHost, inputSize_x);

    CHECK_ACL(aclrtMemcpy(xDevice, inputSize_x, xHost, inputSize_x, ACL_MEMCPY_HOST_TO_DEVICE));

    moe_soft_max_topk_do(blockDim, nullptr, stream, xDevice, yDevice, indicesDevice, k);
    CHECK_ACL(aclrtSynchronizeStream(stream));

    CHECK_ACL(aclrtMemcpy(yHost, outputSize_y, yDevice, outputSize_y, ACL_MEMCPY_DEVICE_TO_HOST));
    CHECK_ACL(aclrtMemcpy(indicesHost, outputSize_indices, indicesDevice, outputSize_indices, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile("./output/output_y.bin", yHost, outputSize_y);
    WriteFile("./output/output_indices.bin", indicesHost, outputSize_indices);

    CHECK_ACL(aclrtFree(xDevice));
    CHECK_ACL(aclrtFree(yDevice));
    CHECK_ACL(aclrtFree(indicesDevice));

    CHECK_ACL(aclrtFreeHost(xHost));
    CHECK_ACL(aclrtFreeHost(yHost));
    CHECK_ACL(aclrtFreeHost(indicesHost));

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
#endif
    return 0;
}
