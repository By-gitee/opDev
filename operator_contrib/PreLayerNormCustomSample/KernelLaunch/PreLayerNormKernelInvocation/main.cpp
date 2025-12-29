/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2023. All rights reserved.
 * This file constains code of cpu debug and npu code.We read data from bin file
 * and write result to file.
 */
#include "data_utils.h"
#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
extern void pre_layer_norm_custom_do(uint32_t coreDim, void* l2ctrl, void* stream, uint8_t* x, uint8_t* y, uint8_t* gamma, uint8_t* beta, uint8_t* z, uint32_t caseNum);
#else
#include "tikicpulib.h"
extern "C" __global__ __aicore__ void pre_layer_norm_custom(GM_ADDR x, GM_ADDR y, GM_ADDR gamma, GM_ADDR beta,GM_ADDR z, uint32_t caseNum);
#endif

std::string caseId = "";
int x_shape = 4980;
int y_shape = 4;
int z_shape = 2048;
float epsilon = 1e-5;
uint32_t caseNum = 1;
int32_t main(int32_t argc, char* argv[])
{
    caseId = std::string(argv[1]);
    if(caseId == "case2"){
        x_shape = 512;
        z_shape = 20480;
        epsilon = 1e-7;
        caseNum = 2;
    }
    uint32_t blockDim = 48;
    size_t inputSize_x = x_shape * y_shape * z_shape * sizeof(float);
    size_t inputSize_y = x_shape * y_shape * z_shape * sizeof(float);
    size_t outputSize_z = x_shape * y_shape * z_shape * sizeof(float);
    size_t inputSize_gamma = z_shape * sizeof(float);
    size_t inputSize_beta = z_shape * sizeof(float);
    
#ifdef ASCENDC_CPU_DEBUG
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputSize_x);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(inputSize_y);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(inputSize_gamma);
    uint8_t* beta = (uint8_t*)AscendC::GmAlloc(inputSize_beta);
    uint8_t* z = (uint8_t*)AscendC::GmAlloc(outputSize_z);

    ReadFile("./input/input_x.bin", inputSize_x, x, inputSize_x);
    ReadFile("./input/input_y.bin", inputSize_y, y, inputSize_y);
    ReadFile("./input/input_gamma.bin", inputSize_gamma, gamma, inputSize_gamma);
    ReadFile("./input/input_beta.bin", inputSize_beta, beta, inputSize_beta);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(pre_layer_norm_custom, blockDim, x, y, gamma, beta, z, caseNum); // use this macro for cpu debug

    WriteFile("./output/output_z.bin", z, outputSize_z);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)gamma);
    AscendC::GmFree((void *)beta);
    AscendC::GmFree((void *)z);
#else
    CHECK_ACL(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t *xHost, *yHost, *zHost, *gammaHost, *betaHost;
    uint8_t *xDevice, *yDevice, *zDevice, *gammaDevice, *betaDevice;

    CHECK_ACL(aclrtMallocHost((void**)(&xHost), inputSize_x));
    CHECK_ACL(aclrtMallocHost((void**)(&yHost), inputSize_y));
    CHECK_ACL(aclrtMallocHost((void**)(&gammaHost), inputSize_gamma));
    CHECK_ACL(aclrtMallocHost((void**)(&betaHost), inputSize_beta));
    CHECK_ACL(aclrtMallocHost((void**)(&zHost), outputSize_z));
    CHECK_ACL(aclrtMalloc((void**)&xDevice, inputSize_x, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void**)&yDevice, inputSize_y, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void**)&gammaDevice, inputSize_gamma, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void**)&betaDevice, inputSize_beta, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void**)&zDevice, outputSize_z, ACL_MEM_MALLOC_HUGE_FIRST));

    ReadFile("./input/input_x.bin", inputSize_x, xHost, inputSize_x);
    ReadFile("./input/input_y.bin", inputSize_y, yHost, inputSize_y);
    ReadFile("./input/input_gamma.bin", inputSize_gamma, gammaHost, inputSize_gamma);
    ReadFile("./input/input_beta.bin", inputSize_beta, betaHost, inputSize_beta);

    CHECK_ACL(aclrtMemcpy(xDevice, inputSize_x, xHost, inputSize_x, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(yDevice, inputSize_y, yHost, inputSize_y, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(gammaDevice, inputSize_gamma, gammaHost, inputSize_gamma, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(betaDevice, inputSize_beta, betaHost, inputSize_beta, ACL_MEMCPY_HOST_TO_DEVICE));

    pre_layer_norm_custom_do(blockDim, nullptr, stream, xDevice, yDevice, gammaDevice, betaDevice, zDevice, caseNum);
    CHECK_ACL(aclrtSynchronizeStream(stream));

    CHECK_ACL(aclrtMemcpy(zHost, outputSize_z, zDevice, outputSize_z, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile("./output/output_z.bin", zHost, outputSize_z);

    CHECK_ACL(aclrtFree(xDevice));
    CHECK_ACL(aclrtFree(yDevice));
    CHECK_ACL(aclrtFree(zDevice));
    CHECK_ACL(aclrtFree(gammaDevice));
    CHECK_ACL(aclrtFree(betaDevice));
    CHECK_ACL(aclrtFreeHost(xHost));
    CHECK_ACL(aclrtFreeHost(yHost));
    CHECK_ACL(aclrtFreeHost(zHost));
    CHECK_ACL(aclrtFreeHost(gammaHost));
    CHECK_ACL(aclrtFreeHost(betaHost));

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
#endif
    return 0;
}
