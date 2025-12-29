/**
 * @file main.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "data_utils.h"
#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#include "aclrtlaunch_abs_gather_mask_custom.h"
#else
#include "tikicpulib.h"

extern "C" __global__ __aicore__ void abs_gather_mask_custom(GM_ADDR inputGM, GM_ADDR outputGM);
#endif
int32_t main(int32_t argc, char *argv[])
{
    uint32_t blockDim = 8;
    size_t inputByteSize = 2318 * sizeof(int16_t); // 2318 = 2304 + 32 - 18
    size_t outputByteSize = 2304 * sizeof(int16_t);

#ifdef ASCENDC_CPU_DEBUG
    uint8_t *inputGM = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *outputGM = (uint8_t *)AscendC::GmAlloc(outputByteSize);
    ReadFile("./input/input_x.bin", inputByteSize, inputGM, inputByteSize);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(abs_gather_mask_custom, blockDim, inputGM, outputGM); // use this macro for cpu debug
    WriteFile("./output/output_z.bin", outputGM, outputByteSize);
    AscendC::GmFree((void *)inputGM);
    AscendC::GmFree((void *)outputGM);
#else
    CHECK_ACL(aclInit(nullptr));
    aclrtContext context;
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    CHECK_ACL(aclrtCreateContext(&context, deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t *xHost, *zHost;
    uint8_t *xDevice, *zDevice;

    CHECK_ACL(aclrtMallocHost((void **)(&xHost), inputByteSize));
    CHECK_ACL(aclrtMallocHost((void **)(&zHost), outputByteSize));
    CHECK_ACL(aclrtMalloc((void **)&xDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&zDevice, outputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));

    ReadFile("./input/input_x.bin", inputByteSize, xHost, inputByteSize);

    CHECK_ACL(aclrtMemcpy(xDevice, inputByteSize, xHost, inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE));

    ACLRT_LAUNCH_KERNEL(abs_gather_mask_custom)(blockDim, stream, xDevice, zDevice);
    CHECK_ACL(aclrtSynchronizeStream(stream));

    CHECK_ACL(aclrtMemcpy(zHost, outputByteSize, zDevice, outputByteSize, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile("./output/output_z.bin", zHost, outputByteSize);

    CHECK_ACL(aclrtFree(xDevice));
    CHECK_ACL(aclrtFree(zDevice));
    CHECK_ACL(aclrtFreeHost(xHost));
    CHECK_ACL(aclrtFreeHost(zHost));

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtDestroyContext(context));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
#endif
    return 0;
}
