/**
 * @file main.cpp
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "data_utils.h"
#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
extern void add_custom_do_v1(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y, uint8_t *z);
extern void add_custom_do_v2(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y, uint8_t *z);
using KernelEntry = void(*)(uint32_t, void *, uint8_t *, uint8_t *, uint8_t *);
#else
#include "tikicpulib.h"
extern "C" __global__ __aicore__ void add_custom_v1(GM_ADDR x, GM_ADDR y, GM_ADDR z);
extern "C" __global__ __aicore__ void add_custom_v2(GM_ADDR x, GM_ADDR y, GM_ADDR z);
using KernelEntry = void(*)(GM_ADDR, GM_ADDR, GM_ADDR);

#endif

struct ArgInfo {
    std::string fileName;
    size_t length;
};

#ifndef ASCENDC_CPU_DEBUG

void KernelCall(KernelEntry kernelEntry, uint32_t blockDim, void *stream, std::vector<ArgInfo> &inputsInfo,
                std::vector<ArgInfo> &outputsInfo)
{
    std::vector<uint8_t *> inputHost(inputsInfo.size());
    std::vector<uint8_t *> inputDevice(inputsInfo.size());
    std::vector<uint8_t *> outputHost(outputsInfo.size());
    std::vector<uint8_t *> outputDevice(outputsInfo.size());

    for (uint32_t i = 0; i < inputsInfo.size(); i++) {
        CHECK_ACL(aclrtMallocHost((void **)(&inputHost[i]), inputsInfo[i].length));
        CHECK_ACL(aclrtMalloc((void **)(&inputDevice[i]), inputsInfo[i].length, ACL_MEM_MALLOC_HUGE_FIRST));
        ReadFile(inputsInfo[i].fileName, inputsInfo[i].length, inputHost[i], inputsInfo[i].length);
        CHECK_ACL(aclrtMemcpy(inputDevice[i], inputsInfo[i].length, inputHost[i], inputsInfo[i].length,
                              ACL_MEMCPY_HOST_TO_DEVICE));
    }

    for (uint32_t i = 0; i < outputsInfo.size(); i++) {
        CHECK_ACL(aclrtMallocHost((void **)(&outputHost[i]), outputsInfo[i].length));
        CHECK_ACL(aclrtMalloc((void **)(&outputDevice[i]), outputsInfo[i].length, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    kernelEntry(blockDim, stream, inputDevice[0], inputDevice[1], outputDevice[0]);
    CHECK_ACL(aclrtSynchronizeStream(stream));
    for (uint32_t i = 0; i < outputsInfo.size(); i++) {
        CHECK_ACL(aclrtMemcpy(outputHost[i], outputsInfo[i].length, outputDevice[i], outputsInfo[i].length,
                              ACL_MEMCPY_DEVICE_TO_HOST));
        WriteFile(outputsInfo[i].fileName, outputHost[i], outputsInfo[i].length);
        CHECK_ACL(aclrtFree(outputDevice[i]));
        CHECK_ACL(aclrtFreeHost(outputHost[i]));
    }

    for (uint32_t i = 0; i < inputsInfo.size(); i++) {
        CHECK_ACL(aclrtFree(inputDevice[i]));
        CHECK_ACL(aclrtFreeHost(inputHost[i]));
    }
}

#else

#define KernelCall(kernelEntry, blockDim, inputsInfo, outputsInfo)                                  \
    {                                                                                               \
        std::vector<uint8_t *> input(inputsInfo.size());                                            \
        std::vector<uint8_t *> output(outputsInfo.size());                                          \
                                                                                                    \
        for (uint32_t i = 0; i < inputsInfo.size(); i++) {                                          \
            input[i] = (uint8_t *)AscendC::GmAlloc(inputsInfo[i].length);                           \
            ReadFile(inputsInfo[i].fileName, inputsInfo[i].length, input[i], inputsInfo[i].length); \
        }                                                                                           \
                                                                                                    \
        for (uint32_t i = 0; i < outputsInfo.size(); i++) {                                         \
            output[i] = (uint8_t *)AscendC::GmAlloc(outputsInfo[i].length);                         \
        }                                                                                           \
                                                                                                    \
        AscendC::SetKernelMode(KernelMode::AIV_MODE);                                               \
        ICPU_RUN_KF(kernelEntry, blockDim, input[0], input[1], output[0]);                          \
        for (uint32_t i = 0; i < inputsInfo.size(); i++) {                                          \
            AscendC::GmFree((void *)input[i]);                                                      \
        }                                                                                           \
                                                                                                    \
        for (uint32_t i = 0; i < outputsInfo.size(); i++) {                                         \
            WriteFile(outputsInfo[i].fileName, output[i], outputsInfo[i].length);                   \
            AscendC::GmFree((void *)output[i]);                                                     \
        }                                                                                           \
    }

#endif

int32_t main(int32_t argc, char *argv[])
{
    uint32_t blockDim = 1;
    uint32_t dataLen = 4096;
    size_t inputByteSize = dataLen * sizeof(float);
    size_t outputByteSize = dataLen * sizeof(float);

    std::vector<ArgInfo> inputsInfo = {{"./input/input_x.bin", inputByteSize}, {"./input/input_y.bin", inputByteSize}};
    std::vector<ArgInfo> outputsV1Info = {{"./output/output_z_v1.bin", outputByteSize}};
    std::vector<ArgInfo> outputsV2Info = {{"./output/output_z_v2.bin", outputByteSize}};

#ifndef ASCENDC_CPU_DEBUG
    CHECK_ACL(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    KernelCall(add_custom_do_v1, blockDim, stream, inputsInfo, outputsV1Info);
    KernelCall(add_custom_do_v2, blockDim, stream, inputsInfo, outputsV2Info);

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
#else
    KernelCall(add_custom_v1, blockDim, inputsInfo, outputsV1Info);
    KernelCall(add_custom_v2, blockDim, inputsInfo, outputsV2Info);
#endif
    return 0;
}
