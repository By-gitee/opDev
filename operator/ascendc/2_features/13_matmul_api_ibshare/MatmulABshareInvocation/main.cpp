/**
 * @file main.cpp
 *
 * Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "data_utils.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include <iostream>
#include <chrono>
#include "acl/acl.h"
#include "aclrtlaunch_matmul_ABshare_custom.h"
#include "aclrtlaunch_matmul_noABshare_custom.h"
extern uint8_t *GenerateTiling(const char *socVersion);
constexpr uint16_t CYCLENUMS = 10000;
constexpr float PRECENT = 100;

int32_t main(int32_t argc, char *argv[])
{
    const char *socVersion = SOC_VERSION;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance(socVersion);
    size_t aFileSize = 49152 * sizeof(int16_t);
    size_t bFileSize = 98304 * sizeof(int16_t);
    size_t cFileSize = 32768 * sizeof(float);
    size_t tilingFileSize = sizeof(TCubeTiling);
    size_t userWorkspaceSize = 0;
    size_t systemWorkspaceSize = static_cast<size_t>(ascendcPlatform->GetLibApiWorkSpaceSize());
    size_t workspaceSize = userWorkspaceSize + systemWorkspaceSize;
    uint32_t blockDim = 1;

    CHECK_ACL(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t *inputAHost;
    uint8_t *inputADevice;
    CHECK_ACL(aclrtMallocHost((void **)(&inputAHost), aFileSize));
    CHECK_ACL(aclrtMalloc((void **)&inputADevice, aFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ReadFile("./input/x1_gm.bin", aFileSize, inputAHost, aFileSize);
    CHECK_ACL(aclrtMemcpy(inputADevice, aFileSize, inputAHost, aFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *inputBHost;
    uint8_t *inputBDevice;
    CHECK_ACL(aclrtMallocHost((void **)(&inputBHost), bFileSize));
    CHECK_ACL(aclrtMalloc((void **)&inputBDevice, bFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    ReadFile("./input/x2_gm.bin", bFileSize, inputBHost, bFileSize);
    CHECK_ACL(aclrtMemcpy(inputBDevice, bFileSize, inputBHost, bFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *outputCHost;
    uint8_t *outputCDevice;
    CHECK_ACL(aclrtMallocHost((void **)(&outputCHost), cFileSize));
    CHECK_ACL(aclrtMalloc((void **)&outputCDevice, cFileSize, ACL_MEM_MALLOC_HUGE_FIRST));

    uint8_t *tilingHost;
    uint8_t *tilingDevice;
    CHECK_ACL(aclrtMallocHost((void **)(&tilingHost), tilingFileSize));
    CHECK_ACL(aclrtMalloc((void **)&tilingDevice, tilingFileSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(
        aclrtMemcpy(tilingHost, tilingFileSize, GenerateTiling(socVersion), tilingFileSize, ACL_MEMCPY_HOST_TO_HOST));
    CHECK_ACL(aclrtMemcpy(tilingDevice, tilingFileSize, tilingHost, tilingFileSize, ACL_MEMCPY_HOST_TO_DEVICE));

    uint8_t *workspaceDevice;
    CHECK_ACL(aclrtMalloc((void **)&workspaceDevice, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST));

    // Execute Matmul_noABshare_custom
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < CYCLENUMS; ++i)
    {
        ACLRT_LAUNCH_KERNEL(matmul_noABshare_custom)
        (blockDim, stream, inputADevice, inputBDevice, outputCDevice, workspaceDevice, tilingDevice);
    };
    CHECK_ACL(aclrtSynchronizeStream(stream));
    auto end = std::chrono::steady_clock::now();
    auto diff_noABshare_custom = ReportTime(start, end);
    INFO_LOG("Matmul_noABshare_custom Execute time [%f]", diff_noABshare_custom);
    CHECK_ACL(aclrtMemcpy(outputCHost, cFileSize, outputCDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile("./output/output_noABshare.bin", outputCHost, cFileSize);

    // Execute Matmul_ABshare_custom
    start = std::chrono::steady_clock::now();
    for (int i = 0; i < CYCLENUMS; ++i)
    {
        ACLRT_LAUNCH_KERNEL(matmul_ABshare_custom)
        (blockDim, stream, inputADevice, inputBDevice, outputCDevice, workspaceDevice, tilingDevice);
    };
    CHECK_ACL(aclrtSynchronizeStream(stream));
    end = std::chrono::steady_clock::now();
    auto diff_ABshare_custom = ReportTime(start, end);
    INFO_LOG("Matmul_ABshare_custom Execute time [%f]", diff_ABshare_custom);
    CHECK_ACL(aclrtMemcpy(outputCHost, cFileSize, outputCDevice, cFileSize, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile("./output/output_ABshare.bin", outputCHost, cFileSize);
    
    float efficiencyImprovePrecent = double(diff_noABshare_custom - diff_ABshare_custom) / (double)diff_noABshare_custom * PRECENT;
    INFO_LOG("Efficiency Improve Precent [%.2f %%]", efficiencyImprovePrecent);
    CHECK_ACL(aclrtFree(inputADevice));
    CHECK_ACL(aclrtFreeHost(inputAHost));
    CHECK_ACL(aclrtFree(inputBDevice));
    CHECK_ACL(aclrtFreeHost(inputBHost));
    CHECK_ACL(aclrtFree(outputCDevice));
    CHECK_ACL(aclrtFreeHost(outputCHost));
    CHECK_ACL(aclrtFree(tilingDevice));
    CHECK_ACL(aclrtFreeHost(tilingHost));
    CHECK_ACL(aclrtFree(workspaceDevice));
    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
    return 0;
}