/**
 * @file main.cpp
 *
 * Copyright (C) 2024-2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

#include "data_utils.h"
#include "./op_host/tbufpool_custom_tiling.h"
#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#include "aclrtlaunch_tbufpool_custom.h"
#include "tiling/platform/platform_ascendc.h"
#else
#include "tikicpulib.h"
extern "C" __global__ __aicore__ void tbufpool_custom(GM_ADDR x, GM_ADDR y, GM_ADDR zAdd, TbufPoolTilingData tiling);
#endif

namespace {
constexpr uint32_t USED_CORE_NUM = 1;
constexpr uint32_t TOTAL_LENGTH = 2048;
constexpr uint32_t DST_LENGTH = 1024;
constexpr uint32_t TILING_SIZE = 1;
}

extern void GenerateTilingData(const uint32_t totalLength, uint8_t *tilingBuf);

static bool CompareResult(const void *outputData, int64_t outSize) {
    void *goldenData;
#ifdef ASCENDC_CPU_DEBUG
    goldenData = (uint8_t *)AscendC::GmAlloc(outSize);
#else
    CHECK_ACL(aclrtMallocHost((void **)(&goldenData), outSize));
#endif
    size_t goldenSize = outSize;
    bool ret = ReadFile("./output/golden.bin", goldenSize, goldenData, goldenSize);
    if (ret) {
        printf("ReadFile golden.bin success!\n");
    } else {
        printf("test failed!\n");
        return false;
    }
    constexpr float EPS = 1e-4;
    int64_t wrongNum = 0;

    for (int i = 0; i < outSize / sizeof(float); i++) {
        float a = (reinterpret_cast<const float *>(outputData))[i];
        float b = (reinterpret_cast<const float *>(goldenData))[i];
        float ae = std::abs(a - b);
        float re = ae / abs(b);
        if (ae > EPS && re > EPS) {
            printf(" %lf CompareResult failed output is %lf, golden is %lf\n", float(i), a, b);
            wrongNum++;
        }
    }
#ifdef ASCENDC_CPU_DEBUG
    AscendC::GmFree((void *)goldenData);
#else
    CHECK_ACL(aclrtFreeHost(goldenData));
#endif
    if (wrongNum != 0) {
        return false;
    } else {
        printf("CompareResult golden.bin success!\n");
        return true;
    }
}

int32_t main(int32_t argc, char *argv[]) {
    size_t tilingSize = TILING_SIZE * sizeof(uint32_t);
    size_t inputSize = TOTAL_LENGTH * sizeof(float);
    size_t outputSizeAdd = inputSize;

#ifdef ASCENDC_CPU_DEBUG
    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(inputSize);
    uint8_t *zAdd = (uint8_t *)AscendC::GmAlloc(outputSizeAdd);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);

    ReadFile("./input/input_x.bin", inputSize, x, inputSize);
    ReadFile("./input/input_y.bin", inputSize, y, inputSize);

    GenerateTilingData(TOTAL_LENGTH, tiling);

    AscendC::SetKernelMode(KernelMode::AIV_MODE); // run in aiv mode

    ICPU_RUN_KF(tbufpool_custom, USED_CORE_NUM, x, y, zAdd, *reinterpret_cast<TbufPoolTilingData *>(tiling)); // use this macro for cpu debug

    WriteFile("./output/output.bin", zAdd, outputSizeAdd);

    bool goldenResult = true;
    goldenResult = CompareResult(zAdd, outputSizeAdd);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)zAdd);
    AscendC::GmFree((void *)tiling);
#else
    CHECK_ACL(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t *xHost; 
    uint8_t *yHost; 
    uint8_t *zHostAdd; 
    uint8_t *tiling;
    uint8_t *xDevice; 
    uint8_t *yDevice; 
    uint8_t *zDeviceAdd;

    CHECK_ACL(aclrtMallocHost((void **)(&xHost), inputSize));
    CHECK_ACL(aclrtMallocHost((void **)(&yHost), inputSize));
    CHECK_ACL(aclrtMallocHost((void **)(&zHostAdd), outputSizeAdd));
    CHECK_ACL(aclrtMallocHost((void **)(&tiling), tilingSize));

    CHECK_ACL(aclrtMalloc((void **)&xDevice, inputSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&yDevice, inputSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&zDeviceAdd, outputSizeAdd, ACL_MEM_MALLOC_HUGE_FIRST));

    ReadFile("./input/input_x.bin", inputSize, xHost, inputSize);
    ReadFile("./input/input_y.bin", inputSize, yHost, inputSize);

    GenerateTilingData(TOTAL_LENGTH, tiling);

    // Copy host memory to device memory
    CHECK_ACL(aclrtMemcpy(xDevice, inputSize, xHost, inputSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(yDevice, inputSize, yHost, inputSize, ACL_MEMCPY_HOST_TO_DEVICE));

    // Execute the kernel
    ACLRT_LAUNCH_KERNEL(tbufpool_custom)
    (USED_CORE_NUM, stream, xDevice, yDevice, zDeviceAdd, reinterpret_cast<TbufPoolTilingData *>(tiling));

    // Wait for the stop event to complete
    CHECK_ACL(aclrtSynchronizeStream(stream));

    // Copy result to host memory and write to output file
    CHECK_ACL(aclrtMemcpy(zHostAdd, outputSizeAdd, zDeviceAdd, outputSizeAdd, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile("./output/output.bin", zHostAdd, outputSizeAdd);

    // Compare the result with the golden result
    bool goldenResult = true;
    goldenResult = CompareResult(zHostAdd, outputSizeAdd);

    // Clean up memory
    CHECK_ACL(aclrtFree(xDevice));
    CHECK_ACL(aclrtFree(yDevice));
    CHECK_ACL(aclrtFree(zDeviceAdd));

    CHECK_ACL(aclrtFreeHost(xHost));
    CHECK_ACL(aclrtFreeHost(yHost));
    CHECK_ACL(aclrtFreeHost(zHostAdd));

    CHECK_ACL(aclrtFreeHost(tiling));

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
#endif

    if (goldenResult) {
        printf("test pass!\n");
    } else {
        printf("test failed!\n");
    }
    return 0;
}
  