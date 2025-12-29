/**
 * @file main.cpp
 *
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "add_custom_tiling.h"
#include "data_utils.h"
#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#include "aclrtlaunch_add_custom.h"
#include "tiling/platform/platform_ascendc.h"
#else
#include "tikicpulib.h"
extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, AddCustomTilingData tiling);
#endif
extern void GenerateTilingData(uint8_t* tilingBuf, uint32_t blockDim);


int32_t main(int32_t argc, char *argv[])
{
    constexpr uint32_t BLOCK_DIM = 8;
    constexpr uint32_t DATA_TYPE_SIZE[] = {2, 2, 4, 1, 2, 4};
    uint8_t *tiling = nullptr;
    size_t tilingSize = 20 * sizeof(uint32_t);

#ifdef ASCENDC_CPU_DEBUG
    tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);
    ReadFile("./input/input_tiling.bin", tilingSize, tiling, tilingSize);
#else
    CHECK_ACL(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t *xHost, *yHost, *zHost;
    uint8_t *xDevice, *yDevice, *zDevice;

    CHECK_ACL(aclrtMallocHost((void **)(&tiling), tilingSize));
    ReadFile("./input/input_tiling.bin", tilingSize, tiling, tilingSize);
#endif
    GenerateTilingData(tiling, BLOCK_DIM);
    uint32_t dataTypeSize = DATA_TYPE_SIZE[reinterpret_cast<AddCustomTilingData *>(tiling)->dataType];
    uint32_t xLen = reinterpret_cast<AddCustomTilingData *>(tiling)->xLen;
    uint32_t yLen = reinterpret_cast<AddCustomTilingData *>(tiling)->yLen;
    uint32_t totalLength = (xLen > yLen)? xLen : yLen;

    size_t inputXByteSize = xLen * dataTypeSize;
    size_t inputYByteSize = yLen * dataTypeSize;
    size_t outputByteSize = totalLength * dataTypeSize;

#ifdef ASCENDC_CPU_DEBUG
    uint8_t *x = (uint8_t *)AscendC::GmAlloc(inputXByteSize);
    uint8_t *y = (uint8_t *)AscendC::GmAlloc(inputYByteSize);
    uint8_t *z = (uint8_t *)AscendC::GmAlloc(outputByteSize);

    ReadFile("./input/input_x.bin", inputXByteSize, x, inputXByteSize);
    ReadFile("./input/input_y.bin", inputYByteSize, y, inputYByteSize);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_RUN_KF(add_custom, BLOCK_DIM, x, y, z,
                *reinterpret_cast<AddCustomTilingData *>(tiling));

    WriteFile("./output/output_z.bin", z, outputByteSize);

    AscendC::GmFree((void *)x);
    AscendC::GmFree((void *)y);
    AscendC::GmFree((void *)z);
    AscendC::GmFree((void *)tiling);
#else
    CHECK_ACL(aclrtMallocHost((void **)(&xHost), inputXByteSize));
    CHECK_ACL(aclrtMallocHost((void **)(&yHost), inputYByteSize));
    CHECK_ACL(aclrtMallocHost((void **)(&zHost), outputByteSize));
    CHECK_ACL(aclrtMalloc((void **)&xDevice, inputXByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&yDevice, inputYByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&zDevice, outputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));

    ReadFile("./input/input_x.bin", inputXByteSize, xHost, inputXByteSize);
    ReadFile("./input/input_y.bin", inputYByteSize, yHost, inputYByteSize);

    CHECK_ACL(aclrtMemcpy(xDevice, inputXByteSize, xHost, inputXByteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(yDevice, inputYByteSize, yHost, inputYByteSize, ACL_MEMCPY_HOST_TO_DEVICE));

    ACLRT_LAUNCH_KERNEL(add_custom)(BLOCK_DIM, stream, xDevice, yDevice, zDevice,
        reinterpret_cast<AddCustomTilingData *>(tiling));
    CHECK_ACL(aclrtSynchronizeStream(stream));

    CHECK_ACL(aclrtMemcpy(zHost, outputByteSize, zDevice, outputByteSize, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile("./output/output_z.bin", zHost, outputByteSize);

    CHECK_ACL(aclrtFree(xDevice));
    CHECK_ACL(aclrtFree(yDevice));
    CHECK_ACL(aclrtFree(zDevice));
    CHECK_ACL(aclrtFreeHost(xHost));
    CHECK_ACL(aclrtFreeHost(yHost));
    CHECK_ACL(aclrtFreeHost(zHost));
    CHECK_ACL(aclrtFreeHost(tiling));

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
#endif
    return 0;
}
