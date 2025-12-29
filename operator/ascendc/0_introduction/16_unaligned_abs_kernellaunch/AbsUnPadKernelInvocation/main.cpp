#include "data_utils.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tiling/platform/platform_ascendc.h"

#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#include "aclrtlaunch_abs_unpad_custom.h"
#else
#include "tikicpulib.h"
extern "C" __global__ __aicore__ void abs_unpad_custom(GM_ADDR inputGM, GM_ADDR outputGM, GM_ADDR tiling);
#endif
extern void GenerateTiling(const std::vector<int64_t> shape, const char *socVersion, uint8_t *tilingBuf);

int32_t main(int32_t argc, char *argv[])
{
    const char *socVersion = SOC_VERSION;
    const std::vector<int64_t> shape({16, 16});
    uint32_t blockDim = 8;
    // 28674 is TOTAL_LENGTH + (BLOCKLEN_CEIL - BLOCK_ELEMENT_NUM)
    // 28672 is TOTAL_LENGTH
    // copy in borrow the next (BLOCKLEN_CEIL - BLOCK_ELEMENT_NUM) elements of srcGM
    uint32_t oriLength = 28672;
    uint32_t colNum = 14;
    uint32_t maxColNum = 32 / sizeof(uint16_t);
    uint32_t padLength = oriLength + maxColNum - colNum;
    size_t inputByteSize = padLength * sizeof(int16_t);
    size_t outputByteSize = oriLength * sizeof(int16_t);
    size_t tilingSize = sizeof(UnPadTiling);
    uint8_t *tilingBuf = (uint8_t *)malloc(tilingSize);
    GenerateTiling(shape, socVersion, tilingBuf);

#ifdef ASCENDC_CPU_DEBUG
    uint8_t *inputGM = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *outputGM = (uint8_t *)AscendC::GmAlloc(outputByteSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(sizeof(UnPadTiling));

    ReadFile("./input/input_x.bin", inputByteSize, inputGM, inputByteSize);
    memcpy_s(tiling, tilingSize, tilingBuf, tilingSize);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    ICPU_RUN_KF(abs_unpad_custom, blockDim, inputGM, outputGM, tiling); // use this macro for cpu debug

    WriteFile("./output/output_z.bin", outputGM, outputByteSize);

    AscendC::GmFree((void *)inputGM);
    AscendC::GmFree((void *)outputGM);
    AscendC::GmFree((void *)tiling);
#else
    CHECK_ACL(aclInit(nullptr));
    aclrtContext context;
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    CHECK_ACL(aclrtCreateContext(&context, deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t *xHost, *zHost, *tilingHost;
    uint8_t *xDevice, *zDevice, *tilingDevice;

    CHECK_ACL(aclrtMallocHost((void **)(&xHost), inputByteSize));
    CHECK_ACL(aclrtMallocHost((void **)(&zHost), outputByteSize));
    CHECK_ACL(aclrtMallocHost((void **)(&tilingHost), tilingSize));

    CHECK_ACL(aclrtMalloc((void **)&xDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&zDevice, outputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&tilingDevice, tilingSize, ACL_MEM_MALLOC_HUGE_FIRST));

    ReadFile("./input/input_x.bin", inputByteSize, xHost, inputByteSize);
    CHECK_ACL(aclrtMemcpy(xDevice, inputByteSize, xHost, inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(tilingDevice, tilingSize, tilingBuf, tilingSize, ACL_MEMCPY_HOST_TO_DEVICE));

    ACLRT_LAUNCH_KERNEL(abs_unpad_custom)(blockDim, stream, xDevice, zDevice, tilingDevice);
    CHECK_ACL(aclrtSynchronizeStream(stream));

    CHECK_ACL(aclrtMemcpy(zHost, outputByteSize, zDevice, outputByteSize, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile("./output/output_z.bin", zHost, outputByteSize);

    CHECK_ACL(aclrtFree(xDevice));
    CHECK_ACL(aclrtFree(zDevice));
    CHECK_ACL(aclrtFree(tilingDevice));
    CHECK_ACL(aclrtFreeHost(xHost));
    CHECK_ACL(aclrtFreeHost(zHost));
    CHECK_ACL(aclrtFreeHost(tilingHost));

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtDestroyContext(context));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
#endif
    free(tilingBuf);
    return 0;
}
