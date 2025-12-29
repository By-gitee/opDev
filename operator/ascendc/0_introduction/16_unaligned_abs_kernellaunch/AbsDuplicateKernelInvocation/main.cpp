#include "data_utils.h"
#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#include "aclrtlaunch_abs_duplicate_custom.h"
#else
#include "tikicpulib.h"

extern "C" __global__ __aicore__ void abs_duplicate_custom(GM_ADDR inputGM, GM_ADDR outputGM, GM_ADDR syncGM);
#endif
int32_t main(int32_t argc, char *argv[])
{
    uint32_t blockDim = 4;
    // 709 is TOTAL_LENGTH + (BLOCKLEN_CEIL - BLOCK_ELEMENT_NUM)
    // copy in borrow the next (BLOCKLEN_CEIL - BLOCK_ELEMENT_NUM) elements of srcGM
    size_t inputByteSize = 709 * sizeof(int16_t);
    // copy out atomic add extra (BLOCKLEN_CEIL - BLOCK_ELEMENT_NUM) zeros to dstGM
    size_t outputByteSize = 709 * sizeof(int16_t);
    uint32_t defaultSyncByteSizeSinleCore = 32; // 32B for SyncAll
    size_t syncByteSize = blockDim * defaultSyncByteSizeSinleCore;

#ifdef ASCENDC_CPU_DEBUG
    uint8_t *inputGM = (uint8_t *)AscendC::GmAlloc(inputByteSize);
    uint8_t *outputGM = (uint8_t *)AscendC::GmAlloc(outputByteSize);
    uint8_t *syncGM = (uint8_t *)AscendC::GmAlloc(syncByteSize);
    ReadFile("./input/input_x.bin", inputByteSize, inputGM, inputByteSize);
    ReadFile("./input/sync.bin", syncByteSize, syncGM, syncByteSize);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(abs_duplicate_custom, blockDim, inputGM, outputGM, syncGM); // use this macro for cpu debug
    WriteFile("./output/output_z.bin", outputGM, outputByteSize);
    AscendC::GmFree((void *)inputGM);
    AscendC::GmFree((void *)outputGM);
    AscendC::GmFree((void *)syncGM);
#else
    CHECK_ACL(aclInit(nullptr));
    aclrtContext context;
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    CHECK_ACL(aclrtCreateContext(&context, deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));

    uint8_t *xHost, *syncHost, *zHost;
    uint8_t *xDevice, *syncDevice, *zDevice;

    CHECK_ACL(aclrtMallocHost((void **)(&xHost), inputByteSize));
    CHECK_ACL(aclrtMallocHost((void **)(&zHost), outputByteSize));
    CHECK_ACL(aclrtMallocHost((void **)(&syncHost), syncByteSize));
    CHECK_ACL(aclrtMalloc((void **)&xDevice, inputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&zDevice, outputByteSize, ACL_MEM_MALLOC_HUGE_FIRST));
    CHECK_ACL(aclrtMalloc((void **)&syncDevice, syncByteSize, ACL_MEM_MALLOC_HUGE_FIRST));

    ReadFile("./input/input_x.bin", inputByteSize, xHost, inputByteSize);
    ReadFile("./input/sync.bin", syncByteSize, syncHost, syncByteSize);

    CHECK_ACL(aclrtMemcpy(xDevice, inputByteSize, xHost, inputByteSize, ACL_MEMCPY_HOST_TO_DEVICE));
    CHECK_ACL(aclrtMemcpy(syncDevice, syncByteSize, syncHost, syncByteSize, ACL_MEMCPY_HOST_TO_DEVICE));

    ACLRT_LAUNCH_KERNEL(abs_duplicate_custom)(blockDim, stream, xDevice, zDevice, syncDevice);
    CHECK_ACL(aclrtSynchronizeStream(stream));

    CHECK_ACL(aclrtMemcpy(zHost, outputByteSize, zDevice, outputByteSize, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile("./output/output_z.bin", zHost, outputByteSize);

    CHECK_ACL(aclrtFree(xDevice));
    CHECK_ACL(aclrtFree(zDevice));
    CHECK_ACL(aclrtFree(syncDevice));
    CHECK_ACL(aclrtFreeHost(xHost));
    CHECK_ACL(aclrtFreeHost(zHost));
    CHECK_ACL(aclrtFreeHost(syncHost));

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtDestroyContext(context));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
#endif
    return 0;
}
