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
#include "quant_group_matmul_custom_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#ifndef ASCENDC_CPU_DEBUG
#include "acl/acl.h"
#include "aclrtlaunch_quant_group_matmul_custom.h"
#else
#include "tikicpulib.h"
extern "C" void quant_group_matmul_custom(uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *,
                                          uint8_t *, uint8_t *, uint8_t *);
#endif
extern bool GenerateTiling(QuantGroupMatmulCustomTilingData &gmmTiling);

struct Data {
    size_t size = 0;
    bool hasFile = true;
    uint8_t *host = nullptr;
    uint8_t *device = nullptr;

    explicit Data(size_t size_, bool hasFile_ = true) : size(size_), hasFile(hasFile_) {}
};

struct OpArgs {
    Data x;
    Data weight;
    Data bias;
    Data groupList;
    Data scale;
    Data perTokenScale;
    Data tiling;
    Data workspace;
    Data y;
};

bool AllocMem(const std::string &name, Data &data)
{
    if (data.size == 0) {
        return true;
    }
    std::string file = std::string("./input/" + name + ".bin");
#ifdef ASCENDC_CPU_DEBUG
    data.host = (uint8_t *)AscendC::GmAlloc(data.size);
    if (data.hasFile) {
        ReadFile(file, data.size, data.host, data.size);
    }
#else
    CHECK_ACL(aclrtMallocHost((void **)(&data.host), data.size));
    CHECK_ACL(aclrtMalloc((void **)&data.device, data.size, ACL_MEM_MALLOC_HUGE_FIRST));
    if (data.hasFile) {
        ReadFile(file, data.size, data.host, data.size);
        CHECK_ACL(aclrtMemcpy(data.device, data.size, data.host, data.size, ACL_MEMCPY_HOST_TO_DEVICE));
    }
#endif
    return true;
}

bool CreateInput(OpArgs &args) {
    AllocMem("x", args.x);
    AllocMem("weight", args.weight);
    AllocMem("bias", args.bias);
    AllocMem("groupList", args.groupList);
    AllocMem("scale", args.scale);
    AllocMem("perTokenScale", args.perTokenScale);
    AllocMem("tiling", args.tiling);
    AllocMem("workspace", args.workspace);
    AllocMem("y", args.y);
    return true;
}

bool FreeMem(Data &data)
{
#ifdef ASCENDC_CPU_DEBUG
    if (data.host != nullptr) {
        AscendC::GmFree((void *)data.host);
    }
#else
    if (data.device != nullptr) {
        CHECK_ACL(aclrtFree(data.device));
    }
    if (data.host != nullptr) {
        CHECK_ACL(aclrtFreeHost(data.host));
    }
#endif
    return true;
}

bool FreeData(OpArgs &args)
{
    FreeMem(args.x);
    FreeMem(args.weight);
    FreeMem(args.bias);
    FreeMem(args.groupList);
    FreeMem(args.scale);
    FreeMem(args.perTokenScale);
    FreeMem(args.tiling);
    FreeMem(args.workspace);
    FreeMem(args.y);
    return true;
}


int32_t main(int32_t argc, char *argv[])
{
    int m = 1024;
    int k = 1024;
    int n = 8192;
    int groupNum = 8;
    uint32_t blockDim = 8;
    uint32_t cvParallNum = 4;
    size_t userWorkspaceSize = cvParallNum * 256 * 128 * sizeof(int32_t) * blockDim;
    auto ascendcPlatform = platform_ascendc::PlatformAscendCManager::GetInstance();
    size_t systemWorkspaceSize = static_cast<size_t>(ascendcPlatform->GetLibApiWorkSpaceSize());
    size_t workspaceSize = userWorkspaceSize + systemWorkspaceSize;
    OpArgs args {
        .x=Data(m * k * sizeof(int8_t)),
        .weight=Data(groupNum * k * n * sizeof(int8_t)),
        .bias=Data(0),  // no bias
        .groupList=Data(groupNum * sizeof(int64_t)),
        .scale=Data(groupNum * n * sizeof(float)),
        .perTokenScale=Data(m * sizeof(float)),
        .tiling=Data(sizeof(QuantGroupMatmulCustomTilingData), false),
        .workspace=Data(workspaceSize, false),
        .y=Data(m * n * sizeof(int16_t), false)  // sizeof(half)
    };
#ifndef ASCENDC_CPU_DEBUG
    CHECK_ACL(aclInit(nullptr));
    int32_t deviceId = 0;
    CHECK_ACL(aclrtSetDevice(deviceId));
    aclrtStream stream = nullptr;
    CHECK_ACL(aclrtCreateStream(&stream));
#endif
    CreateInput(args);
    QuantGroupMatmulCustomTilingData &gmmTiling = *reinterpret_cast<QuantGroupMatmulCustomTilingData*>(args.tiling.host);
    gmmTiling.coreNum = blockDim;
    gmmTiling.groupNum = groupNum;
    gmmTiling.totalInGroup = m;
    gmmTiling.k = k;
    gmmTiling.n = n;
    gmmTiling.ubCalSize = 24 * 256;  // 24: vector每次计算的行数，256: 每次计算的列数，与cube baseN保持一致
    gmmTiling.ubRestBytes = 118784u;  // 118784: 除分配给TQue外剩余给TBuf的大小为118784
    gmmTiling.parallNum = cvParallNum;

    GenerateTiling(gmmTiling);

#ifdef ASCENDC_CPU_DEBUG
    // AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(quant_group_matmul_custom, blockDim, args.x.host, args.weight.host, args.bias.host,
                args.groupList.host, args.scale.host, args.perTokenScale.host, args.y.host, args.workspace.host,
                args.tiling.host);

    WriteFile("./output/output.bin", args.y.host, args.y.size);
    FreeData(args);
#else
    CHECK_ACL(aclrtMemcpy(args.tiling.device, args.tiling.size, args.tiling.host, args.tiling.size,
                          ACL_MEMCPY_HOST_TO_DEVICE));

    ACLRT_LAUNCH_KERNEL(quant_group_matmul_custom)
    (blockDim, stream, args.x.device, args.weight.device, args.bias.device, args.groupList.device, args.scale.device,
     args.perTokenScale.device, args.y.device, args.workspace.device, args.tiling.device);

    CHECK_ACL(aclrtSynchronizeStream(stream));

    CHECK_ACL(aclrtMemcpy(args.y.host, args.y.size, args.y.device, args.y.size, ACL_MEMCPY_DEVICE_TO_HOST));
    WriteFile("./output/output.bin", args.y.host, args.y.size);
    FreeData(args);

    CHECK_ACL(aclrtDestroyStream(stream));
    CHECK_ACL(aclrtResetDevice(deviceId));
    CHECK_ACL(aclFinalize());
#endif
    return 0;
}