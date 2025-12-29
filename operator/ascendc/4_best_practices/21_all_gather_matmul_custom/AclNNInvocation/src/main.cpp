/**
* @file main.cpp
*
* Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#include <cstdint>
#include <iostream>
#include <thread>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include "hccl/hccl.h"
#include "hccl/hccl_types.h"

#include "../../all_gather_matmul_demo_def.h"
#include "common.h"
#include "op_runner.h"

bool g_isDevice = false;

namespace {
OperatorDesc CreateOpDesc()
{
    // define operator
    std::vector<int64_t> shapeA { RANK_M, RANK_K };
    std::vector<int64_t> shapeB { RANK_K, RANK_N };
    std::vector<int64_t> shapeC { RANK_M * RANK_DIM, RANK_N };
    std::vector<int64_t> shapeGatherOut { RANK_M * RANK_DIM, RANK_K };
    aclDataType dataTypeA = ACL_FLOAT16;
    aclDataType dataTypeB = ACL_FLOAT16;
    aclDataType dataTypeC = ACL_FLOAT16;
    aclDataType dataTypeGatherOut = ACL_FLOAT16;
    aclFormat format = ACL_FORMAT_ND;

    OperatorDesc opDesc;
    opDesc.AddInputTensorDesc(dataTypeA, shapeA.size(), shapeA.data(), format);
    opDesc.AddInputTensorDesc(dataTypeB, shapeB.size(), shapeB.data(), format);
    opDesc.AddOutputTensorDesc(dataTypeC, shapeC.size(), shapeC.data(), format);
    opDesc.AddOutputTensorDesc(dataTypeGatherOut, shapeGatherOut.size(), shapeGatherOut.data(), format);
    return opDesc;
}

bool SetInputData(OpRunner &runner, uint32_t rankId)
{
    size_t fileSize = 0;
    ReadFile("../input/input_a_" + std::to_string(rankId) + ".bin", fileSize,
        runner.GetInputBuffer<void>(0), runner.GetInputSize(0)); // Read input_a file
    ReadFile("../input/input_b_" + std::to_string(rankId) + ".bin", fileSize,
        runner.GetInputBuffer<void>(1), runner.GetInputSize(1)); // Read input_b file
    INFO_LOG("Set input success");
    return true;
}

bool ProcessOutputData(OpRunner &runner, uint32_t rankId)
{
    WriteFile("../output/out_" + std::to_string(rankId) + ".bin", runner.GetOutputBuffer<void>(0),
        runner.GetOutputSize(0));
    WriteFile("../output/gather_out_" + std::to_string(rankId) + ".bin", runner.GetOutputBuffer<void>(1),
        runner.GetOutputSize(1));
    INFO_LOG("Write output success");
    return true;
}

bool InitResource()
{
    std::string output = "../output";
    if (access(output.c_str(), 0) == -1) {
        constexpr mode_t OUTPUT_DIR_PERMISSIONS = 0700; // 文件权限
        if (mkdir(output.c_str(), OUTPUT_DIR_PERMISSIONS) != 0) {
            ERROR_LOG("Make output directory fail");
            return false;
        }
    }

    // acl.json is dump or profiling config file
    if (aclInit(NULL) != ACL_SUCCESS) {
        ERROR_LOG("acl init failed");
        return false;
    }

    // set device
    for (int32_t i = 0; i < RANK_DIM; i++) {
        if (aclrtSetDevice(i) != ACL_SUCCESS) {
            ERROR_LOG("Set device failed. deviceId is %u", i);
            for (uint32_t j = 0; j < i; j++) {
                (void)aclrtResetDevice(j);
            }
            (void)aclFinalize();
            return false;
        }
    }
    return true;
}

bool RunOp(uint32_t rankId, HcclComm &comm)
{
    // create contest
    aclrtContext context;
    if (aclrtCreateContext(&context, rankId) != ACL_SUCCESS) {
        ERROR_LOG("Create context failed. deviceId is %u", rankId);
        (void)HcclCommDestroy(comm);
        (void)aclrtResetDevice(rankId);
        return false;
    }
    // create stream
    aclrtStream stream;
    if (aclrtCreateStream(&stream) != ACL_SUCCESS) {
        ERROR_LOG("Create stream failed. deviceId is %u", rankId);
        (void)HcclCommDestroy(comm);
        (void)aclrtDestroyContext(context);
        (void)aclrtResetDevice(rankId);
        return false;
    }

    // set context
    if (aclrtSetCurrentContext(context) != ACL_SUCCESS) {
        ERROR_LOG("Set current context failed, deviceId=%u", rankId);
        (void)HcclCommDestroy(comm);
        (void)aclrtDestroyStream(stream);
        (void)aclrtDestroyContext(context);
        (void)aclrtResetDevice(rankId);
        return false;
    }

    // Get hccl comm
    char group[128] = {0};
    if (HcclGetCommName(comm, group) != HCCL_SUCCESS) {
        ERROR_LOG("Hccl get comm name failed, deviceId=%u", rankId);
        (void)HcclCommDestroy(comm);
        (void)aclrtDestroyStream(stream);
        (void)aclrtDestroyContext(context);
        (void)aclrtResetDevice(rankId);
        return false;
    }

    // create op desc
    OperatorDesc opDesc = CreateOpDesc();

    // create Runner
    OpRunner opRunner(&opDesc);
    if (!opRunner.Init()) {
        ERROR_LOG("Init OpRunner failed, deviceId=%u", rankId);
        (void)HcclCommDestroy(comm);
        (void)aclrtDestroyStream(stream);
        (void)aclrtDestroyContext(context);
        (void)aclrtResetDevice(rankId);
        return false;
    }

    // Load inputs
    if (!SetInputData(opRunner, rankId)) {
        ERROR_LOG("Set input data failed, deviceId=%u", rankId);
        (void)HcclCommDestroy(comm);
        (void)aclrtDestroyStream(stream);
        (void)aclrtDestroyContext(context);
        (void)aclrtResetDevice(rankId);
        return false;
    }

    // Run op
    if (!opRunner.RunOp(group, stream)) {
        ERROR_LOG("Run op failed, deviceId=%u", rankId);
        (void)HcclCommDestroy(comm);
        (void)aclrtDestroyStream(stream);
        (void)aclrtDestroyContext(context);
        (void)aclrtResetDevice(rankId);
        return false;
    }

    // process output data
    if (!ProcessOutputData(opRunner, rankId)) {
        ERROR_LOG("Process output data failed, deviceId=%u", rankId);
        (void)HcclCommDestroy(comm);
        (void)aclrtDestroyStream(stream);
        (void)aclrtDestroyContext(context);
        (void)aclrtResetDevice(rankId);
        return false;
    }

    (void)HcclCommDestroy(comm);
    (void)aclrtDestroyStream(stream);
    (void)aclrtDestroyContext(context);
    (void)aclrtResetDevice(rankId);

    INFO_LOG("Run op success, deviceId=%u", rankId);
    return true;
}
}

int main(int argc, char **argv)
{
    if (!InitResource()) {
        ERROR_LOG("Init resource failed");
        return FAILED;
    }
    INFO_LOG("Init resource success");

    HcclComm comms[RANK_DIM];
    int32_t devices[RANK_DIM];
    for (int32_t i = 0; i < RANK_DIM; i++) {
        devices[i] = i;
    }
    if (HcclCommInitAll(RANK_DIM, devices, comms) != HCCL_SUCCESS) {
        ERROR_LOG("Hccl comm init failed.");
        (void)aclFinalize();
        return FAILED;
    }

    // run with multithread
    std::vector<std::unique_ptr<std::thread>> threads(RANK_DIM);
    for (uint32_t rankId = 0; rankId < RANK_DIM; rankId++) {
        threads[rankId].reset(new(std::nothrow) std::thread(&RunOp, rankId, std::ref(comms[rankId])));
    }
    for (uint32_t rankId = 0; rankId < RANK_DIM; rankId++) {
        threads[rankId]->join();
    }

    (void)aclFinalize();
    return SUCCESS;
}
