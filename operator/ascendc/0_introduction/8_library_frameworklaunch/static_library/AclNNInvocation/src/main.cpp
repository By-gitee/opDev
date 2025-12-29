/**
 * @file main.cpp
 *
 * Copyright (C) 2025-2025. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstdint>
#include <iostream>

#include "acl/acl.h"
#include "common.h"
#include "op_runner.h"

#define STATIC_FRAME_TWO 2
bool g_isDevice = false;
int deviceId = 0;

OperatorDesc CreateOpDescAdd()
{
    // define operator
    std::vector<int64_t> shape{8, 2048};
    aclDataType dataType = ACL_FLOAT16;
    aclFormat format = ACL_FORMAT_ND;
    OperatorDesc opDesc;
    opDesc.AddInputTensorDesc(dataType, shape.size(), shape.data(), format);
    opDesc.AddInputTensorDesc(dataType, shape.size(), shape.data(), format);
    opDesc.AddOutputTensorDesc(dataType, shape.size(), shape.data(), format);
    return opDesc;
}

bool SetInputDataAdd(OpRunner &runner)
{
    size_t fileSize = 0;
    ReadFile("../input/input_x.bin", fileSize, runner.GetInputBuffer<void>(0), runner.GetInputSize(0));
    ReadFile("../input/input_y.bin", fileSize, runner.GetInputBuffer<void>(1), runner.GetInputSize(1));
    INFO_LOG("Set input success");
    return true;
}

OperatorDesc CreateOpDescMatmul()
{
    // define operator
    std::vector<int64_t> shapeA{1024, 256};
    std::vector<int64_t> shapeB{256, 640};
    std::vector<int64_t> shapeBias{640};
    std::vector<int64_t> shapeC{1024, 640};
    aclDataType dataTypeA = ACL_FLOAT16;
    aclDataType dataTypeB = ACL_FLOAT16;
    aclDataType dataTypeBias = ACL_FLOAT;
    aclDataType dataTypeC = ACL_FLOAT;
    aclFormat format = ACL_FORMAT_ND;
    OperatorDesc opDesc;
    opDesc.AddInputTensorDesc(dataTypeA, shapeA.size(), shapeA.data(), format);
    opDesc.AddInputTensorDesc(dataTypeB, shapeB.size(), shapeB.data(), format);
    opDesc.AddInputTensorDesc(dataTypeBias, shapeBias.size(), shapeBias.data(), format);
    opDesc.AddOutputTensorDesc(dataTypeC, shapeC.size(), shapeC.data(), format);
    return opDesc;
}

bool SetInputDataMatmul(OpRunner &runner)
{
    size_t fileSize = 0;
    ReadFile("../input/input_a.bin", fileSize, runner.GetInputBuffer<void>(0), runner.GetInputSize(0));
    ReadFile("../input/input_b.bin", fileSize, runner.GetInputBuffer<void>(1), runner.GetInputSize(1));
    ReadFile("../input/input_bias.bin", fileSize, runner.GetInputBuffer<void>(STATIC_FRAME_TWO),
        runner.GetInputSize(STATIC_FRAME_TWO));
    INFO_LOG("Set input success");
    return true;
}

bool ProcessOutputData(OpRunner &runner, std::string opName)
{
    std::string filePath = "../output/output_z_" + opName + ".bin";
    WriteFile(filePath, runner.GetOutputBuffer<void>(0), runner.GetOutputSize(0));
    INFO_LOG("Write output success");
    return true;
}

void DestroyResource()
{
    bool flag = false;
    if (aclrtResetDevice(deviceId) != ACL_SUCCESS) {
        ERROR_LOG("Reset device %d failed", deviceId);
        flag = true;
    }
    INFO_LOG("Reset Device success");
    if (aclFinalize() != ACL_SUCCESS) {
        ERROR_LOG("Finalize acl failed");
        flag = true;
    }
    if (flag) {
        ERROR_LOG("Destroy resource failed");
    } else {
        INFO_LOG("Destroy resource success");
    }
}

bool InitResource()
{
    std::string output = "../output";
    if (access(output.c_str(), 0) == -1) {
        int ret = mkdir(output.c_str(), 0700);
        if (ret == 0) {
            INFO_LOG("Make output directory successfully");
        } else {
            ERROR_LOG("Make output directory fail");
            return false;
        }
    }

    if (aclInit(nullptr) != ACL_SUCCESS) {
        ERROR_LOG("acl init failed");
        return false;
    }

    if (aclrtSetDevice(deviceId) != ACL_SUCCESS) {
        ERROR_LOG("Set device failed. deviceId is %d", deviceId);
        (void)aclFinalize();
        return false;
    }
    INFO_LOG("Set device[%d] success", deviceId);

    // runMode is ACL_HOST which represents app is running in host
    // runMode is ACL_DEVICE which represents app is running in device
    aclrtRunMode runMode;
    if (aclrtGetRunMode(&runMode) != ACL_SUCCESS) {
        ERROR_LOG("Get run mode failed");
        DestroyResource();
        return false;
    }
    g_isDevice = (runMode == ACL_DEVICE);
    INFO_LOG("Get RunMode[%d] success", runMode);

    return true;
}

bool RunOpMatmul()
{
    OperatorDesc opDesc = CreateOpDescMatmul();

    OpRunner opRunner(&opDesc);
    if (!opRunner.Init()) {
        ERROR_LOG("Init matmul_custom OpRunner failed");
        return false;
    }

    if (!SetInputDataMatmul(opRunner)) {
        ERROR_LOG("Set matmul_custom input data failed");
        return false;
    }

    if (!opRunner.RunOpMatmul()) {
        ERROR_LOG("Run matmul_custom op failed");
        return false;
    }

    if (!ProcessOutputData(opRunner, "matmul")) {
        ERROR_LOG("Process matmul_custom output data failed");
        return false;
    }

    INFO_LOG("Run matmul_custom op success");
    return true;
}

bool RunOpAdd()
{
    OperatorDesc opDesc = CreateOpDescAdd();

    OpRunner opRunner(&opDesc);
    if (!opRunner.Init()) {
        ERROR_LOG("Init add_custom OpRunner failed");
        return false;
    }

    if (!SetInputDataAdd(opRunner)) {
        ERROR_LOG("Set add_custom input data failed");
        return false;
    }

    if (!opRunner.RunOpAdd()) {
        ERROR_LOG("Run add_custom op failed");
        return false;
    }

    if (!ProcessOutputData(opRunner, "add")) {
        ERROR_LOG("Process add_custom output data failed");
        return false;
    }

    INFO_LOG("Run add_custom op success");
    return true;
}

int main(int argc, char **argv)
{
    if (!InitResource()) {
        ERROR_LOG("Init resource failed");
        return FAILED;
    }
    INFO_LOG("Init resource success");

    if (!RunOpMatmul()) {
        DestroyResource();
        return FAILED;
    }
    if (!RunOpAdd()) {
        DestroyResource();
        return FAILED;
    }

    DestroyResource();

    return SUCCESS;
}
