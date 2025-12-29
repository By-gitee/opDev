/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <string>
#include <vector>
#include <fstream>
#include "unistd.h"

#include "utils.h"
#include "llm_config.h"

#include "graph/graph.h"
#include "graph/types.h"
#include "graph/tensor.h"
#include "ge/ge_error_codes.h"
#include "ge/ge_api_types.h"
#include "ge/ge_api.h"
#include "exe_graph/runtime/tensor_data.h"
#include "all_ops.h"
#include "acl/acl_rt.h"
#include "acl/acl.h"
using namespace llm;
namespace {
const uint32_t kDeviceId = 1;
const uint32_t kFullGraphPath = 2;
const uint32_t kIncGraphPath = 3;
const uint32_t kExecRankId = 4;
const uint32_t kConfigFilePath = 5;
const uint32_t kParamNumber = 6;

struct MainStartParma {
    std::string fullGraphPath;
    std::string incGraphPath;
    int32_t deviceId = 0;
    int32_t execRankId = 0;
    std::string configFilePath;
};

// 构造模型的输入tensor，并将bin文件中的数据设置到tensor的data中
bool ConstructUserInputTensor(const InputDataDesc &inputDataDesc, std::vector<ge::Tensor> &inputs) {
    if ((inputDataDesc.dataFiles.size() != inputDataDesc.shapes.size()) ||
        (inputDataDesc.shapes.size() != inputDataDesc.dataTypes.size())) {
        ERROR_LOG("Input desc shape, datatype, datafile size not equal");
        return false;
    }
    if (inputDataDesc.shapes.empty()) {
        ERROR_LOG("Input desc shape is empty");
        return false;
    }
    for (size_t i = 0; i < inputDataDesc.shapes.size(); ++i) {
        ge::TensorDesc desc(ge::Shape(inputDataDesc.shapes[i]), ge::FORMAT_ND, inputDataDesc.dataTypes[i]);
        ge::Tensor tensor;
        int64_t dataSize = CommonUtils::CalcTensorMemSize(inputDataDesc.shapes[i], inputDataDesc.dataTypes[i]);
        if (dataSize <= 0) {
            ERROR_LOG("Calc tensor mem size failed");
            return false;
        }
        uint8_t *pData = new(std::nothrow) uint8_t[dataSize];
        if (pData == nullptr) {
            ERROR_LOG("Malloc data buffer failed");
            return false;
        }
        if (!CommonUtils::ReadBinFile(inputDataDesc.dataFiles[i], dataSize, pData)) {
            ERROR_LOG("Read bin file failed");
            delete[] pData;
            return false;
        }

        auto deleter = [](uint8_t *pData){ delete[] pData; };
        tensor.SetData(pData, dataSize, deleter);
        tensor.SetTensorDesc(desc);
        inputs.emplace_back(tensor);
    }
    return true;
}

// 构造kv的输入tensor，此处直接申请device内存进行构造，使增量图和全量图同一个kv tensor使用同一块内存
bool ConstructKvInputs(const KvDataDesc &kvRefDataDesc, const int32_t deviceId, std::vector<ge::Tensor> &kvInputs) {
    if (kvRefDataDesc.shape.empty()) {
        ERROR_LOG("Kv data desc shape is empty");
        return false;
    }
    auto ret = aclInit(nullptr);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("aclInit failed, ret = %d", ret);
        return false;
    }
    ret = aclrtSetDevice(deviceId);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("aclrtSetDevice failed, ret = %d", ret);
        return false;
    }
    int64_t dataSize = CommonUtils::CalcTensorMemSize(kvRefDataDesc.shape, kvRefDataDesc.dataType);
    if (dataSize < 0) {
        ERROR_LOG("Calc tensor mem size failed");
        return false;
    }
    for (uint32_t i = 0; i < kvRefDataDesc.kvNum; ++i) {
        ge::TensorDesc desc(ge::Shape(kvRefDataDesc.shape), ge::FORMAT_ND, kvRefDataDesc.dataType);
        void *pdata = nullptr;
        ret = aclrtMalloc(&pdata, dataSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_ERROR_NONE) {
            ERROR_LOG("aclrtMalloc failed, ret = %d", ret);
            return false;
        }
        ret = aclrtMemset(pdata, dataSize, 0, dataSize);
        if (ret != ACL_ERROR_NONE) {
            ERROR_LOG("aclrtMemSet failed, ret = %d", ret);
            aclrtFree(pdata);
            return false;
        }
        auto deleter = [](void *devPtr){ aclrtFree(devPtr); };
        ge::Tensor tensor;
        tensor.SetTensorDesc(desc);
        tensor.SetData(static_cast<uint8_t *>(pdata), dataSize, deleter);
        tensor.SetPlacement(ge::Placement::kPlacementDevice);
        kvInputs.emplace_back(tensor);
    }
    return true;
}

// 保存模型的第一个输出，torch air切分后air图的第一个输出为logits
bool SaveFirstOutput(ge::Tensor &firstOutputTensor, const std::string &outputFileDir) {
    int32_t pid = static_cast<int32_t>(getpid());
    std::string outputName = outputFileDir + "/" + std::to_string(pid) + "_0.bin";
    std::ofstream output(outputName, std::ios::binary);
    if (!output) {
        ERROR_LOG("Open file[%s] failed.", outputName.c_str());
        return false;
    }
    output.write(reinterpret_cast<char *>(firstOutputTensor.GetData()), firstOutputTensor.GetSize());
    output.close();
    return true;
}

// 执行全量图
bool ExecuteFullGraph(const std::vector<ge::Tensor> &kvInputs, const std::string &fullGraphPath,
                      const std::shared_ptr<ge::Session> &session) {
    std::vector<ge::Tensor> inputs;
    std::vector<ge::Tensor> outputs;
    if (!ConstructUserInputTensor(LlmConfig::Instance().GetFullGraphInputDesc(), inputs)) {
        ERROR_LOG("Construct full model user input tensor failed.");
        return false;
    }
    ge::Graph graph;
    auto geRet = graph.LoadFromFile(fullGraphPath.c_str());
    if (geRet != ge::SUCCESS) {
        ERROR_LOG("Load graph from file failed, ret[%d] filePath[%s].", geRet, fullGraphPath.c_str());
        return false;
    }
    std::map<ge::AscendString, ge::AscendString> config;
    if (LlmConfig::Instance().CacheEnable()) {
        std::string fullGraphKey = LlmConfig::Instance().GetGraphKey() + "_full";
        config["ge.graph_key"] = fullGraphKey.c_str();
    }
    geRet = session->AddGraph(0, graph, config);
    if (geRet !=  ge::SUCCESS) {
        ERROR_LOG("Add graph to session failed, ret[%d].", geRet);
        return false;
    }
    inputs.insert(inputs.cend(), kvInputs.cbegin(), kvInputs.cend());
    for (uint32_t i = 0; i < LlmConfig::Instance().GetIncGraphLoopNum(); ++i) {
        geRet = session->RunGraph(0, inputs, outputs);
        if (geRet != ge::SUCCESS) {
            ERROR_LOG("Run graph failed, ret[%d].", geRet);
            return false;
        }
    }
    if (!outputs.empty()) {
        return SaveFirstOutput(outputs[0],  "./full_output/");
    }
    return true;
}

// 执行增量图
bool ExecuteIncGraph(const std::vector<ge::Tensor> &kvInputs, const std::string &incGraphPath,
                      const std::shared_ptr<ge::Session> &session) {
    std::vector<ge::Tensor> inputs;
    std::vector<ge::Tensor> outputs;
    if (!ConstructUserInputTensor(LlmConfig::Instance().GetIncGraphInputDesc(), inputs)) {
        ERROR_LOG("Construct increment model user input tensor failed.");
        return false;
    }
    ge::Graph graph;
    auto geRet = graph.LoadFromFile(incGraphPath.c_str());
    if (geRet != ge::SUCCESS) {
        ERROR_LOG("Load graph from file failed, ret[%d] filePath[%s].", geRet, incGraphPath.c_str());
        return false;
    }
    std::map<ge::AscendString, ge::AscendString> config;
    if (LlmConfig::Instance().CacheEnable()) {
        std::string incGraphKey = LlmConfig::Instance().GetGraphKey() + "_inc";
        config["ge.graph_key"] = incGraphKey.c_str();
    }
    geRet = session->AddGraph(1, graph, config);
    if (geRet != ge::SUCCESS) {
        ERROR_LOG("Add graph to session failed, ret[%d].", geRet);
        return false;
    }
    inputs.insert(inputs.cend(), kvInputs.cbegin(), kvInputs.cend());
    for (uint32_t i = 0; i < LlmConfig::Instance().GetIncGraphLoopNum(); ++i) {
        geRet = session->RunGraph(1, inputs, outputs);
        if (geRet != ge::SUCCESS) {
            ERROR_LOG("Run graph failed, ret[%d].", geRet);
            return false;
        }
    }
    if (!outputs.empty()) {
        return SaveFirstOutput(outputs[0],  "./inc_output/");
    }
    return true;
}

// 解析输入参数
bool ParseArgs(int32_t argc, char *argv[], MainStartParma &startParm) {
    if (argc != kParamNumber) {
        ERROR_LOG("Param number not equal");
        ERROR_LOG("Five parameters as following by order: "
                  "deviceId, fullGraphPath, incGraphPath, execRankId and configFilePath.");
        return false;
    }
    startParm.fullGraphPath = CommonUtils::GetRealPath(argv[kFullGraphPath]);
    startParm.incGraphPath = CommonUtils::GetRealPath(argv[kIncGraphPath]);
    startParm.configFilePath = CommonUtils::GetRealPath(argv[kConfigFilePath]);
    if (startParm.fullGraphPath.empty() || startParm.incGraphPath.empty() || startParm.configFilePath.empty()) {
        ERROR_LOG("Full model path[%s] or inc model path[%s] or config file path[%s] is empty.",
                  startParm.fullGraphPath.c_str(), startParm.incGraphPath.c_str(), startParm.configFilePath.c_str());
        return false;
    }
    if (!CommonUtils::ConvertStrToInt32(argv[kDeviceId], startParm.deviceId)) {
        ERROR_LOG("Parse deviceId[%s] failed.", argv[kDeviceId]);
        return false;
    }
    if (!CommonUtils::ConvertStrToInt32(argv[kExecRankId], startParm.execRankId)) {
        ERROR_LOG("Parse execRankId[%s] failed.", argv[kExecRankId]);
        return false;
    }
    if (startParm.deviceId < 0 || startParm.execRankId < 0) {
        ERROR_LOG("DeviceId[%d] or execRankId[%d] is invalid", startParm.deviceId, startParm.execRankId);
        return false;
    }
    INFO_LOG("Parse user input successfully. fullGraphPath[%s] incGraphPath[%s] "
             "configFilePath[%s] deviceId[%d] execRankId[%d]",
             startParm.fullGraphPath.c_str(), startParm.incGraphPath.c_str(),
            startParm.configFilePath.c_str(), startParm.deviceId, startParm.execRankId);
    return true;
}

bool Initialize(const MainStartParma &startParm) {
    std::string deviceListStr = "[" + std::to_string(startParm.deviceId) + "]";
    std::map<ge::AscendString, ge::AscendString> config = {
        {"ge.exec.hcomGrouplist", LlmConfig::Instance().GetGroupNameList().c_str()}, // group name与rankid关系
        {"ge.exec.modelDeplyMode", "SPMD"}, // 多进程模式执行
        {"ge.exec.modelDeployDevicelist", deviceListStr.c_str()}, // 与SPMD参数配对使用，表示当前进程在那些device上加载模型
        {"ge.exec.rankId", std::to_string(startParm.execRankId).c_str()}, // 当前进程的exec rank id
        {"ge.exec.rankTableFile", LlmConfig::Instance().GetRankTableFile().c_str()}, // rank table文件
        {"ge.exec.deviceId", std::to_string(startParm.deviceId).c_str()},
        {"ge.socVersion", LlmConfig::Instance().GetSocVersion().c_str()},
        {"ge.externalWeight", "1"} // 使能权重外置
    };
    if (LlmConfig::Instance().CacheEnable()) {
        // 如果开启编译缓存需要设置缓存路径
        std::string cache_dir = LlmConfig::Instance().GetCacheDir() + "_" + std::to_string(startParm.execRankId);
        config["ge.graph_compiler_cache_dir"] = cache_dir.c_str();
        INFO_LOG("Current config is enable cache. cacheDir[%s] base graphKey[%s]",
            LlmConfig::Instance().GetCacheDir().c_str(), LlmConfig::Instance().GetGraphKey().c_str());
    }
    auto geRet = ge::GEInitialize(config);
    if (geRet != ge::SUCCESS) {
        ERROR_LOG("GEInitialize failed, ret = %d", geRet);
        return false;
    }
    return true;
}
}

int32_t main(int32_t argc, char *argv[]) {
    MainStartParma startParm = {};
    // 1.解析入参
    if (!ParseArgs(argc, argv, startParm)) {
        ERROR_LOG("Parse Args failed.");
        return -1;
    }
    // 2.解析配置文件内容
    if (!LlmConfig::Instance().GenerateLlmConfig(startParm.configFilePath)) {
        ERROR_LOG("Generate llm config failed.");
        return -1;
    }
    // 3.构造kv输入
    std::vector<ge::Tensor> kvInputs;
    if (!ConstructKvInputs(LlmConfig::Instance().GetKvDataDesc(), startParm.deviceId, kvInputs)) {
        ERROR_LOG("Construct kv input tensor failed.");
        return -1;
    }
    // 4.GE初始化
    if (!Initialize(startParm)) {
        ERROR_LOG("Do GeInitialize failed.");
        return -1;
    }
    // 5.构造session
    std::map<ge::AscendString, ge::AscendString> sessionOptions;
    auto session = std::make_shared<ge::Session>(sessionOptions);
    if (session == nullptr) {
        ERROR_LOG("Create session failed.");
        ge::GEFinalize();
        return -1;
    }
    // 6.执行全量图
    if (!ExecuteFullGraph(kvInputs, startParm.fullGraphPath, session)) {
        ERROR_LOG("Execute full graph failed.");
        ge::GEFinalize();
        return -1;
    }
    // 7.执行增量图
    if (!ExecuteIncGraph(kvInputs, startParm.incGraphPath, session)) {
        ERROR_LOG("Execute increment graph failed.");
        ge::GEFinalize();
        return -1;
    }
    ge::GEFinalize();
    return 0;
}