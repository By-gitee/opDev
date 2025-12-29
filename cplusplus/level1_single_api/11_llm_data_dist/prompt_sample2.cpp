/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include <numeric>
#include <cstdio>
#include <thread>
#include <iostream>
#include "acl/acl.h"
#include "llm_datadist/llm_datadist.h"

using namespace llm_datadist;
namespace{
constexpr uint16_t PROMPT_LISTEN_PORT = 26000;
constexpr uint16_t PROMPT_CLUSTER_ID = 0;
constexpr uint32_t NUM_TENSORS = 4U;
constexpr size_t TENSOR_SIZE = 8 * 16 * sizeof(int32_t);
const std::vector<int64_t> TENSOR_SHAPE = {8, 16};
constexpr int32_t WAIT_TIME = 10;
constexpr int32_t EXPECTED_ARG_CNT = 3;
constexpr uint32_t ARG_INDEX_DEVICE_ID = 1;
constexpr uint32_t ARG_INDEX_LOCAL_IP = 2;

#define CHECK_ACL(x)                                                                        \
    do {                                                                                    \
        aclError __ret = x;                                                                 \
        if (__ret != ACL_ERROR_NONE) {                                                      \
            std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << __ret << std::endl; \
        }                                                                                   \
    } while (0);
}

int Initialize(LlmDataDist &llmDataDist, const std::string &deviceId, const std::string &localIp)
{
    std::map<AscendString, AscendString> options;
    options[OPTION_DEVICE_ID] = deviceId.c_str();
    options[OPTION_LISTEN_IP_INFO] = (localIp + ":" + std::to_string(PROMPT_LISTEN_PORT)).c_str();
    auto ret = llmDataDist.Initialize(options);
    if (ret != LLM_SUCCESS) {
        printf("[ERROR] Initialize failed, ret = %u\n", ret);
        return -1;
    }
    printf("[INFO] Initialize success\n");
    return LLM_SUCCESS;
}

void Finalize(LlmDataDist &llmDataDist, int64_t cacheId, const std::vector<void *> buffers)
{
    if (cacheId > 0) {
        auto ret = llmDataDist.UnregisterKvCache(cacheId);
        if (ret != 0) {
            printf("[ERROR] UnregisterKvCache failed, ret = %u\n", ret);
        } else {
            printf("[INFO] UnregisterKvCache success\n");
        }
    }
    for (auto buffer : buffers) {
        aclrtFree(buffer);
    }
    llmDataDist.Finalize();
}

int32_t RunPromptSample(const char *deviceId, const char *localIp)
{
    printf("[INFO] Prompt Sample start\n");
    // 1. 初始化
    LlmDataDist llmDataDist(PROMPT_CLUSTER_ID, LlmRole::kPrompt);
    if (Initialize(llmDataDist, deviceId, localIp) != 0) {
        printf("[ERROR] Initialize LlmDataDist failed\n");
        return -1;
    }
    // 2. 注册内存地址
    CacheDesc cacheDesc{};
    cacheDesc.num_tensors = NUM_TENSORS;
    cacheDesc.data_type = DT_INT32;
    cacheDesc.shape = TENSOR_SHAPE;
    std::vector<uint64_t> tensorAddrs;
    std::vector<void *> buffers;
    for (uint32_t i = 0U; i < NUM_TENSORS; ++i) {
        int32_t *buffer = nullptr;
        CHECK_ACL(aclrtMalloc((void **)&buffer, TENSOR_SIZE, ACL_MEM_MALLOC_HUGE_ONLY));

        // init device buffer
        std::vector<int32_t> hostBuffer(TENSOR_SIZE / sizeof(int32_t));
        std::iota(hostBuffer.begin(), hostBuffer.end(), 0);
        CHECK_ACL(aclrtMemcpy(buffer, TENSOR_SIZE, &hostBuffer[0], TENSOR_SIZE, ACL_MEMCPY_HOST_TO_DEVICE));

        tensorAddrs.emplace_back(reinterpret_cast<uint64_t>(buffer));
        buffers.emplace_back(reinterpret_cast<void *>(buffer));
    }
    int64_t cacheId = -1;
    auto ret = llmDataDist.RegisterKvCache(cacheDesc, tensorAddrs, {}, cacheId);
    if (ret != LLM_SUCCESS) {
        printf("[ERROR] RegisterKvCache failed, ret = %u\n", ret);
        Finalize(llmDataDist, cacheId, buffers);
        return -1;
    }
    // 3. RegisterKvCache成功后，可以获取cache中各tensor的地址用于后续操作
    printf("[INFO] RegisterKvCache success\n");
    for (size_t i = 0U; i < tensorAddrs.size(); ++i) {
        printf("[INFO] Tensor[%zu] addr = %p\n", i, reinterpret_cast<void *>(tensorAddrs[i]));
    }

    // 4. 等待decoder拉取cache
    std::this_thread::sleep_for(std::chrono::seconds(WAIT_TIME));

    // 5. 释放Cache与llmDataDist
    Finalize(llmDataDist, cacheId, buffers);
    printf("[INFO] Prompt Sample end\n");
    return 0;
}

int main(int32_t argc, char **argv)
{
    if (argc != EXPECTED_ARG_CNT) {
        printf("[ERROR] expect 2 args(deviceId, localHostIp), but got %d\n", argc - 1);
        return -1;
    }
    const auto deviceId = argv[ARG_INDEX_DEVICE_ID];
    const auto localIp = argv[ARG_INDEX_LOCAL_IP];
    printf("[INFO] deviceId = %s, localIp = %s\n", deviceId, localIp);
    auto ret = RunPromptSample(deviceId, localIp);
    return ret;
}