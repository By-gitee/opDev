/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "llm_datadist/llm_datadist.h"

using namespace llm_datadist;

constexpr uint16_t PROMPT_LISTEN_PORT = 26000;
constexpr uint16_t PROMPT_CLUSTER_ID = 0;
constexpr uint32_t NUM_TENSORS = 4U;
constexpr int32_t WAIT_TIME = 30;
constexpr int32_t EXPECTED_ARG_CNT = 3;
constexpr uint32_t ARG_INDEX_DEVICE_ID = 1;
constexpr uint32_t ARG_INDEX_LOCAL_IP = 2;

int Initialize(LlmDataDist &llmDataDist, const std::string &deviceId, const std::string &localIp)
{
    std::map<AscendString, AscendString> options;
    options[OPTION_DEVICE_ID] = deviceId.c_str();
    options[OPTION_LISTEN_IP_INFO] = (localIp + ":" + std::to_string(PROMPT_LISTEN_PORT)).c_str();
    options[OPTION_BUF_POOL_CFG] = R"({
"buf_cfg":[{"total_size":2097152,"blk_size":256,"max_buf_size":8192}],
"buf_pool_size": 2147483648
})";
    options[OPTION_ENABLE_SET_ROLE] = "1";
    auto ret = llmDataDist.Initialize(options);
    if (ret != LLM_SUCCESS) {
        printf("[ERROR] Initialize failed, ret = %u\n", ret);
        return -1;
    }
    printf("[INFO] Initialize success\n");
    return LLM_SUCCESS;
}

int32_t RunWithCache(LlmDataDist &llmDataDist, Cache &npuCache)
{
    // 在本Sample中，通过CopyCache/CopyBlocksCache接口对其进行赋值
    // 1. 申请host cache, 并赋值，用于后续Copy
    Cache hostCache{};
    hostCache.cache_desc = npuCache.cache_desc;
    hostCache.cache_desc.placement = CachePlacement::kHost;
    auto buffers = std::vector<std::vector<int32_t>>(hostCache.cache_desc.num_tensors,
                                                     std::vector<int32_t>(8 * 16));
    for (uint32_t i = 0; i < hostCache.cache_desc.num_tensors; ++i) {
        std::iota(buffers[i].begin(), buffers[i].end(), 0);
        hostCache.tensor_addrs.emplace_back(reinterpret_cast<uint64_t>(buffers[i].data()));
    }

    // 通过Copy H2D 为npu_cache赋值(CacheKvCache接口暂不支持H2D copy)
    std::vector<uint64_t> blockIndices(hostCache.cache_desc.shape.front());
    std::iota(blockIndices.begin(), blockIndices.end(), 0);
    auto ret = llmDataDist.CopyKvBlocks(hostCache, npuCache, blockIndices, {blockIndices});
    if (ret != LLM_SUCCESS) {
        printf("[ERROR] CopyKvBlocks failed, ret = %u\n", ret);
        return -1;
    }
    printf("[INFO] CopyKvBlocks success\n");
    return 0;
}

void OnError(LlmDataDist &llmDataDist, Cache &cache)
{
    if (cache.cache_id > 0) {
        (void) llmDataDist.DeallocateCache(cache.cache_id);
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
    // 2. 申请device cache
    CacheDesc cache_desc{};
    cache_desc.num_tensors = NUM_TENSORS;
    cache_desc.data_type = DT_INT32;
    cache_desc.shape = {8, 16};
    Cache cache{};
    auto ret = llmDataDist.AllocateCache(cache_desc, cache);
    if (ret != LLM_SUCCESS) {
        printf("[ERROR] AllocateCache failed, ret = %u\n", ret);
        OnError(llmDataDist, cache);
        return -1;
    }
    // 3. Allocate成功后，可以获取cache中各tensor的地址用于后续操作
    printf("[INFO] AllocateCache success\n");
    for (size_t i = 0U; i < cache.tensor_addrs.size(); ++i) {
        printf("[INFO] Tensor[%zu] addr = %p\n", i, reinterpret_cast<void *>(cache.tensor_addrs[i]));
    }
    if (RunWithCache(llmDataDist, cache) != 0) {
        printf("[ERROR] RunWithCache failed\n");
        OnError(llmDataDist, cache);
        return -1;
    }

    // 4. 等待Decoder拉取cache
    std::this_thread::sleep_for(std::chrono::seconds(WAIT_TIME));

    // 5. 释放Cache与LlmDatadist
    ret = llmDataDist.DeallocateCache(cache.cache_id);
    if (ret != LLM_SUCCESS) {
        printf("[ERROR] DeallocateCache failed, ret = %u\n", ret);
    } else {
        printf("[INFO] DeallocateCache success\n");
    }
    llmDataDist.Finalize();
    printf("[INFO] Finalize success\n");
    printf("[INFO] Prompt Sample end\n");
    return 0;
}

int main(int32_t argc, char **argv)
{
    if (argc != EXPECTED_ARG_CNT) {
        printf("[ERROR] expect 2 args(deviceId, localIp), but got %d\n", argc - 1);
        return -1;
    }
    const auto deviceId = argv[ARG_INDEX_DEVICE_ID];
    const auto localIp = argv[ARG_INDEX_LOCAL_IP];
    printf("[INFO] deviceId = %s, localIp = %s\n", deviceId, localIp);
    auto ret = RunPromptSample(deviceId, localIp);
    return ret;
}