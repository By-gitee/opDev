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
constexpr uint16_t DECODER_CLUSTER_ID = 1;
constexpr uint32_t NUM_TENSORS = 4U;
constexpr int32_t WAIT_PROMPT_TIME = 5;
constexpr int32_t EXPECTED_ARG_CNT = 4;
constexpr uint32_t ARG_INDEX_DEVICE_ID = 1;
constexpr uint32_t ARG_INDEX_LOCAL_IP = 2;
constexpr uint32_t ARG_INDEX_REMOTE_IP = 3;

int Initialize(LlmDataDist &llmDataDist, const std::string &deviceId)
{
    std::map<AscendString, AscendString> options;
    options[OPTION_DEVICE_ID] = deviceId.c_str();
    options[OPTION_BUF_POOL_CFG] = R"({
"buf_pool_size": 2147483648
})";
    auto ret = llmDataDist.Initialize(options);
    if (ret != LLM_SUCCESS) {
        printf("[ERROR] Initialize failed, ret = %u\n", ret);
        return -1;
    }
    printf("[INFO] Initialize success\n");
    return LLM_SUCCESS;
}

int Link(LlmDataDist &llmDataDist, const char *localIp, const char *remoteIp)
{
    std::vector<Status> rets;
    std::vector<ClusterInfo> clusters;
    ClusterInfo clusterInfo;
    IpInfo localIpInfo;
    localIpInfo.ip = localIp;
    IpInfo remoteIpInfo;
    remoteIpInfo.ip = remoteIp;
    remoteIpInfo.port = PROMPT_LISTEN_PORT;
    clusterInfo.remote_cluster_id = PROMPT_CLUSTER_ID;
    clusterInfo.local_ip_infos.emplace_back(std::move(localIpInfo));
    clusterInfo.remote_ip_infos.emplace_back(std::move(remoteIpInfo));
    clusters.emplace_back(std::move(clusterInfo));
    auto ret = llmDataDist.LinkLlmClusters(clusters, rets);
    if (ret != LLM_SUCCESS) {
        printf("[ERROR] LinkLlmClusters failed, ret = %u\n", ret);
        return -1;
    }
    printf("[INFO] LinkLlmClusters success\n");
    return 0;
}

int32_t PullCache(LlmDataDist &llmDataDist, Cache &cache)
{
    std::vector<uint64_t> promptBlocks {1,2,3,6,5,4,7};
    std::vector<uint64_t> decoderBlocks {1,2,3,6,5,4,7};
    CacheIndex cacheIndex{PROMPT_CLUSTER_ID, 1, 0};
    // 可以使用PullKvBlock拉取多块block的数据
    auto ret = llmDataDist.PullKvBlocks(cacheIndex, cache, promptBlocks, decoderBlocks);
    if (ret != LLM_SUCCESS) {
        printf("[ERROR] PullKvBlocks failed, ret = %u\n", ret);
        return -1;
    }
    printf("[INFO] PullKvBlocks success\n");
    // 也可以使用PullKvCache拉取一个batch中的连续数据
    cacheIndex.batch_index = 0;
    ret = llmDataDist.PullKvCache(cacheIndex, cache, 0);
    if (ret != LLM_SUCCESS) {
        printf("[ERROR] PullKvCache failed, ret = %u\n", ret);
        return -1;
    }
    printf("[INFO] PullKvCache success\n");
    return 0;
}

int32_t RunWithCache(LlmDataDist &llmDataDist, Cache &npuCache)
{
    Cache host_cache{};
    host_cache.cache_desc = npuCache.cache_desc;
    host_cache.cache_desc.placement = CachePlacement::kHost;
    auto buffers = std::vector<std::vector<int32_t>>(host_cache.cache_desc.num_tensors,
                                                     std::vector<int32_t>(8 * 16));
    for (uint32_t i = 0; i < host_cache.cache_desc.num_tensors; ++i) {
        std::iota(buffers[i].begin(), buffers[i].end(), 0);
        host_cache.tensor_addrs.emplace_back(reinterpret_cast<uint64_t>(buffers[i].data()));
    }

    // 通过D2H拷贝npu cache的值到host
    std::vector<uint64_t> blockIndices(host_cache.cache_desc.shape.front());
    // 可以使用CopyKvBlocks一次拷贝多块内存
    std::iota(blockIndices.begin(), blockIndices.end(), 0);
    auto ret = llmDataDist.CopyKvBlocks(host_cache, npuCache, blockIndices, {blockIndices});
    if (ret != LLM_SUCCESS) {
        printf("[ERROR] CopyKvBlocks failed, ret = %u\n", ret);
        return -1;
    }
    printf("[INFO] CopyKvBlocks success\n");
    // 也可以使用CopyKvCache拷贝一个batch中的连续数据
    ret = llmDataDist.CopyKvCache(npuCache, host_cache, 0, 0);
    if (ret != LLM_SUCCESS) {
        printf("[ERROR] CopyKvCache failed, ret = %u\n", ret);
        return -1;
    }
    printf("[INFO] CopyKvCache success\n");
    // 打印数据
    printf("[INFO] print cache data:\n");
    for (size_t i = 0; i < buffers[0].size(); ++i) {
        if ((i > 0) && (i % host_cache.cache_desc.shape.back() == 0)) {
            printf("\n");
        }
        printf("%d\t", buffers[0][i]);
    }
    printf("\n");
    return 0;
}

void OnError(LlmDataDist &llmDataDist, Cache &cache)
{
    if (cache.cache_id > 0) {
        (void) llmDataDist.DeallocateCache(cache.cache_id);
    }
    llmDataDist.Finalize();
}

int32_t RunDecoderSample(const char *deviceId, const char *localIp, const char *remoteIp)
{
    printf("[INFO] Decoder Sample start\n");
    // 1. 初始化
    LlmDataDist llmDataDist(DECODER_CLUSTER_ID, LlmRole::kDecoder);
    if (Initialize(llmDataDist, deviceId) != 0) {
        return -1;
    }
    // 2. 与prompt建链
    if (Link(llmDataDist, localIp, remoteIp) != 0) {
        return -1;
    }

    // 3. 申请device cache
    CacheDesc kv_cache_desc{};
    kv_cache_desc.num_tensors = NUM_TENSORS;
    kv_cache_desc.data_type = DT_INT32;
    kv_cache_desc.shape = {8, 16};
    Cache cache{};
    auto ret = llmDataDist.AllocateCache(kv_cache_desc, cache);
    if (ret != LLM_SUCCESS) {
        printf("[ERROR] AllocateCache failed, ret = %u\n", ret);
        OnError(llmDataDist, cache);
        return -1;
    }
    printf("[INFO] AllocateCache success\n");
    for (size_t i = 0U; i < cache.tensor_addrs.size(); ++i) {
        printf("[INFO] Tensor[%zu] addr = %p\n", i, reinterpret_cast<void *>(cache.tensor_addrs[i]));
    }

    // 4. 等待Prompt写完cache，实际业务场景可通过合适方式实现通知
    std::this_thread::sleep_for(std::chrono::seconds(WAIT_PROMPT_TIME));
    // 5. 从prompt拉取Cache
    if (PullCache(llmDataDist, cache) != 0) {
        OnError(llmDataDist, cache);
        return -1;
    }
    // 6. 使用拉取后的Cache，如进行模型推理，此处简单打印下pull回来的数据
    if (RunWithCache(llmDataDist, cache) != 0) {
        OnError(llmDataDist, cache);
        return -1;
    }

    // 7. 释放Cache与LlmDatadist
    ret = llmDataDist.DeallocateCache(cache.cache_id);
    if (ret != LLM_SUCCESS) {
        printf("[ERROR] DeallocateCache failed, ret = %u\n", ret);
    } else {
        printf("[INFO] DeallocateCache success\n");
    }
    llmDataDist.Finalize();
    printf("[INFO] Finalize success\n");
    printf("[INFO] Decoder Sample end\n");
    return 0;
}

int main(int32_t argc, char **argv)
{
    if (argc != EXPECTED_ARG_CNT) {
        printf("[ERROR] expect 3 args(deviceId, localIp, remoteIp), but got %d\n", argc - 1);
        return -1;
    }
    const auto deviceId = argv[ARG_INDEX_DEVICE_ID];
    const auto localIp = argv[ARG_INDEX_LOCAL_IP];
    const auto remoteIp = argv[ARG_INDEX_REMOTE_IP];
    printf("[INFO] deviceId = %s, localIp = %s, remoteIp = %s\n", deviceId, localIp, remoteIp);
    auto ret = RunDecoderSample(deviceId, localIp, remoteIp);
    return ret;
}