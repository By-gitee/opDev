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
constexpr uint16_t DECODER_LISTEN_PORT = 26001;
constexpr uint16_t PROMPT_CLUSTER_ID = 0;
constexpr uint32_t NUM_TENSORS = 4U;
constexpr size_t TENSOR_SIZE = 8 * 16 * sizeof(int32_t);
const std::vector<int64_t> TENSOR_SHAPE = {8, 16};
constexpr int32_t WAIT_TIME = 5;
constexpr int32_t EXPECTED_ARG_CNT = 4;
constexpr uint32_t ARG_INDEX_DEVICE_ID = 1;
constexpr uint32_t ARG_INDEX_LOCAL_IP = 2;
constexpr uint32_t ARG_INDEX_REMOTE_IP = 3;

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
    clusterInfo.remote_cluster_id = 1;
    IpInfo localIpInfo;
    localIpInfo.ip = localIp;
    localIpInfo.port = DECODER_LISTEN_PORT;
    clusterInfo.local_ip_infos.emplace_back(std::move(localIpInfo));
    IpInfo remoteIpInfo;
    remoteIpInfo.ip = remoteIp;
    remoteIpInfo.port = DECODER_LISTEN_PORT;
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

int Unlink(LlmDataDist &llmDataDist, const char *remoteIp)
{
    std::vector<Status> rets;
    std::vector<ClusterInfo> clusters;
    ClusterInfo clusterInfo;
    clusterInfo.remote_cluster_id = 1;
    IpInfo remoteIpInfo;
    remoteIpInfo.ip = remoteIp;
    remoteIpInfo.port = DECODER_LISTEN_PORT;
    clusterInfo.remote_ip_infos.emplace_back(std::move(remoteIpInfo));
    clusters.emplace_back(std::move(clusterInfo));
    auto ret = llmDataDist.UnlinkLlmClusters(clusters, rets);
    if (ret != LLM_SUCCESS) {
        printf("[ERROR] UnlinkLlmClusters failed, ret = %u\n", ret);
        return -1;
    }
    printf("[INFO] UnlinkLlmClusters success\n");
    return 0;
}

int32_t PushCache(LlmDataDist &llmDataDist, int64_t cacheId)
{
    std::vector<uint64_t> promptBlocks {5, 6, 7};
    std::vector<uint64_t> decoderBlocks {5, 6, 7};
    // 可以使用PushKvBlock推送多块block的数据
    Cache cache{};
    cache.cache_id = cacheId;
    auto ret = LLM_SUCCESS;
    CacheIndex cacheIndex{1, 1};
    for (uint32_t i = 0U; i < NUM_TENSORS; ++i) {
        KvCacheExtParam param{};
        param.src_layer_range = std::pair<int32_t, int32_t>(i, i);
        param.dst_layer_range = std::pair<int32_t, int32_t>(i, i);
        param.tensor_num_per_layer = 1;
        ret = llmDataDist.PushKvBlocks(cache, cacheIndex, promptBlocks, decoderBlocks, param);
        if (ret != LLM_SUCCESS) {
            printf("[ERROR] PushKvBlocks failed, ret = %u\n", ret);
            return -1;
        }
    }
    printf("[INFO] PushKvBlocks success\n");

    // 也可以使用PushKvCache推送一个batch中的连续数据
    CacheIndex cacheIndex2{1, 1, 4};
    KvCacheExtParam param2{};
    param2.src_layer_range = std::pair<int32_t, int32_t>(0, 0);
    param2.dst_layer_range = std::pair<int32_t, int32_t>(0, 0);
    param2.tensor_num_per_layer = 4;
    ret = llmDataDist.PushKvCache(cache, cacheIndex2, 4, -1, param2);
    if (ret != LLM_SUCCESS) {
        printf("[ERROR] PushKvCache failed, ret = %u\n", ret);
        return -1;
    }
    printf("[INFO] PushKvCache success\n");
    return 0;
}

void Finalize(LlmDataDist &llmDataDist, int64_t cacheId, bool linked, const char *remoteIp,
              const std::vector<void *> buffers)
{
    if (linked) {
        auto ret = Unlink(llmDataDist, remoteIp);
        if (ret != 0) {
            printf("[ERROR] Unlink failed, ret = %d\n", ret);
        } else {
            printf("[INFO] Unlink success\n");
        }
    }
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

int32_t RunPromptSample(const char *deviceId, const char *localIp, const char *remoteIp)
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
    bool linked = false;
    auto ret = llmDataDist.RegisterKvCache(cacheDesc, tensorAddrs, {}, cacheId);
    if (ret != LLM_SUCCESS) {
        printf("[ERROR] RegisterKvCache failed, ret = %u\n", ret);
        Finalize(llmDataDist, cacheId, linked, remoteIp, buffers);
        return -1;
    }
    // 3. RegisterKvCache成功后，可以获取cache中各tensor的地址用于后续操作
    printf("[INFO] RegisterKvCache success\n");
    for (size_t i = 0U; i < tensorAddrs.size(); ++i) {
        printf("[INFO] Tensor[%zu] addr = %p\n", i, reinterpret_cast<void *>(tensorAddrs[i]));
    }

    // 等待decoder注册完成
    std::this_thread::sleep_for(std::chrono::seconds(WAIT_TIME));

    // 4. 与decoder建链
    if (Link(llmDataDist, localIp, remoteIp) != 0) {
        Finalize(llmDataDist, cacheId, linked, remoteIp, buffers);
        return -1;
    }
    linked = true;

    // 5. 向decoder push cache
    if (PushCache(llmDataDist, cacheId) != 0) {
        Finalize(llmDataDist, cacheId, linked, remoteIp, buffers);
        return -1;
    }

    // 6. 释放Cache与llmDataDist
    Finalize(llmDataDist, cacheId, linked, remoteIp, buffers);
    printf("[INFO] Prompt Sample end\n");
    return 0;
}

int main(int32_t argc, char **argv)
{
    if (argc != EXPECTED_ARG_CNT) {
        printf("[ERROR] expect 3 args(deviceId, localHostIp, remoteHostIp), but got %d\n", argc - 1);
        return -1;
    }
    const auto deviceId = argv[ARG_INDEX_DEVICE_ID];
    const auto localIp = argv[ARG_INDEX_LOCAL_IP];
    const auto remoteIp = argv[ARG_INDEX_REMOTE_IP];
    printf("[INFO] deviceId = %s, localIp = %s, remoteIp = %s\n", deviceId, localIp, remoteIp);
    auto ret = RunPromptSample(deviceId, localIp, remoteIp);
    return ret;
}