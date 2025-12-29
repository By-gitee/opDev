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
constexpr uint16_t DECODER_CLUSTER_ID = 1;
constexpr uint32_t NUM_TENSORS = 4U;
constexpr size_t TENSOR_SIZE = 8 * 16 * sizeof(int32_t);
const std::vector<int64_t> TENSOR_SHAPE = {8, 16};
constexpr size_t TENSOR_BLOCK_ELEMENT_NUM = 16;
constexpr int32_t WAIT_PROMPT_TIME = 5;
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

int Initialize(LlmDataDist &llmDataDist, const std::string &deviceId)
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
    clusterInfo.remote_cluster_id = 0;
    IpInfo localIpInfo;
    localIpInfo.ip = localIp;
    localIpInfo.port = PROMPT_LISTEN_PORT;
    clusterInfo.local_ip_infos.emplace_back(std::move(localIpInfo));
    IpInfo remoteIpInfo;
    remoteIpInfo.ip = remoteIp;
    remoteIpInfo.port = PROMPT_LISTEN_PORT;
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
    clusterInfo.remote_cluster_id = 0;
    IpInfo remoteIpInfo;
    remoteIpInfo.ip = remoteIp;
    remoteIpInfo.port = PROMPT_LISTEN_PORT;
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

int32_t CheckBuffers(const std::vector<void *> &buffers, const std::vector<uint32_t> &checkIndexList)
{
    for (auto buffer : buffers) {
        std::vector<int32_t> hostBuffer(TENSOR_SIZE / sizeof(int32_t));
        CHECK_ACL(aclrtMemcpy(&hostBuffer[0], TENSOR_SIZE, buffer, TENSOR_SIZE, ACL_MEMCPY_DEVICE_TO_HOST));
        for (auto checkIndex : checkIndexList) {
            for (size_t i = 0U; i < TENSOR_BLOCK_ELEMENT_NUM; ++i) {
                auto expect = checkIndex * TENSOR_BLOCK_ELEMENT_NUM + i;
                if (hostBuffer[expect] != expect) {
                    printf("[ERROR] Buffer check failed, index = %zu, val = %d, expect val = %zu\n",
                           expect, hostBuffer[expect], expect);
                    return -1;
                }
            }
        }
    }
    printf("[INFO] CheckBuffers success\n");
    return 0;
}

int32_t PullCache(LlmDataDist &llmDataDist, int64_t cacheId)
{
    std::vector<uint64_t> promptBlocks {1, 2, 3};
    std::vector<uint64_t> decoderBlocks {1, 2, 3};
    CacheIndex cacheIndex{PROMPT_CLUSTER_ID, 1, 0};
    // 可以使用PullKvBlock拉取多块block的数据
    Cache cache{};
    cache.cache_id = cacheId;
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

int32_t RunDecoderSample(const char *deviceId, const char *localIp, const char *remoteIp)
{
    printf("[INFO] Decoder Sample start\n");
    // 1. 初始化
    LlmDataDist llmDataDist(DECODER_CLUSTER_ID, LlmRole::kDecoder);
    if (Initialize(llmDataDist, deviceId) != 0) {
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

    // 4. 等待prompt写完cache，实际业务场景可通过合适方式实现通知
    std::this_thread::sleep_for(std::chrono::seconds(WAIT_PROMPT_TIME));

    // 5. 与prompt建链
    if (Link(llmDataDist, localIp, remoteIp) != 0) {
        Finalize(llmDataDist, cacheId, linked, remoteIp, buffers);
        return -1;
    }
    linked = true;

    // 6. 从prompt拉取cache
    if (PullCache(llmDataDist, cacheId) != 0) {
        Finalize(llmDataDist, cacheId, linked, remoteIp, buffers);
        return -1;
    }

    if (CheckBuffers(buffers, {0, 1, 2, 3}) != 0) {
        Finalize(llmDataDist, cacheId, linked, remoteIp, buffers);
        return -1;
    }

    // 7. 解除链路
    if (Unlink(llmDataDist, remoteIp) != 0) {
        Finalize(llmDataDist, cacheId, linked, remoteIp, buffers);
        return -1;
    }
    linked = false;

    // 8. 释放cache与llmDataDist
    llmDataDist.Finalize();
    printf("[INFO] Finalize success\n");
    printf("[INFO] Decoder Sample end\n");
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
    auto ret = RunDecoderSample(deviceId, localIp, remoteIp);
    return ret;
}