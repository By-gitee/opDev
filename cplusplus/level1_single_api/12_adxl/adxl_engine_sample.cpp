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
#include <fstream>
#include "acl/acl.h"
#include "adxl/adxl_engine.h"

using namespace adxl;
namespace{
constexpr int32_t WAIT_REG_TIME = 5;
constexpr int32_t WAIT_TRANS_TIME = 20;
constexpr int32_t CLIENT_EXPECTED_ARG_CNT = 4;
constexpr uint32_t ARG_INDEX_DEVICE_ID = 1;
constexpr uint32_t ARG_INDEX_LOCAL_ENINE = 2;
constexpr uint32_t CLIENT_ARG_INDEX_REMOTE_ENINE = 3;
constexpr int32_t SERVER_EXPECTED_ARG_CNT = 3;

#define CHECK_ACL(x)                                                                        \
    do {                                                                                    \
        aclError __ret = x;                                                                 \
        if (__ret != ACL_ERROR_NONE) {                                                      \
            std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << __ret << std::endl; \
        }                                                                                   \
    } while (0);
}

int Initialize(AdxlEngine &adxlEngine, const char *localEngine)
{
    std::map<AscendString, AscendString> options;
    auto ret = adxlEngine.Initialize(localEngine, options);
    if (ret != SUCCESS) {
        printf("[ERROR] Initialize failed, ret = %u\n", ret);
        return -1;
    }
    printf("[INFO] Initialize success\n");
    return 0;
}

int Connect(AdxlEngine &adxlEngine, const char *remoteEngine)
{
    auto ret = adxlEngine.Connect(remoteEngine);
    if (ret != SUCCESS) {
        printf("[ERROR] Connect failed, ret = %u\n", ret);
        return -1;
    }
    printf("[INFO] Connect success\n");
    return 0;
}

int Disconnect(AdxlEngine &adxlEngine, const char *remoteEngine)
{
    auto ret = adxlEngine.Disconnect(remoteEngine);
    if (ret != SUCCESS) {
        printf("[ERROR] Disconnect failed, ret = %u\n", ret);
        return -1;
    }
    printf("[INFO] Disconnect success\n");
    return 0;
}

int32_t Transfer(AdxlEngine &adxlEngine, int32_t &src, const char *remoteEngine)
{
    uintptr_t dstAddr;
    std::ifstream("./tmp") >> std::hex >> dstAddr;

    TransferOpDesc desc{reinterpret_cast<uintptr_t>(&src), reinterpret_cast<uintptr_t>(dstAddr), sizeof(int32_t)};
    auto ret = adxlEngine.TransferSync(remoteEngine, READ, {desc});
    if (ret != SUCCESS) {
        printf("[ERROR] TransferSync read failed, ret = %u\n", ret);
        return -1;
    }
    printf("[INFO] TransferSync read success, src = %d\n", src);

    src = 2;
    ret = adxlEngine.TransferSync(remoteEngine, WRITE, {desc});
    if (ret != SUCCESS) {
        printf("[ERROR] TransferSync write failed, ret = %u\n", ret);
        return -1;
    }
    printf("[INFO] TransferSync write success, src = %d\n", src);
    return 0;
}

void ClientFinalize(AdxlEngine &adxlEngine, bool connected, const char *remoteEngine,
                    const std::vector<MemHandle> handles, const std::vector<void *> hostBuffers = {})
{
    if (connected) {
        auto ret = Disconnect(adxlEngine, remoteEngine);
        if (ret != 0) {
            printf("[ERROR] Disconnect failed, ret = %d\n", ret);
        } else {
            printf("[INFO] Disconnect success\n");
        }
    }

    for (auto handle : handles) {
        auto ret = adxlEngine.DeregisterMem(handle);
        if (ret != 0) {
            printf("[ERROR] DeregisterMem failed, ret = %u\n", ret);
        } else {
            printf("[INFO] DeregisterMem success\n");
        }
    }
    for (auto buffer : hostBuffers) {
        aclrtFreeHost(buffer);
    }
    adxlEngine.Finalize();
}

void ServerFinalize(AdxlEngine &adxlEngine,
                    const std::vector<MemHandle> handles,
                    const std::vector<void *> devBuffers = {})
{
    for (auto handle : handles) {
        auto ret = adxlEngine.DeregisterMem(handle);
        if (ret != 0) {
            printf("[ERROR] DeregisterMem failed, ret = %u\n", ret);
        } else {
            printf("[INFO] DeregisterMem success\n");
        }
    }
    for (auto buffer : devBuffers) {
        aclrtFree(buffer);
    }
    adxlEngine.Finalize();
}

int32_t RunClient(const char *localEngine, const char *remoteEngine)
{
    printf("[INFO] client start\n");
    // 1. 初始化
    AdxlEngine adxlEngine;
    if (Initialize(adxlEngine, localEngine) != 0) {
        printf("[ERROR] Initialize AdxlEngine failed\n");
        return -1;
    }
    // 2. 注册内存地址
    int32_t *src = nullptr;
    CHECK_ACL(aclrtMallocHost(reinterpret_cast<void **>(&src), sizeof(int32_t)));
    bool connected = false;
    MemDesc desc{};
    desc.addr = reinterpret_cast<uintptr_t>(src);
    desc.len = sizeof(int32_t);
    MemHandle handle = nullptr;
    auto ret = adxlEngine.RegisterMem(desc, MEM_HOST, handle);
    if (ret != SUCCESS) {
        printf("[ERROR] RegisterMem failed, ret = %u\n", ret);
        ClientFinalize(adxlEngine, connected, remoteEngine, {handle}, {src});
        return -1;
    }
    printf("[INFO] RegisterMem success\n");

    // 等待server注册完成
    std::this_thread::sleep_for(std::chrono::seconds(WAIT_REG_TIME));

    // 3. 与server建链
    if (Connect(adxlEngine, remoteEngine) != 0) {
        ClientFinalize(adxlEngine, connected, remoteEngine, {handle}, {src});
        return -1;
    }
    connected = true;

    // 4. 从server get内存，并向server put内存
    if (Transfer(adxlEngine, *src, remoteEngine) != 0) {
        ClientFinalize(adxlEngine, connected, remoteEngine, {handle}, {src});
        return -1;
    }

    // 5. 释放Cache与llmDataDist
    ClientFinalize(adxlEngine, connected, remoteEngine, {handle}, {src});
    printf("[INFO] Finalize success\n");
    printf("[INFO] Prompt Sample end\n");
    return 0;
}

int32_t RunServer(const char *localEngine)
{
    printf("[INFO] server start\n");
    // 1. 初始化
    AdxlEngine adxlEngine;
    if (Initialize(adxlEngine, localEngine) != 0) {
        printf("[ERROR] Initialize AdxlEngine failed\n");
        return -1;
    }
    // 2. 注册内存地址
    int32_t dst = 1;
    int32_t *buffer = nullptr;
    CHECK_ACL(aclrtMalloc((void **)&buffer, sizeof(int32_t), ACL_MEM_MALLOC_HUGE_ONLY));
    // init device buffer
    CHECK_ACL(aclrtMemcpy(buffer, sizeof(int32_t), &dst, sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE));

    MemDesc desc{};
    desc.addr = reinterpret_cast<uintptr_t>(buffer);
    desc.len = sizeof(int32_t);
    MemHandle handle = nullptr;
    auto ret = adxlEngine.RegisterMem(desc, MEM_DEVICE, handle);
    if (ret != SUCCESS) {
        printf("[ERROR] RegisterMem failed, ret = %u\n", ret);
        ServerFinalize(adxlEngine, {handle}, {buffer});
        return -1;
    }
    // 3. RegisterMem成功后，将地址保存到本地文件中等待client读取
    printf("[INFO] RegisterMem success, dst addr:%p\n", buffer);
    std::ofstream tmp_file("./tmp");  // 默认就是 std::ios::out | std::ios::trunc
    if (tmp_file) {
        tmp_file << buffer << std::endl;
    }

    // 4. 等待client transfer
    std::this_thread::sleep_for(std::chrono::seconds(WAIT_TRANS_TIME));

    CHECK_ACL(aclrtMemcpy(&dst, sizeof(int32_t), buffer, sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST));
    printf("[INFO] After transfer, dst value:%d\n", dst);

    // 5. 释放Cache与llmDataDist
    ServerFinalize(adxlEngine, {handle}, {buffer});
    printf("[INFO] Finalize success\n");
    printf("[INFO] server Sample end\n");
    return 0;
}

int main(int32_t argc, char **argv)
{
    bool isClient = false;
    std::string deviceId;
    std::string localEngine;
    std::string remoteEngine;
    if (argc == CLIENT_EXPECTED_ARG_CNT) {
        isClient = true;
        deviceId = argv[ARG_INDEX_DEVICE_ID];
        localEngine = argv[ARG_INDEX_LOCAL_ENINE];
        remoteEngine = argv[CLIENT_ARG_INDEX_REMOTE_ENINE];
        printf("[INFO] deviceId = %s, localEngine = %s, remoteEngine = %s\n",
               deviceId.c_str(), localEngine.c_str(), remoteEngine.c_str());
    } else if (argc == SERVER_EXPECTED_ARG_CNT) {
        deviceId = argv[ARG_INDEX_DEVICE_ID];
        localEngine = argv[ARG_INDEX_LOCAL_ENINE];
        printf("[INFO] deviceId = %s, localEngine = %s\n", deviceId.c_str(), localEngine.c_str());
    } else {
        printf("[ERROR] client expect 3 args(deviceId, localEngine, remoteEngine), "
               "server expect 2 args(deviceId, localEngine), but got %d\n", argc - 1);
        return -1;
    }
    int32_t device = std::stoi(deviceId);
    CHECK_ACL(aclrtSetDevice(device));

    int32_t ret = 0;
    if (isClient) {
        ret = RunClient(localEngine.c_str(), remoteEngine.c_str());
    } else {
        ret = RunServer(localEngine.c_str());
    }
    CHECK_ACL(aclrtResetDevice(device));
    return ret;
}