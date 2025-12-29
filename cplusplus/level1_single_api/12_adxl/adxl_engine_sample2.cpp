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
#include <string.h>
#include "acl/acl.h"
#include "adxl/adxl_engine.h"

using namespace adxl;
namespace{
constexpr int32_t WAIT_TIME = 5;
constexpr int32_t EXPECTED_ARG_CNT = 4;
constexpr uint32_t ARG_INDEX_DEVICE_ID = 1;
constexpr uint32_t ARG_INDEX_LOCAL_ENGINE = 2;
constexpr uint32_t ARG_INDEX_REMOTE_ENGINE = 3;
constexpr uint32_t MAX_ENGINE_NAME_LEN = 30;

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

int32_t Transfer(AdxlEngine &adxlEngine, uint8_t *&buffer, uint8_t *&buffer2,
                 const char *localEngine, const char *remoteEngine)
{
    uintptr_t remoteAddr;
    uintptr_t remoteAddr2;
    std::ifstream(remoteEngine) >> std::hex >> remoteAddr >> remoteAddr2;
    printf("[INFO] Get remote addr success, addr:%p, add2:%p\n",
           reinterpret_cast<void *>(remoteAddr), reinterpret_cast<void *>(remoteAddr2));
    if (std::string(localEngine) == std::min(std::string(localEngine), std::string(remoteEngine))) {
        // init device buffer
        printf("[INFO] Local engine test write, write value:%s\n", localEngine);
        CHECK_ACL(aclrtMemcpy(buffer, MAX_ENGINE_NAME_LEN, localEngine, strlen(localEngine),
                              ACL_MEMCPY_HOST_TO_DEVICE));
        TransferOpDesc desc{reinterpret_cast<uintptr_t>(buffer), remoteAddr, strlen(localEngine)};
        auto ret = adxlEngine.TransferSync(remoteEngine, WRITE, {desc});
        if (ret != SUCCESS) {
            printf("[ERROR] TransferSync write failed, remoteAddr:%p, ret = %u\n",
                   reinterpret_cast<void *>(remoteAddr), ret);
            return -1;
        }
        printf("[INFO] TransferSync write success, remoteAddr:%p, value:%s\n",
               reinterpret_cast<void *>(remoteAddr), localEngine);

        // 等待对端读
        CHECK_ACL(aclrtMemcpy(buffer2, MAX_ENGINE_NAME_LEN, localEngine, strlen(localEngine),
                              ACL_MEMCPY_HOST_TO_DEVICE));
        std::this_thread::sleep_for(std::chrono::seconds(WAIT_TIME));
    } else {
        // 等待对端写内存完成
        printf("[INFO] Local engine test read, expect read value:%s\n", remoteEngine);
        std::this_thread::sleep_for(std::chrono::seconds(WAIT_TIME));
        char value[MAX_ENGINE_NAME_LEN] = {};
        CHECK_ACL(aclrtMemcpy(value, MAX_ENGINE_NAME_LEN, buffer, strlen(remoteEngine), ACL_MEMCPY_DEVICE_TO_HOST));
        printf("[INFO] Wait peer TransferSync write end, remoteAddr:%p, value = %s\n",
               reinterpret_cast<void *>(remoteAddr), value);
        if (std::string(remoteEngine) != value) {
            printf("[ERROR] Failed to check peer write value:%s, expect:%s\n", value, remoteEngine);
            return -1;
        } else {
            printf("[INFO] Check peer write value success\n");
        }

        TransferOpDesc desc{reinterpret_cast<uintptr_t>(buffer2), remoteAddr2, strlen(remoteEngine)};
        auto ret = adxlEngine.TransferSync(remoteEngine, READ, {desc});
        if (ret != SUCCESS) {
            printf("[ERROR] TransferSync read failed, remoteAddr:%p, ret = %u\n",
                   reinterpret_cast<void *>(remoteAddr2), ret);
            return -1;
        }

        char value2[MAX_ENGINE_NAME_LEN] = {};
        CHECK_ACL(aclrtMemcpy(value2, MAX_ENGINE_NAME_LEN, buffer2, strlen(remoteEngine), ACL_MEMCPY_DEVICE_TO_HOST));
        printf("[INFO] TransferSync read success, remoteAddr:%p, value = %s\n",
               reinterpret_cast<void *>(remoteAddr2), value2);
        if (std::string(remoteEngine) != value2) {
            printf("[ERROR] Failed to check read value:%s, expect:%s\n", value, remoteEngine);
            return -1;
        } else {
            printf("[INFO] Check read value success\n");
        }
    }
    return 0;
}

void Finalize(AdxlEngine &adxlEngine, bool connected, const char *remoteEngine,
              const std::vector<MemHandle> handles)
{
    if (connected) {
        auto ret = adxlEngine.Disconnect(remoteEngine);
        if (ret != 0) {
            printf("[ERROR] Disconnect failed, ret = %d\n", ret);
        } else {
            printf("[INFO] Disconnect success\n");
        }
        // 等待对端写disconnect完成, 销毁本地链路后进行解注册
        std::this_thread::sleep_for(std::chrono::seconds(WAIT_TIME));
    }

    for (auto handle : handles) {
        auto ret = adxlEngine.DeregisterMem(handle);
        if (ret != 0) {
            printf("[ERROR] DeregisterMem failed, ret = %u\n", ret);
        } else {
            printf("[INFO] DeregisterMem success\n");
        }
    }
    adxlEngine.Finalize();
}

int32_t Run(const char *localEngine, const char *remoteEngine)
{
    printf("[INFO] run start\n");
    // 1. 初始化
    AdxlEngine adxlEngine;
    if (Initialize(adxlEngine, localEngine) != 0) {
        printf("[ERROR] Initialize AdxlEngine failed\n");
        return -1;
    }
    // 2. 注册内存地址
    uint8_t *buffer = nullptr;  // 用于write
    CHECK_ACL(aclrtMalloc((void **)&buffer, MAX_ENGINE_NAME_LEN, ACL_MEM_MALLOC_HUGE_ONLY));
    uint8_t *buffer2 = nullptr; // 用于read
    CHECK_ACL(aclrtMalloc((void **)&buffer2, MAX_ENGINE_NAME_LEN, ACL_MEM_MALLOC_HUGE_ONLY));

    MemDesc desc{};
    desc.addr = reinterpret_cast<uintptr_t>(buffer);
    desc.len = MAX_ENGINE_NAME_LEN;
    MemHandle handle = nullptr;
    bool connected = false;
    auto ret = adxlEngine.RegisterMem(desc, MEM_DEVICE, handle);
    if (ret != SUCCESS) {
        printf("[ERROR] RegisterMem failed, ret = %u\n", ret);
        Finalize(adxlEngine, connected, remoteEngine, {});
        return -1;
    }
    MemHandle handle2 = nullptr;
    desc.addr = reinterpret_cast<uintptr_t>(buffer2);
    ret = adxlEngine.RegisterMem(desc, MEM_DEVICE, handle2);
    if (ret != SUCCESS) {
        printf("[ERROR] RegisterMem failed, ret = %u\n", ret);
        Finalize(adxlEngine, connected, remoteEngine, {handle});
        return -1;
    }
    // RegisterMem成功后，将地址保存到本地文件中等待对端读取
    printf("[INFO] RegisterMem success, addr:%p, add2:%p\n", buffer, buffer2);
    std::ofstream tmp_file(localEngine);  // 默认就是 std::ios::out | std::ios::trunc
    if (tmp_file) {
        tmp_file << reinterpret_cast<void *>(buffer) << " " << reinterpret_cast<void *>(buffer2) << std::endl;
    }

    // 等待对端server注册完成
    std::this_thread::sleep_for(std::chrono::seconds(WAIT_TIME));

    // 3. 与对端server建链
    if (Connect(adxlEngine, remoteEngine) != 0) {
        Finalize(adxlEngine, connected, remoteEngine, {handle, handle2});
        return -1;
    }
    connected = true;

    // 4. 测试d2d write和read
    if (Transfer(adxlEngine, buffer, buffer2, localEngine, remoteEngine) != 0) {
        Finalize(adxlEngine, connected, remoteEngine, {handle, handle2});
        return -1;
    }

    // 5. 释放Cache与llmDataDist
    Finalize(adxlEngine, connected, remoteEngine, {handle, handle2});
    printf("[INFO] run Sample end\n");
    return 0;
}

int main(int32_t argc, char **argv)
{
    std::string deviceId;
    std::string localEngine;
    std::string remoteEngine;
    if (argc == EXPECTED_ARG_CNT) {
        deviceId = argv[ARG_INDEX_DEVICE_ID];
        localEngine = argv[ARG_INDEX_LOCAL_ENGINE];
        remoteEngine = argv[ARG_INDEX_REMOTE_ENGINE];
        printf("[INFO] deviceId = %s, localEngine = %s, remoteEngine = %s\n",
               deviceId.c_str(), localEngine.c_str(), remoteEngine.c_str());
    } else {
        printf("[ERROR] expect 3 args(deviceId, localEngine, remoteEngine), but got %d\n", argc - 1);
        return -1;
    }
    int32_t device = std::stoi(deviceId);
    CHECK_ACL(aclrtSetDevice(device));
    int32_t ret = Run(localEngine.c_str(), remoteEngine.c_str());
    CHECK_ACL(aclrtResetDevice(device));
    return ret;
}