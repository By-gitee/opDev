/**
* @file common.cpp
*
* Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#include "common.h"
#include <fstream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

extern bool g_isDevice;
bool ReadFile(const std::string &filePath, size_t fileSize, void *buffer, size_t bufferSize)
{
    struct stat sBuf;
    int fileStatus = stat(filePath.data(), &sBuf);
    if (fileStatus == -1) {
        ERROR_LOG("failed to get file %s", filePath.c_str());
        return false;
    }
    if (S_ISREG(sBuf.st_mode) == 0) {
        ERROR_LOG("%s is not a file, please enter a file", filePath.c_str());
        return false;
    }

    std::ifstream file;
    file.open(filePath, std::ios::binary);
    if (!file.is_open()) {
        ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    std::filebuf *buf = file.rdbuf();
    size_t size = buf->pubseekoff(0, std::ios::end, std::ios::in);
    if (size == 0) {
        ERROR_LOG("file size is 0");
        file.close();
        return false;
    }
    if (size > bufferSize) {
        ERROR_LOG("file size is larger than buffer size");
        file.close();
        return false;
    }
    buf->pubseekpos(0, std::ios::in);
    buf->sgetn(static_cast<char *>(buffer), size);
    fileSize = size;
    file.close();
    return true;
}

bool WriteFile(const std::string &filePath, const void *buffer, size_t size)
{
    if (buffer == nullptr) {
        ERROR_LOG("Write file failed. buffer is nullptr");
        return false;
    }

    int fd = open(filePath.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWRITE);
    if (fd < 0) {
        ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    auto writeSize = write(fd, buffer, size);
    (void) close(fd);
    if (writeSize != size) {
        ERROR_LOG("Write file Failed.");
        return false;
    }

    return true;
}

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shape_size = 1;
    for (auto i : shape) {
      shape_size *= i;
    }
    return shape_size;
}

int CreateAclTensor(const int64_t hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclFormat dataFormat, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * DT2LONG[dataType];

    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);

    ret = aclrtMemcpy(*deviceAddr, size, &hostData, size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
      strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, dataFormat,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int CreateAclTensor(const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclFormat dataFormat, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * DT2LONG[dataType];
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
      strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, dataFormat,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int CreateAclTensorWithData(const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclFormat dataFormat, aclTensor** tensor) {
    auto size = GetShapeSize(shape) * DT2LONG[dataType];
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; i--) {
      strides[i] = shape[i + 1] * strides[i + 1];
    }
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, dataFormat,
                              shape.data(), shape.size(), *deviceAddr);
    return 0;
}

int ReadDataToDevice(const std::string dataPath, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclFormat dataFormat) {
    auto size = GetShapeSize(shape) * DT2LONG[dataType];
    void* hostData = nullptr;
    if (g_isDevice) {
        auto ret = aclrtMalloc(&hostData, size, ACL_MEM_MALLOC_NORMAL_ONLY);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    } else {
        auto ret = aclrtMallocHost(&hostData, size);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    }
    size_t fileSize = 0;
    auto retRead = ReadFile(dataPath, fileSize, reinterpret_cast<void *>(hostData), size);
    CHECK_RET(retRead == true, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", retRead); return !retRead);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
    ret = aclrtMemcpy(*deviceAddr, size, hostData, size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);
    if (g_isDevice) {
        auto ret = aclrtFree(hostData);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtFree failed. ERROR: %d\n", ret); return ret);
        hostData = nullptr;
    } else {
        auto ret = aclrtFreeHost(hostData);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtFreeHost failed. ERROR: %d\n", ret); return ret);
        hostData = nullptr;
    }
    return 0;
}