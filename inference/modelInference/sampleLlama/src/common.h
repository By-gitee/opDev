/**
* @file common.h
*
* Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
#ifndef COMMON_H
#define COMMON_H

#include <cstdio>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <unordered_map>

#include "acl/acl.h"
#include "acl/acl_op_compiler.h"
#include "aclnn/acl_meta.h"

#define SUCCESS 0
#define FAILED 1

#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO]  " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN]  " fmt "\n", ##args)
#define ERROR_LOG(fmt, args...) fprintf(stderr, "[ERROR]  " fmt "\n", ##args)
#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

static std::unordered_map<aclDataType, size_t> DT2LONG = {
    {aclDataType::ACL_FLOAT16, sizeof(uint16_t)},
    {aclDataType::ACL_INT64, sizeof(int64_t)},
    {aclDataType::ACL_FLOAT, sizeof(float)},
    {aclDataType::ACL_UINT8, sizeof(uint8_t)}
};

static std::unordered_map<int, std::string> INT2STR = {
    {0, "0"}, {1, "1"}, {2, "2"}, {3, "3"}, {4, "4"},
    {5, "5"}, {6, "6"}, {7, "7"}, {8, "8"}, {9, "9"},
    {10, "10"}, {11, "11"}, {12, "12"}, {13, "13"}, {14, "14"},
    {15, "15"}, {16, "16"}, {17, "17"}, {18, "18"}, {19, "19"},
    {20, "20"}, {21, "21"}, {22, "22"}, {23, "23"}, {24, "24"},
    {25, "25"}, {26, "26"}, {27, "27"}, {28, "28"}, {29, "29"},
    {30, "30"}, {31, "31"}
};

bool ReadFile(const std::string &filePath, size_t fileSize, void *buffer, size_t bufferSize);

bool WriteFile(const std::string &filePath, const void *buffer, size_t size);

int64_t GetShapeSize(const std::vector<int64_t>& shape);

int CreateAclTensor(const int64_t hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclFormat dataFormat, aclTensor** tensor);

int CreateAclTensor(const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclFormat dataFormat, aclTensor** tensor);

int CreateAclTensorWithData(const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclFormat dataFormat, aclTensor** tensor);

int ReadDataToDevice(const std::string dataPath, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclFormat dataFormat);
#endif // COMMON_H

