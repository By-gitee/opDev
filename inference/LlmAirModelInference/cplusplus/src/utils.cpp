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

#include <iostream>
#include <fstream>
#include <cstring>
#include <sstream>
#include <limits>
#include "utils.h"
namespace llm {
namespace {
const std::map<std::string, ge::DataType> kStringToDataTypeMap = {
    {"DT_FLOAT", ge::DT_FLOAT}, {"DT_FLOAT16", ge::DT_FLOAT16},
    {"DT_INT8", ge::DT_INT8}, {"DT_INT16", ge::DT_INT16},
    {"DT_UINT16", ge::DT_UINT16}, {"DT_UINT8", ge::DT_UINT8},
    {"DT_INT32", ge::DT_INT32}, {"DT_INT64", ge::DT_INT64},
    {"DT_UINT32", ge::DT_UINT32}, {"DT_UINT64", ge::DT_UINT64},
    {"DT_BOOL", ge::DT_BOOL}, {"DT_DOUBLE", ge::DT_DOUBLE}
};
const std::map<ge::DataType, uint32_t> kDataTypeToSizeMap = {
    {ge::DT_FLOAT, sizeof(float)}, {ge::DT_FLOAT16, sizeof(int16_t)},
    {ge::DT_INT8, sizeof(int8_t)}, {ge::DT_INT16, sizeof(int16_t)},
    {ge::DT_UINT16, sizeof(uint16_t)}, {ge::DT_UINT8, sizeof(uint8_t)},
    {ge::DT_INT32, sizeof(int32_t)}, {ge::DT_INT64, sizeof(int64_t)},
    {ge::DT_UINT32, sizeof(uint32_t)}, {ge::DT_UINT64, sizeof(uint64_t)},
    {ge::DT_BOOL, sizeof(bool)}, {ge::DT_DOUBLE, sizeof(double)}
};
constexpr uint32_t kPathMax = 1024U;
}

std::string CommonUtils::GetRealPath(const std::string &path) {
    if (path.empty()) {
        INFO_LOG("Path string is empty.");
        return "";
    }
    if (path.size() >= kPathMax) {
        INFO_LOG("File path %s is too long.", path.c_str());
        return "";
    }
    char resolvedPath[kPathMax] = {0};
    std::string realPath;
    if (realpath(path.c_str(), resolvedPath) == nullptr) {
        INFO_LOG("Get real path failed, path: %s", path.c_str());
        return "";
    } else {
        realPath = resolvedPath;
    }
    return realPath;
}

bool CommonUtils::ReadBinFile(const std::string &inputPath, const int64_t &dataLen, uint8_t *data) {
    if (data == nullptr) {
        ERROR_LOG("Data pointer is nullptr.");
        return false;
    }
    std::ifstream file(inputPath.c_str(), std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        ERROR_LOG("Open file %s failed.", inputPath.c_str());
        return false;
    }
    file.seekg(0, std::ios::end);
    std::istream::pos_type fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    if (fileSize != dataLen) {
        ERROR_LOG("File size %ld is not equal to data length %ld.", fileSize, dataLen);
        file.close();
        return false;
    }
    file.read(reinterpret_cast<char *>(data), fileSize);
    file.close();
    return true;
}

bool CommonUtils::ConvertStrToInt64(const std::string &str, int64_t &value) {
    try {
        value = std::stol(str);
    } catch (std::invalid_argument &) {
        ERROR_LOG("Digit str:%s is invalid", str.c_str());
        return false;
    } catch (std::out_of_range &) {
        ERROR_LOG("Convert string:%s to int64 failed result of out of range.", str.c_str());
        return false;
    }
    return true;
}

bool CommonUtils::ConvertStrToInt32(const std::string &str, int32_t &value) {
    try {
        value = std::stoi(str);
    } catch (std::invalid_argument &) {
        ERROR_LOG("Digit str:%s is invalid", str.c_str());
        return false;
    } catch (std::out_of_range &) {
        ERROR_LOG("Convert string:%s to int32 failed result of out of range.", str.c_str());
        return false;
    }
    return true;
}

bool CommonUtils::CheckInt64MulOverflow(const int64_t a, const int64_t b) {
  if (a > 0) {
    if (b > 0) {
      if (a > (std::numeric_limits<int64_t>::max() / b)) {
        return true;
      }
    } else {
      if (b < (std::numeric_limits<int64_t>::min() / a)) {
        return true;
      }
    }
  } else {
    if (b > 0) {
      if (a < (std::numeric_limits<int64_t>::min() / b)) {
        return true;
      }
    } else {
      if ((a != 0) && (b < (std::numeric_limits<int64_t>::max() / a))) {
        return true;
      }
    }
  }
  return false;
}

int64_t CommonUtils::CalcTensorMemSize(const std::vector<int64_t> &shape, const ge::DataType &dataType) {
    int64_t memSize = 1;
    for (auto dim : shape) {
        if (CheckInt64MulOverflow(memSize, dim)) {
            ERROR_LOG("The shape of tensor is too large, memSize:%ld, dim:%ld", memSize, dim);
            return -1;
        }
        memSize *= dim;
    }
    const auto iter = kDataTypeToSizeMap.find(dataType);
    if (iter == kDataTypeToSizeMap.cend()) {
        ERROR_LOG("Unsupported data type:%d", dataType);
        return -1;
    }
    if (CheckInt64MulOverflow(memSize, iter->second)) {
        ERROR_LOG("The shape of tensor mul data type is too large, memSize:%ld, dataType:%d", memSize, dataType);
        return -1;
    }
    return memSize * iter->second;
}

std::vector<std::string> CommonUtils::SplitString(const std::string &str, const std::string &delim) {
    std::vector<std::string> result;
    if (str.empty()) {
        return result;
    }
    std::string strWithDelim = str + delim;
    size_t pos = strWithDelim.find(delim);
    size_t size = strWithDelim.size();
    while (pos != std::string::npos) {
        std::string subStr = strWithDelim.substr(0, pos);
        result.push_back(subStr);
        strWithDelim = strWithDelim.substr(pos + delim.size(), size);
        pos = strWithDelim.find(delim);
    }
    return result;
}

bool CommonUtils::ParseShapes(const std::string &input, std::vector<std::vector<int64_t>> &shapes) {
    std::vector<std::string> splitVec = SplitString(input, ";");
    for (const auto &shape : splitVec) {
        std::vector<int64_t> shapeVec;
        if (shape == "0") {
            shapes.emplace_back(shapeVec);
            INFO_LOG("Get empty shape.");
            continue;
        }
        std::vector<std::string> shapeStrVec = SplitString(shape, ",");
        for (const auto &shapeStr : shapeStrVec) {
            int64_t dimValue = -1;
            if (ConvertStrToInt64(shapeStr, dimValue)) {
                shapeVec.emplace_back(dimValue);
            } else {
                ERROR_LOG("Convert string to int64 failed.");
                return false;
            }
        }
        shapes.emplace_back(shapeVec);
    }
    return true;
}

bool CommonUtils::ParseDataTypes(const std::string &input, std::vector<ge::DataType> &dataTypes) {
    std::vector<std::string> splitVec = SplitString(input, ";");
    for (const auto &dataType : splitVec) {
        const auto iter = kStringToDataTypeMap.find(dataType);
        if (iter == kStringToDataTypeMap.cend()) {
            ERROR_LOG("Data type %s is not supported.", dataType.c_str());   
            return false;
        }
        dataTypes.emplace_back(iter->second);
    }
    return true;
}

bool CommonUtils::ParseShape(const std::string &input, std::vector<int64_t> &shape) {
    std::vector<std::string> splitVec = SplitString(input, ",");
    for (const auto &shapeStr : splitVec) {
        int64_t dimValue = -1;
        if (ConvertStrToInt64(shapeStr, dimValue)) {
            shape.emplace_back(dimValue);
        } else {
            ERROR_LOG("Convert string to int64 failed.");
            return false;
        }
    }
    return true;
}

bool CommonUtils::ParseDataType(const std::string &input, ge::DataType &dataType) {
    const auto iter = kStringToDataTypeMap.find(input);
    if (iter == kStringToDataTypeMap.cend()) {
        ERROR_LOG("Data type %s is not supported.", input.c_str());   
        return false;
    }
    dataType = iter->second;
    return true;
}

bool CommonUtils::ParseInputFiles(const std::string &input, std::vector<std::string> &inputFiles) {
    std::vector<std::string> splitVec = SplitString(input, ";");
    for (const auto &file : splitVec) {
        std::string inputFile = GetRealPath(file);
        if (inputFile.empty()) {
            ERROR_LOG("Input file[%s] is not existed.", file.c_str());
            return false;
        }
        INFO_LOG("Input data file is [%s].", file.c_str());
        inputFiles.emplace_back(inputFile);
    }
    return true;
}

bool CommonUtils::ParseConfig(const std::string &configFile,
                              std::map<std::string, std::string> &cfgKeyValueMap) {
    std::ifstream configFileStream(configFile);
    if (!configFileStream.is_open()) {
        ERROR_LOG("Open config file[%s] failed.", configFile.c_str());
        return false;
    }
    std::string configLine;
    std::vector<std::string> splitVec;
    while (std::getline(configFileStream, configLine)) {
        if ((!configLine.empty()) && (configLine.back() == '\r')) {
            configLine.pop_back();
        }
        splitVec.push_back(configLine);
        INFO_LOG("Get config line[%s]", configLine.c_str());
    }
    configFileStream.close();
    const uint32_t pairNum = 2U;
    for (const auto &configLine : splitVec) {
        std::vector<std::string> innerSplitVec = SplitString(configLine, "=");
        if (innerSplitVec.size() != pairNum) {
            ERROR_LOG("Parse config line[%s] failed.", configLine.c_str());
            return false;
        }
        cfgKeyValueMap[innerSplitVec[0]] = innerSplitVec[1];
    }
    return true;
}

bool CommonUtils::CheckAndGetConfigItem(const std::map<std::string, std::string> &configs,
                                        const std::string &key, std::string &value) {
    const auto iter = configs.find(key);
    if (iter == configs.cend()) {
        return false;
    }
    value = iter->second;
    return true;
}
}