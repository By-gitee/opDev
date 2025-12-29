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

#include <string>
#include <vector>

#include "llm_config.h"
namespace llm {
namespace {
constexpr const char *kFullInputShapes = "fullInputShapes";
constexpr const char *kFullInputDataTypes = "fullInputDataTypes";
constexpr const char *kFullInputFiles = "fullInputFiles";

constexpr const char *kIncInputShapes = "incInputShapes";
constexpr const char *kIncInputDataTypes = "incInputDataTypes";
constexpr const char *kIncInputFiles = "incInputFiles";

constexpr const char *kKvTensorShapes = "kvShapes";
constexpr const char *kKvTensorDataTypes = "kvDataTypes";
constexpr const char *kKvTensorNumber = "kvNum";

constexpr const char *kFullLoopNumber = "fullLoopNumber";
constexpr const char *kIncLoopNumber = "incLoopNumber";

constexpr const char *kGroupNameList = "groupNameList";
constexpr const char *kSocVersion = "socVersion";
constexpr const char *kRankTableFile = "rankTableFile";
constexpr const char *kCacheDir = "cacheDir";
constexpr const char *kGraphKey = "graphKey";
}

LlmConfig &LlmConfig::Instance() {
  static LlmConfig instance;
  return instance;
}

// get groupName socVersion rankTableFile and cacheDir graphKey
bool LlmConfig::ParseCommonConfig(const std::map<std::string, std::string> &cfgKeyValueMap) {
    if(!CommonUtils::CheckAndGetConfigItem(cfgKeyValueMap, kGroupNameList, groupNameList_)) {
        ERROR_LOG("Can not get %s from config file", kGroupNameList);
        return false;
    }
    if(!CommonUtils::CheckAndGetConfigItem(cfgKeyValueMap, kSocVersion, socVersion_)) {
        ERROR_LOG("Can not get %s from config file", kSocVersion);
        return false;
    }
    std::string inputRankTableFile;
    if(!CommonUtils::CheckAndGetConfigItem(cfgKeyValueMap, kRankTableFile, inputRankTableFile)) {
        ERROR_LOG("Can not get %s from config file", kRankTableFile);
        return false;
    }
    rankTableFile_ = CommonUtils::GetRealPath(inputRankTableFile);
    if (rankTableFile_.empty()) {
        ERROR_LOG("Get rank table file path failed");
        return false;
    }
    enableCache_ = CommonUtils::CheckAndGetConfigItem(cfgKeyValueMap, kCacheDir, cacheDir_) &&
                   CommonUtils::CheckAndGetConfigItem(cfgKeyValueMap, kGraphKey, graphKey_);
    return true;
}

// get loop number
bool LlmConfig::ParseLoopNum(const std::map<std::string, std::string> &cfgKeyValueMap) {
    std::string loopNum;
    if(!CommonUtils::CheckAndGetConfigItem(cfgKeyValueMap, kFullLoopNumber, loopNum)) {
        ERROR_LOG("Can not get %s from config file", kFullLoopNumber);
        return false;
    }
    int32_t loopNumDigit;
    if (!CommonUtils::ConvertStrToInt32(loopNum, loopNumDigit)) {
        ERROR_LOG("Convert full graph loop num from string %s to int failed", loopNum.c_str());
        return false;
    }
    if (loopNumDigit < 0) {
        ERROR_LOG("Full graph loop number must be a positive digit.");
        return false;
    }
    fullGraphLoopNum_ = static_cast<uint32_t>(loopNumDigit);

    if(!CommonUtils::CheckAndGetConfigItem(cfgKeyValueMap, kIncLoopNumber, loopNum)) {
        ERROR_LOG("Can not get %s from config file", kIncLoopNumber);
        return false;
    }
    if (!CommonUtils::ConvertStrToInt32(loopNum, loopNumDigit)) {
        ERROR_LOG("Convert increment graph loop num from string %s to int failed", loopNum.c_str());
        return false;
    }
    if (loopNumDigit < 0) {
        ERROR_LOG("Increase graph loop number must be a positive digit.");
        return false;
    }
    incGraphLoopNum_ = static_cast<uint32_t>(loopNumDigit);
    return true;
} 

// get full graph input desc
bool LlmConfig::ParseFullGraphInputDesc(const std::map<std::string, std::string> &cfgKeyValueMap) {
    std::string allShapeStr;
    if(!CommonUtils::CheckAndGetConfigItem(cfgKeyValueMap, kFullInputShapes, allShapeStr)) {
        ERROR_LOG("Can not get %s from config file", kFullInputShapes);
        return false;
    }
    if(!CommonUtils::ParseShapes(allShapeStr, fullGraphInputDesc_.shapes)) {
        ERROR_LOG("Parse full graph input shapes failed. Original shape string[%s].", allShapeStr.c_str());
        return false;
    }

    std::string allDataTypeStr;
    if(!CommonUtils::CheckAndGetConfigItem(cfgKeyValueMap, kFullInputDataTypes, allDataTypeStr)) {
        ERROR_LOG("Can not get %s from config file", kFullInputDataTypes);
        return false;
    }
    if(!CommonUtils::ParseDataTypes(allDataTypeStr, fullGraphInputDesc_.dataTypes)) {
        ERROR_LOG("Parse full graph input shapes failed. Original shape string[%s].", allDataTypeStr.c_str());
        return false;
    }

    std::string allInputFilesStr;
    if(!CommonUtils::CheckAndGetConfigItem(cfgKeyValueMap, kFullInputFiles, allInputFilesStr)) {
        ERROR_LOG("Can not get %s from config file", kFullInputFiles);
        return false;
    }
    if(!CommonUtils::ParseInputFiles(allInputFilesStr, fullGraphInputDesc_.dataFiles)) {
        ERROR_LOG("Parse full graph input shapes failed. Original shape string[%s].", allInputFilesStr.c_str());
        return false;
    }
    return true;
}

// get inc graph input desc
bool LlmConfig::ParseIncGraphInputDesc(const std::map<std::string, std::string> &cfgKeyValueMap) {
    std::string allShapeStr;
    if(!CommonUtils::CheckAndGetConfigItem(cfgKeyValueMap, kIncInputShapes, allShapeStr)) {
        ERROR_LOG("Can not get %s from config file", kIncInputShapes);
        return false;
    }
    if(!CommonUtils::ParseShapes(allShapeStr, incGraphInputDesc_.shapes)) {
        ERROR_LOG("Parse increment graph input shapes failed. Original shape string[%s].", allShapeStr.c_str());
        return false;
    }

    std::string allDataTypeStr;
    if(!CommonUtils::CheckAndGetConfigItem(cfgKeyValueMap, kIncInputDataTypes, allDataTypeStr)) {
        ERROR_LOG("Can not get %s from config file", kIncInputDataTypes);
        return false;
    }
    if(!CommonUtils::ParseDataTypes(allDataTypeStr, incGraphInputDesc_.dataTypes)) {
        ERROR_LOG("Parse increment graph input shapes failed. Original shape string[%s].", allDataTypeStr.c_str());
        return false;
    }

    std::string allInputFilesStr;
    if(!CommonUtils::CheckAndGetConfigItem(cfgKeyValueMap, kIncInputFiles, allInputFilesStr)) {
        ERROR_LOG("Can not get %s from config file", kIncInputFiles);
        return false;
    }
    if(!CommonUtils::ParseInputFiles(allInputFilesStr, incGraphInputDesc_.dataFiles)) {
        ERROR_LOG("Parse increment graph input shapes failed. Original shape string[%s].", allInputFilesStr.c_str());
        return false;
    }
    return true;
}

// get kv tensor desc and number
bool LlmConfig::ParseKvInputDesc(const std::map<std::string, std::string> &cfgKeyValueMap) {
    std::string ShapeStr;
    if(!CommonUtils::CheckAndGetConfigItem(cfgKeyValueMap, kKvTensorShapes, ShapeStr)) {
        ERROR_LOG("Can not get %s from config file", kKvTensorShapes);
        return false;
    }
    if(!CommonUtils::ParseShape(ShapeStr, kvDesc_.shape)) {
        ERROR_LOG("Parse kv input shapes failed. Original shape string[%s].", ShapeStr.c_str());
        return false;
    }

    std::string dataTypeStr;
    if(!CommonUtils::CheckAndGetConfigItem(cfgKeyValueMap, kKvTensorDataTypes, dataTypeStr)) {
        ERROR_LOG("Can not get %s from config file", kKvTensorDataTypes);
        return false;
    }
    if(!CommonUtils::ParseDataType(dataTypeStr, kvDesc_.dataType)) {
        ERROR_LOG("Parse kv graph input shapes failed. Original shape string[%s].", dataTypeStr.c_str());
        return false;
    }

    std::string kvNumStr;
    if(!CommonUtils::CheckAndGetConfigItem(cfgKeyValueMap, kKvTensorNumber, kvNumStr)) {
        ERROR_LOG("Can not get %s from config file", kKvTensorNumber);
        return false;
    }
    int32_t kvNumDigit;
    if (!CommonUtils::ConvertStrToInt32(kvNumStr, kvNumDigit)) {
        ERROR_LOG("Convert kv tensor num from string %s to int failed", kvNumStr.c_str());
        return false;
    }
    if (kvNumDigit < 0) {
        ERROR_LOG("Kv tensor number must be a positive digit.");
        return false;
    }
    kvDesc_.kvNum = static_cast<uint32_t>(kvNumDigit);
    return true;
}

bool LlmConfig::GenerateLlmConfig(const std::string &configFile) {
    const std::string configPath = CommonUtils::GetRealPath(configFile);
    if (configPath.empty()) {
        ERROR_LOG("Failed to get real path of config file: %s", configFile.c_str());
        return false;
    }
    std::map<std::string, std::string> cfgKeyValueMap;
    if(!CommonUtils::ParseConfig(configPath, cfgKeyValueMap)) {
        ERROR_LOG("Failed to parse config file: %s", configFile.c_str());
        return false;
    }
    if (!ParseFullGraphInputDesc(cfgKeyValueMap)) {
        ERROR_LOG("Parse full graph input desc failed");
        return false;
    }
    if (!ParseIncGraphInputDesc(cfgKeyValueMap)) {
        ERROR_LOG("Parse increment graph input desc failed");
        return false;
    }
    if (!ParseKvInputDesc(cfgKeyValueMap)) {
        ERROR_LOG("Parse kv input tensor desc input desc failed");
        return false;
    }
    if (!ParseLoopNum(cfgKeyValueMap)) {
        ERROR_LOG("Parse loop number config failed");
        return false;
    }
    if (!ParseCommonConfig(cfgKeyValueMap)) {
        ERROR_LOG("Parse group name list, soc version and rank table file config failed");
        return false;
    }
    return true;
}
}