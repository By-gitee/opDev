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

#ifndef LLM_CONFIG_H
#define LLM_CONFIG_H
#include <string>
#include <vector>
#include "utils.h"
namespace llm {
struct InputDataDesc {
    std::vector<std::vector<int64_t>> shapes;
    std::vector<ge::DataType> dataTypes;
    std::vector<std::string> dataFiles;
};
struct KvDataDesc {
    std::vector<int64_t> shape;
    ge::DataType dataType;
    uint32_t kvNum = 0U;
};

class LlmConfig {
public:
    static LlmConfig &Instance();
    bool GenerateLlmConfig(const std::string &configFile);

    const std::string &GetGroupNameList() const { return groupNameList_; }
    const std::string &GetSocVersion() const { return socVersion_; }
    const std::string &GetRankTableFile() const { return rankTableFile_; }
    bool CacheEnable() const { return enableCache_; }
    const std::string &GetCacheDir() const { return cacheDir_; }
    const std::string &GetGraphKey() const { return graphKey_; }
    uint32_t GetFullGraphLoopNum() { return fullGraphLoopNum_; }
    uint32_t GetIncGraphLoopNum() { return incGraphLoopNum_; }
    const InputDataDesc &GetFullGraphInputDesc() const { return fullGraphInputDesc_; }
    const InputDataDesc &GetIncGraphInputDesc() const { return incGraphInputDesc_; }
    const KvDataDesc &GetKvDataDesc() const { return kvDesc_; }
private:
    LlmConfig() = default;
    ~LlmConfig() = default;

    bool ParseCommonConfig(const std::map<std::string, std::string> &cfgKeyValueMap);
    bool ParseLoopNum(const std::map<std::string, std::string> &cfgKeyValueMap);
    bool ParseFullGraphInputDesc(const std::map<std::string, std::string> &cfgKeyValueMap);
    bool ParseIncGraphInputDesc(const std::map<std::string, std::string> &cfgKeyValueMap);
    bool ParseKvInputDesc(const std::map<std::string, std::string> &cfgKeyValueMap);

    std::string groupNameList_;
    std::string socVersion_;
    std::string rankTableFile_;
    bool enableCache_ = false;
    std::string cacheDir_;
    std::string graphKey_;
    uint32_t fullGraphLoopNum_ = 1U;
    uint32_t incGraphLoopNum_ = 1U;
    InputDataDesc fullGraphInputDesc_ = {};
    InputDataDesc incGraphInputDesc_ = {};
    KvDataDesc kvDesc_ = {};
};
}
#endif // LLM_CONFIG_H