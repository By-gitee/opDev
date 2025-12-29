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
#ifndef UTILS_H
#define UTILS_H
#include <string>
#include <vector>
#include <map>
#include "graph/tensor.h"
namespace llm {
#define INFO_LOG(fmt, args...) {fprintf(stdout, "[INFO][%s][%d]  " fmt "\n", __FILE__, __LINE__, ##args); \
                                fflush(stdout);}
#define WARN_LOG(fmt, args...) {fprintf(stdout, "[WARN][%s][%d]  " fmt "\n", __FILE__, __LINE__, ##args); \
                                fflush(stdout);}
#define ERROR_LOG(fmt, args...) {fprintf(stdout, "[ERROR][%s][%d]  " fmt "\n", __FILE__, __LINE__, ##args); \
                                fflush(stdout);}

class CommonUtils {
public:
    // Get the absolute path
    static std::string GetRealPath(const std::string &path);
    static bool ConvertStrToInt64(const std::string &str, int64_t &value);
    static bool ConvertStrToInt32(const std::string &str, int32_t &value);
    // true: overflow, false: not overflow
    static bool CheckInt64MulOverflow(const int64_t a, const int64_t b);
    static int64_t CalcTensorMemSize(const std::vector<int64_t> &shape, const ge::DataType &dataType);
    // Read input binary file to memory
    static bool ReadBinFile(const std::string &inputPath, const int64_t &dataLen, uint8_t *data);
    static std::vector<std::string> SplitString(const std::string &str, const std::string &delim);
    static bool CheckAndGetConfigItem(const std::map<std::string, std::string> &configs,
                                      const std::string &key, std::string &value);

    static bool ParseShapes(const std::string &input, std::vector<std::vector<int64_t>> &shapes);
    static bool ParseDataTypes(const std::string &input, std::vector<ge::DataType> &dataTypes);
    static bool ParseShape(const std::string &input, std::vector<int64_t> &shape);
    static bool ParseDataType(const std::string &input, ge::DataType &dataType);
    static bool ParseInputFiles(const std::string &input, std::vector<std::string> &inputFiles);

    static bool ParseConfig(const std::string &configFile, std::map<std::string, std::string> &cfgKeyValueMap);
};
}
#endif //UTILS_H