/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
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

#ifndef UTILS_H_
#define UTILS_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <string>
#include <tuple>
#include <vector>

namespace wrapper {
inline bool CheckInt64MulOverflow(int64_t a, int64_t b)
{
    if ((a == 0) || (b == 0)) {
        return false;
    }
    const auto aAbs = std::abs(a);
    const auto bAbs = std::abs(b);
    if (INT64_MAX / aAbs < bAbs) {
        return true;
    }
    return false;
}

inline std::string ComputeStrides(ssize_t itemSize, const std::vector<int64_t> &dims, std::vector<ssize_t> &strides)
{
    if (dims.empty()) {
        return "";
    }
    strides.emplace_back(itemSize);
    int64_t stride = static_cast<int64_t>(itemSize);
    for (auto it = dims.crbegin(); it != (dims.crend() - 1); it++) {
        if (CheckInt64MulOverflow(stride, *it)) {
            return std::to_string(stride) + " mul " + std::to_string(*it) + "is overflow.";
        }
        stride *= *it;
        strides.emplace_back(static_cast<ssize_t>(stride));
    }
    std::reverse(strides.begin(), strides.end());
    return "";
}
}
#endif  // UTILS_H_