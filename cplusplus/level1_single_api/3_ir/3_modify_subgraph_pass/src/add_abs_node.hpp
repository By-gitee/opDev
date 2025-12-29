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

#ifndef ADD_ABS_NODE_HPP_
#define ADD_ABS_NODE_HPP_

#include <iostream>

#include "all_ops.h"
#include "register_custom_pass.h"

namespace pass {
constexpr const char *kOpTypeData = "Data";
constexpr const char *kOpTypFrameworkOp = "FrameworkOp";
int32_t kCount = 0;

#define CHECK_STATUS(exp, msg)                             \
    do {                                                   \
        if ((exp) != ge::GRAPH_SUCCESS) {                  \
            std::cout << "Check (" << #exp << ") failed";  \
            if (std::string(msg).length() > 0) {           \
                std::cout << ", error message: " << (msg); \
            }                                              \
            std::cout << std::endl;                        \
            return ge::GRAPH_FAILED;                       \
        }                                                  \
    } while (0)

ge::graphStatus InsertAbsNode(const ge::GraphPtr &graph, ge::GNode &src_node, const int32_t src_idx, ge::GNode &dst_node,
                              const int32_t dst_idx) {
    // 删除Data和FrameworkOp节点之间的边
    CHECK_STATUS(graph->RemoveEdge(src_node, src_idx, dst_node, dst_idx), "Remove edge failed.");
    // 在Data和FrameworkOp节点之间插入Abs节点
    std::string name = "abs_" + std::to_string(kCount++);
    auto abs = ge::op::Abs(name.c_str());
    ge::GNode node_abs = graph->AddNodeByOp(abs);
    CHECK_STATUS(graph->AddDataEdge(src_node, src_idx, node_abs, 0), "Add data edge failed between Data and Abs.");
    CHECK_STATUS(graph->AddDataEdge(node_abs, 0, dst_node, dst_idx), "Add data edge failed between Abs and FrameworkOp.");
    std::cout << "Add abs node success." << std::endl;
    return ge::GRAPH_SUCCESS;
}
// |o>-----------------------------------
// |o>      Data              Data
// |o>       |                 |
// |o>       |       ==>      Abs
// |o>       |                 |
// |o>   FrameworkOp       FrameworkOp
// |o>-----------------------------------
// pass修改子图说明：本例识别上图中左边的Data和FrameworkOp节点，并在中间插入Abs节点得到右图
ge::graphStatus AddAbsNodeInSubgraph(ge::GraphPtr &graph, ge::CustomPassContext &custom_context) {
    // 1. 获取子图中的Data和FrameworkOp节点
    auto all_nodes = graph->GetAllNodes();
    for (auto &dst_node : all_nodes) {
        ge::AscendString dst_type;
        CHECK_STATUS(dst_node.GetType(dst_type), "Failed to get the type of the dst_node.");
        if (dst_type != kOpTypFrameworkOp) {
            continue;
        }
        ge::AscendString dst_name;
        CHECK_STATUS(dst_node.GetName(dst_name), "Failed to get the name of the dst_node");
        std::cout << "Find dst node: " << dst_name.GetString() << "." << std::endl;
        // 2. 找到目标节点FrameworkOp，然后依次获取输入节点
        for (int32_t i = 0; i < dst_node.GetInputsSize(); ++i) {
            const auto src_node_and_port = dst_node.GetInDataNodesAndPortIndexs(i);
            // 3. 如果没有找到目标节点或者目标节点间无连边，跳过不改图
            if (src_node_and_port.first != nullptr) {
                ge::AscendString src_name;
                CHECK_STATUS(src_node_and_port.first->GetName(src_name), "Failed to get the name of the src_node");
                std::cout << "Find src node: " << src_name.GetString() << "." << std::endl;
                CHECK_STATUS(InsertAbsNode(graph, *src_node_and_port.first, src_node_and_port.second, dst_node, i),
                             "Failed to insert Abs node");
            }
        }
    }
    return ge::GRAPH_SUCCESS;
}
}  // namespace pass

#endif