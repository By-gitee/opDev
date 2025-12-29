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

#include "add_abs_node.hpp"

using namespace std;
using namespace ge;
using namespace pass;

namespace {
constexpr const char *kOpName = "cond";
constexpr int32_t kSubgraphNum = 2;
} // namespace

// 本例介绍自定义pass修改子图结构需要使用到的接口之一"GetSubgraph"，使用此接口可获取到当前节点的所有子图，
// 然后调用AddAbsNodeInSubgraph接口修改子图结构
graphStatus ModifySubgraphPass(GraphPtr &graph, CustomPassContext &custom_context) {
    cout << "ModifySubgraphPass begin." << endl;
    // 1.获取本图中的所有节点，不包含本图内子图的节点
    for (auto &node : graph->GetDirectNode()) {
        AscendString node_name;
        auto ret = node.GetName(node_name);
        if (ret != GRAPH_SUCCESS) {
            custom_context.SetErrorMessage("Get node name failed.");
            return -1;
        }
        // 2. 找到cond节点
        if (node_name == kOpName) {
            cout << "Find cond node." << endl;
            // 3. 获取cond节点的所有子图
            for (int32_t i = 0; i < kSubgraphNum; i++) {
                GraphPtr subgraph;
                ret = node.GetSubgraph(i, subgraph);
                if (ret != GRAPH_SUCCESS) {
                    cout << "Get subgraph failed." << endl;
                    return GRAPH_SUCCESS;
                }
                // 4. 调用AddAbsNodeInSubgraph接口修改子图结构
                ret = AddAbsNodeInSubgraph(subgraph, custom_context);
                if (ret != GRAPH_SUCCESS) {
                    custom_context.SetErrorMessage("Add abs node in subgraph failed.");
                    return -1;
                }
            }
        }
    }
    cout << "ModifySubgraphPass end." << endl;
    return GRAPH_SUCCESS;
}

REGISTER_CUSTOM_PASS("ModifySubgraphPass").CustomPassFn(ModifySubgraphPass);