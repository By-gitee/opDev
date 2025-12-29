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

// 本例介绍自定义pass修改子图结构需要使用到的接口之一"GetALLSubgraphs"，使用此接口可获取到节点所在根图的所有子图，
// 然后调用AddAbsNodeInSubgraph接口修改子图结构
graphStatus ModifySubgraphPass(GraphPtr &graph, CustomPassContext &custom_context) {
    cout << "ModifySubgraphPass begin." << endl;
    // 1.获取本图中的所有节点，不包含本图内子图的节点
    auto nodes = graph->GetDirectNode();
    if (nodes.empty()) {
        cout << "Graph has no node." << endl;
        return GRAPH_SUCCESS;
    }

    // 2.通过单个节点获取本图的所有子图
    vector<GraphPtr> graph_list;
    auto ret = nodes[0].GetALLSubgraphs(graph_list);
    if (ret != GRAPH_SUCCESS) {
        cout << "Get all subgraphs failed." << endl;
        return GRAPH_SUCCESS;
    }
    if (!graph_list.empty()) {
        cout << "Graph has " << graph_list.size() << " subgraphs." << endl;
    } else {
        cout << "Graph has no subgraph." << endl;
        return GRAPH_SUCCESS;
    }

    // 3.遍历所有子图，调用AddAbsNodeInSubgraph接口修改子图结构
    for (auto &subgraph : graph_list) {
        ret = AddAbsNodeInSubgraph(subgraph, custom_context);
        if (ret != GRAPH_SUCCESS) {
            custom_context.SetErrorMessage("Add abs node in subgraph failed.");
            return -1;
        }
    }

    cout << "ModifySubgraphPass end." << endl;
    return GRAPH_SUCCESS;
}

REGISTER_CUSTOM_PASS("ModifySubgraphPass").CustomPassFn(ModifySubgraphPass);