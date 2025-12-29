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
#include <cstring>
#include <unordered_set>
#include <algorithm>
#include "register_custom_pass.h"
#include "all_ops.h"

using namespace std;
using namespace ge;

namespace {
constexpr const char *kInputX = "x";
constexpr const char *kOutputY = "y";
constexpr const char *kInputMultiples = "multiples";
constexpr const char *kAttrConcatDim = "concat_dim";
constexpr const char *kInputConcatDim = "concat_dim";
constexpr const char *kOpTypeConcatD = "ConcatD";
constexpr const char *kOpTypeConcat = "Concat";
constexpr const char *kOpTypeConcatV2 = "ConcatV2";
constexpr const char *kOpTypeTile = "Tile";
constexpr const char *kOpNameTile1 = "tile1";
constexpr const char *kOpNameConcat1 = "concat1";
constexpr const char *kOpNameConcat2 = "concat2";
GraphPtr graph = nullptr;

GNode CreateConcat1Node(const vector<GNodePtr> &tiles, const int32_t concat_axis) {
    op::ConcatD concat1(kOpNameConcat1);
    concat1.set_attr_concat_dim(concat_axis);
    concat1.set_attr_N(tiles.size());
    concat1.create_dynamic_input_x(tiles.size());
    auto concat1_node = graph->AddNodeByOp(concat1);
    vector<int32_t> concat1_indexs;
    graphStatus ret = concat1_node.GetDynamicInputIndexesByName(kInputX, concat1_indexs);
    for (size_t i = 0; i < tiles.size(); ++i) {
        int32_t tile_in_index;
        ret = tiles[i]->GetInputIndexByName(kInputX, tile_in_index);
        TensorDesc tile_in_desc;
        ret = tiles[i]->GetInputDesc(tile_in_index, tile_in_desc);

        auto [input_node, input_index] = tiles[i]->GetInDataNodesAndPortIndexs(tile_in_index);
        int32_t concat1_in_index = concat1_indexs[i];
        ret = graph->AddDataEdge(*input_node, input_index, concat1_node, concat1_in_index);
        ret = concat1_node.UpdateInputDesc(concat1_in_index, tile_in_desc);
    }
    return concat1_node;
}

GNode CreateTile1Node(GNode &concat1_node, const vector<GNodePtr> &tiles) {
    op::Tile tile1(kOpNameTile1);
    auto tile1_node = graph->AddNodeByOp(tile1);

    int32_t concat1_out_index;
    graphStatus ret = concat1_node.GetOutputIndexByName(kOutputY, concat1_out_index);
    int32_t tiles_multiple_index;
    ret = tiles[0]->GetInputIndexByName(kInputMultiples, tiles_multiple_index);
    auto [multiple_node, multiple_index] = tiles[0]->GetInDataNodesAndPortIndexs(tiles_multiple_index);
    int32_t tile1_in_index;
    ret = tile1_node.GetInputIndexByName(kInputX, tile1_in_index);
    int32_t tile1_multiple_index;
    ret = tile1_node.GetInputIndexByName(kInputMultiples, tile1_multiple_index);

    ret = graph->AddDataEdge(concat1_node, concat1_out_index, tile1_node, tile1_in_index);
    ret = graph->AddDataEdge(*multiple_node, multiple_index, tile1_node, tile1_multiple_index);
    return tile1_node;
}

GNode CreateConcat2Node(const GNode &node, GNode &tile1_node, const vector<int32_t> &tiles_index,
                        const vector<int32_t> &concat_x_indexs, const int32_t concat_axis) {
    op::ConcatD concat2(kOpNameConcat2);
    concat2.set_attr_concat_dim(concat_axis);
    int32_t input_num = concat_x_indexs.size() - tiles_index.size() + 1;
    concat2.set_attr_N(input_num);
    concat2.create_dynamic_input_x(input_num);

    // 6.1. 插入新的concat2算子
    auto concat2_node = graph->AddNodeByOp(concat2);

    // 6.2. 链接新的tile输入
    TensorDesc tile1_out_desc;
    int32_t tile1_out_index;
    graphStatus ret = tile1_node.GetOutputIndexByName(kOutputY, tile1_out_index);
    ret = tile1_node.GetOutputDesc(tile1_out_index, tile1_out_desc);
    vector<int32_t> concat2_indexs;
    ret = concat2_node.GetDynamicInputIndexesByName(kInputX, concat2_indexs);
    int32_t concat2_in_index = 0;
    ret = graph->AddDataEdge(tile1_node, tile1_out_index, concat2_node, concat2_indexs[concat2_in_index]);
    ret = concat2_node.UpdateInputDesc(concat2_indexs[concat2_in_index], tile1_out_desc);

    // 6.3. 链接其他concat的输入
    for (auto &concat_input_index: concat_x_indexs) {
        auto [in_node, in_node_out_index] = node.GetInDataNodesAndPortIndexs(concat_input_index);
        if (find(tiles_index.begin(), tiles_index.end(), in_node_out_index) != tiles_index.end()) {
            continue;
        }
        TensorDesc in_node_out_desc;
        ret = in_node->GetOutputDesc(in_node_out_index, in_node_out_desc);
        auto concat2_in_index_tmp = concat2_indexs[++concat2_in_index];
        ret = graph->AddDataEdge(*in_node, in_node_out_index, concat2_node, concat2_in_index_tmp);
        ret = concat2_node.UpdateInputDesc(concat2_in_index_tmp, in_node_out_desc);
    }

    // 6.4. 链接原来concat的输出
    // 此步需要放在原来concat连接输出断开后，输出有位置接收新的连接

    return concat2_node;
}

bool IsCyclicUtil(const GNodePtr &node, unordered_set<GNodePtr> &visited, unordered_set<GNodePtr> &recur_set) {
    visited.insert(node);
    recur_set.insert(node);

    for (size_t i = 0; i < node->GetOutputsSize(); ++i) {
        auto out_node_vec = node->GetOutDataNodesAndPortIndexs(i);
        for (auto &[out_node, _]: out_node_vec) {
            if (visited.find(out_node) == visited.end()) {
                if (IsCyclicUtil(out_node, visited, recur_set)) {
                    return true;
                }
            } else if (recur_set.find(out_node) != recur_set.end()) {
                return true;
            }
        }
    }

    recur_set.erase(node);
    return false;
}

bool IsCyclic(const GNodePtr &node) {
    unordered_set<GNodePtr> visited;
    unordered_set<GNodePtr> recur_set;
    return IsCyclicUtil(node, visited, recur_set);
}

bool CheckValidAndDealCyclic(GNode& concat1_node, GNode &tile1_node, GNode &concat2_node) {
    // 7.1. 从concat1_node开始深度优先搜索遍历有向图，查看是否成环
    if (IsCyclic(make_shared<GNode>(concat1_node))) {
        // 处理成环的情况
        // 7.2. 解除新的concat1算子与多个tile的输入的链接
        for (size_t i = 0; i < concat1_node.GetInputsSize(); ++i) {
            auto [in_node, in_id] = concat1_node.GetInDataNodesAndPortIndexs(i);
            graphStatus ret = graph->RemoveEdge(*in_node, in_id, concat1_node, i);
        }

        // 7.3. 解除新的tile算子与concat1的输入的链接
        for (size_t i = 0; i < tile1_node.GetInputsSize(); ++i) {
            auto [in_node, in_id] = tile1_node.GetInDataNodesAndPortIndexs(i);
            graphStatus ret = graph->RemoveEdge(*in_node, in_id, tile1_node, i);
        }

        // 7.4. 解除新的concat2算子所有输入链接, 输出这时候还没连
        for (size_t i = 0; i < concat2_node.GetInputsSize(); ++i) {
            auto [in_node, in_id] = concat2_node.GetInDataNodesAndPortIndexs(i);
            graphStatus ret = graph->RemoveEdge(*in_node, in_id, concat2_node, i);
        }

        // 7.5. 删除新增算子
        graphStatus ret = graph->RemoveNode(concat1_node);
        ret = graph->RemoveNode(tile1_node);
        ret = graph->RemoveNode(concat2_node);

        return false;
    }

    return true;
}

void InheritControlEdges(GNode &node, const vector<GNodePtr> &tiles, GNode& concat1_node, GNode &tile1_node,
                         GNode &concat2_node) {
    for (auto t : tiles) {
        // 8.1. 继承原先的控制边，将多个tile的输入控制边链接到新的concat1上
        for (auto in_control_node : t->GetInControlNodes()) {
            graphStatus ret = graph->AddControlEdge(*in_control_node, concat1_node);
        }

        // 8.2. 继承原先的控制边，将多个tile的输出控制边，链接到新的tile上
        for (auto out_control_node : t->GetOutControlNodes()) {
            graphStatus ret = graph->AddControlEdge(tile1_node, *out_control_node);
        }
    }
    // 8.3. 继承原先的控制边，将原先concat的输入输出控制边链接到新的concat2上
    for (auto in_control_node : node.GetInControlNodes()) {
        graphStatus ret = graph->AddControlEdge(*in_control_node, concat2_node);
    }
    for (auto out_control_node : node.GetOutControlNodes()) {
        graphStatus ret = graph->AddControlEdge(concat2_node, *out_control_node);
    }
}

void RemoveOldNodesEdgesAndAddNewNodeOutput(GNode &node, GNode &concat2_node, const vector<GNodePtr> &tiles) {
    // 9.1. 删除原先的tile算子连边
    for (auto t : tiles) {
        for (size_t i = 0; i < t->GetInputsSize(); ++i) {
            auto [in_node, in_id] = t->GetInDataNodesAndPortIndexs(i);
            graphStatus ret = graph->RemoveEdge(*in_node, in_id, *t, i);
        }
    }

    for (size_t i = 0; i < node.GetInputsSize(); ++i) {
        auto [in_node, in_id] = node.GetInDataNodesAndPortIndexs(i);
        graphStatus ret = graph->RemoveEdge(*in_node, in_id, node, i);
    }

    // 记录node的输出连接的节点
    int32_t node_out_index;
    graphStatus ret = node.GetOutputIndexByName(kOutputY, node_out_index);
    auto out_node_vec = node.GetOutDataNodesAndPortIndexs(node_out_index);

    for (size_t i = 0; i < node.GetOutputsSize(); ++i) {
        for (auto &[out_node, out_id] : node.GetOutDataNodesAndPortIndexs(i)) {
            ret = graph->RemoveEdge(node, i, *out_node, out_id);
        }
    }

    // 6.4. concat2_node链接原来concat的输出
    int32_t concat2_out_index;
    ret = concat2_node.GetOutputIndexByName(kOutputY, concat2_out_index);
    for (auto &[out_node, out_node_in_index]: out_node_vec) {
        ret = graph->AddDataEdge(concat2_node, concat2_out_index, *out_node, out_node_in_index);
    }

    // 9.3. 删除算子
    for (auto t : tiles) {
        ret = graph->RemoveNode(*t);
    }

    ret = graph->RemoveNode(node);
}

bool FuseTileConcat(GNode &node, const vector<GNodePtr> &tiles, const vector<int32_t> &tiles_index,
                    const int32_t concat_axis, const vector<int32_t> &concat_x_indexs) {
    cout << "start to fuse tile and concat." << endl;
    // 4. 插入新的concat1算子，链接多个tile的输入
    GNode concat1_node = CreateConcat1Node(tiles, concat_axis);
    // 5. 插入新的tile算子，链接concat1的输入
    GNode tile1_node = CreateTile1Node(concat1_node, tiles);
    // 6. 插入新的concat2算子，链接新的tile输入、其他concat的输入和原来concat的输出
    GNode concat2_node = CreateConcat2Node(node, tile1_node, tiles_index, concat_x_indexs, concat_axis);
    // 7. 检查是否成环，包括输入边，如果成环，融合失败，删除新增的算子，保留原图，处理成环情况
    if (!CheckValidAndDealCyclic(concat1_node, tile1_node, concat2_node)) {
        cout << "fusion failed!!!" << endl;
        return false;
    }
    // 8.继承控制边
    InheritControlEdges(node, tiles, concat1_node, tile1_node, concat2_node);
    // 9. 检查通过，保留新的算子，删除原先的算子和连边关系, 链接新算子的输出节点
    RemoveOldNodesEdgesAndAddNewNodeOutput(node, concat2_node, tiles);
    cout << "fusion success!!!" << endl;
    return true;
}

bool IsConcatNode(const GNode &node) {
    AscendString node_type;
    auto ret = node.GetType(node_type);
    return node_type == kOpTypeConcatD || node_type == kOpTypeConcat || node_type == kOpTypeConcatV2;
}

int32_t GetConcatAxis(GNode &node) {
    int32_t axis_value;
    if (node.HasAttr(kAttrConcatDim)) {
        graphStatus ret = node.GetAttr(kAttrConcatDim, axis_value);
    } else {
        int32_t axis_index;
        graphStatus ret = node.GetInputIndexByName(kInputConcatDim, axis_index);
        Tensor data;
        ret = node.GetInputConstData(axis_index, data);
        axis_value = data.GetData()[0];
    }
    return axis_value;
}

bool IsTileNode(const GNode &node) {
    AscendString in_node_type;
    auto ret = node.GetType(in_node_type);
    return in_node_type == kOpTypeTile;
}

bool isSatisfyFusionCondition(GNode &node, const vector<GNodePtr> &tiles, const int32_t concat_axis) {
    // 3.1. 判断tile的multiple输入的值，在concat的axis上值为1，即不能在concat轴做广播
    int32_t tile_multiple_index;
    graphStatus ret = node.GetInputIndexByName(kInputMultiples, tile_multiple_index);
    Tensor tile_multiple;
    ret = node.GetInputConstData(tile_multiple_index, tile_multiple);
    constexpr int32_t kTileNoBroadcast = 1;
    if (tile_multiple.GetData()[concat_axis] != kTileNoBroadcast) {
        return false;
    }

    // 3.2. 判断多个tile的mulitiple输入是一样的，表示是做相同的tile运算
    if (!tiles.empty()) {
        Tensor first_tile_mulitple;
        tiles[0]->GetInputConstData(tile_multiple_index, first_tile_mulitple);
        if (memcmp(first_tile_mulitple.GetData(), tile_multiple.GetData(),
                    first_tile_mulitple.GetSize()) != 0) {
            return false;
        }
    }

    // 3.3. 判断tile的dtype和shape，属于小shape（在512byte内）才优化
    int32_t tile_x_index;
    ret = node.GetInputIndexByName(kInputX, tile_x_index);
    TensorDesc tile_x_desc;
    ret = node.GetInputDesc(tile_x_index, tile_x_desc);
    auto tile_x_shape = tile_x_desc.GetShape();
    constexpr int32_t kTileMax = 512;
    if (GetSizeInBytes(tile_x_shape.GetShapeSize(), tile_x_desc.GetDataType()) > kTileMax) {
        return false;
    }

    return true;
}
} // namespace

// |o>---------------------------------
// |o>                          \    /
// |o>      |       |           Concat
// |o>     Tile   Tile            |
// |o>      \      /    ==>      Tile
// |o>       Concat               |
// |o>         |                Concat
// |o>                            |
// |o>---------------------------------
// 融合说明：当tile的shape较小时，将Tile(个数>=2)+concat修改为concat+tile+concat，从而提升HBM搬移和vector多核的利用率
// 改图接口返回值说明：本文件中的改图接口需要判断返回值，基于可读性考虑除了pass入口函数外其他函数中的改图接口只接收返回值
// 但不增加返回值处理代码。如需判断返回值，可配合使用custom_context.SetErrorMessage("xxx")方法
graphStatus FuseTileConcatPass(GraphPtr &graph_ptr, CustomPassContext &custom_context) {
    // 1. 遍历concat，找到concat的tile输入
    cout << "FuseTileConcatPass begin." << endl;
    graph = graph_ptr;
    auto all_nodes = graph->GetAllNodes();
    for (auto &node : all_nodes) {
        if (!IsConcatNode(node)) {
            continue;
        }
        int32_t concat_axis = GetConcatAxis(node);

        vector<GNodePtr> tiles;
        vector<int32_t> tiles_index;
        vector<int32_t> concat_x_indexs;
        auto ret = node.GetDynamicInputIndexesByName(kInputX, concat_x_indexs);
        if (ret != GRAPH_SUCCESS) {
            custom_context.SetErrorMessage("GetDynamicInputIndexesByName from concat node failed.");
            return -1;
        }  
        constexpr int32_t kConcatMinInputNum = 2;
        for (auto &concat_input_index: concat_x_indexs) {
            auto [in_node, in_node_index] = node.GetInDataNodesAndPortIndexs(concat_input_index);
            // 2. 判断多个tile是作为concat输入时，是连续的几个
            // pass，按顺序遍历，因此不需要判断
            if (IsTileNode(*in_node)) {
                // 3. 判断Tile节点是否满足融合条件
                if (isSatisfyFusionCondition(*in_node, tiles, concat_axis)) {
                    tiles.push_back(in_node);
                    tiles_index.push_back(in_node_index);
                }
            } else if (tiles.size() >= kConcatMinInputNum) { // 开始融合
                // 如果融合成功，此concat结点已被删除，不再继续遍历
                // 但如果新的concat节点上还可能存在可优化点，那么可以多加一层外层循环如果pass内改图成功需再次执行此pass
                bool ret = FuseTileConcat(node, tiles, tiles_index, concat_axis, concat_x_indexs);
                tiles.clear();
                tiles_index.clear();
                if (ret) {
                    break;
                }
            } else {
                tiles.clear();
                tiles_index.clear();
            }
        }
        // 考虑到tile节点可能是concat的末尾节点，需要处理边界情况
        if (tiles.size() >= kConcatMinInputNum) { // 开始融合
            FuseTileConcat(node, tiles, tiles_index, concat_axis, concat_x_indexs);
        }
    }
    cout << "FuseTileConcatPass end." << endl;
    return GRAPH_SUCCESS;
}

REGISTER_CUSTOM_PASS("FuseTileConcatPass").CustomPassFn(FuseTileConcatPass);