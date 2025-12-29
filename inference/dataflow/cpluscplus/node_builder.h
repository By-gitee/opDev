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
#ifndef NODE_BUILDER_H
#define NODE_BUILDER_H

#include <fstream>
#include <vector>
#include <map>
#include <memory.h>

#include "all_ops.h"
#include "graph/graph.h"
#include "parser/tensorflow_parser.h"
#include "parser/onnx_parser.h"

#include "ge/ge_api.h"
#include "flow_graph/data_flow.h"

using namespace ge;
using namespace dflow;

struct BuildBasicConfig {
    std::string nodeName;
    uint32_t inputNum;
    uint32_t outputNum;
    std::string compileCfg;
};

FlowNode BuildGraphNode(const BuildBasicConfig &buildCfg, const GraphBuilder &builder)
{
    // construct graph pp
    std::string ppName = buildCfg.nodeName + "_pp";
    auto pp = GraphPp(ppName.c_str(), builder).SetCompileConfig(buildCfg.compileCfg.c_str());
    // construct graph node
    auto node = FlowNode(buildCfg.nodeName.c_str(), buildCfg.inputNum, buildCfg.outputNum);

    node.AddPp(pp);
    return node;
}

FlowNode BuildGraphNode(const BuildBasicConfig &buildCfg, const std::string &fmk, const std::string &modelFile,
                        const std::map<AscendString, AscendString> &parserParams)
{
    std::string nodeName = buildCfg.nodeName;
    return BuildGraphNode(buildCfg,
                          [nodeName, fmk, modelFile, parserParams]() {
        std::string graphName = nodeName + "_pp_" + fmk + "_graph";
        Graph graph(graphName.c_str());
        uint32_t ret;
        if (fmk.compare("TF") == 0) {
            ret = ge::aclgrphParseTensorFlow(modelFile.c_str(), parserParams, graph);
            if (ret != 0) {
                std::cout << "ERROR:TEST======parse tensorflow failed.====================" << std::endl;
            } else {
                std::cout << "SUCCESS:TEST======parse tensorflow succeed.====================" << std::endl;
            }
        } else if (fmk.compare("ONNX") == 0) {
            ret = ge::aclgrphParseONNX(modelFile.c_str(), parserParams, graph);
            if (ret != 0) {
                std::cout << "ERROR:TEST======parse ONNX failed.====================" << std::endl;
            } else {
                std::cout << "SUCCESS:TEST======parse ONNX succeed.====================" << std::endl;
            }
        } else {
            std::cout << "ERROR:TEST======model type is not support=====================" << std::endl;
        }
        return graph;
    });
}

FlowNode BuildOnnxGraphNode(const BuildBasicConfig &buildCfg, const std::string &onnxFile,
                            const std::map<AscendString, AscendString> &parserParams)
{
  return BuildGraphNode(buildCfg, "ONNX", onnxFile, parserParams);
}

FlowNode BuildTfGraphNode(const BuildBasicConfig &buildCfg, const std::string &pbFile,
                            const std::map<AscendString, AscendString> &parserParams)
{
  return BuildGraphNode(buildCfg, "TF", pbFile, parserParams);
}

FlowNode BuildFunctionNodeSimple(const BuildBasicConfig &buildCfg, const bool enableException = false)
{
    // construct FunctionPp
    std::string ppName = buildCfg.nodeName + "_pp";
    auto pp = FunctionPp(ppName.c_str()).SetCompileConfig(buildCfg.compileCfg.c_str());
    pp.SetInitParam("enableExceptionCatch", enableException);
    // construct FlowNode
    auto node = FlowNode(buildCfg.nodeName.c_str(), buildCfg.inputNum, buildCfg.outputNum);
    node.AddPp(pp);
    return node;
}

using FuctionPpSetter = std::function<FunctionPp(FunctionPp)>;

FlowNode BuildFunctionNode(const BuildBasicConfig &buildCfg, FuctionPpSetter ppSetter)
{
    // construct FunctionPp
    std::string ppName = buildCfg.nodeName + "_pp";
    auto pp = FunctionPp(ppName.c_str()).SetCompileConfig(buildCfg.compileCfg.c_str());
    pp = ppSetter(pp);
    // construct FlowNode

    auto node = FlowNode(buildCfg.nodeName.c_str(), buildCfg.inputNum, buildCfg.outputNum);
    node.AddPp(pp);
    return node;
}

using MapSetter = std::function<FlowNode(FlowNode, FunctionPp)>;
FlowNode BuildFunctionNode(const BuildBasicConfig &buildCfg, FuctionPpSetter ppSetter, MapSetter mapSetter)
{
    // construct FunctionPp
    std::string ppName = buildCfg.nodeName + "_pp";
    auto pp = FunctionPp(ppName.c_str()).SetCompileConfig(buildCfg.compileCfg.c_str());
    pp = ppSetter(pp);

    // construct node
    auto node = FlowNode(buildCfg.nodeName.c_str(), buildCfg.inputNum, buildCfg.outputNum);
    // node add pp
    node.AddPp(pp);
    node = mapSetter(node, pp);
    return node;
}
#endif // NODE_BUILDER_H