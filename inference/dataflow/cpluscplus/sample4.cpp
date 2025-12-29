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
#include <vector>
#include <map>
#include <string>

#include "all_ops.h"
#include "ge/ge_api.h"
#include "graph/graph.h"
#include "flow_graph/data_flow.h"
#include "node_builder.h"

using namespace ge;
using namespace dflow;
namespace {
constexpr int32_t kFeedTimeout = 3000;
constexpr int32_t kFetchTimeout = 30000;
/**
 * @brief
 * Build a dataflow graph by DataFlow API
 * The dataflow graph contains 3 flow nodes and DAG shows as following:
 * FlowData      FlowData
 *    \            /
 *     \          /
 *      \        /
 *       \      /
 *       FlowNode0
 *           |
 *           |
 *       FlowOutput
 *
 * @return DataFlow graph
 * 
*/
FlowGraph BuildDataFlowGraph()
{
    //construct graph
    dflow::FlowGraph flow_graph("flow_graph");

    auto data0 = dflow::FlowData("Data0", 0);
    auto data1 = dflow::FlowData("Data1", 1);

    std::map<ge::AscendString, ge::AscendString> parserParams = {
        {ge::AscendString(ge::ir_option::INPUT_DATA_NAMES), ge::AscendString("Placeholder,Placeholder_1")}
    };
    BuildBasicConfig buildCfg = {.nodeName = "node0", .inputNum = 2, .outputNum = 1,
                                 .compileCfg = "../config/add_graph.json"};
    auto node0 = BuildTfGraphNode(buildCfg, "../config/add", parserParams)
                 .SetInput(0, data0)
                 .SetInput(1, data1);
    std::vector<FlowOperator> inputsOperator{data0, data1};
    std::vector<FlowOperator> outputsOperator{node0};
    flow_graph.SetInputs(inputsOperator).SetOutputs(outputsOperator);
    return flow_graph;
}

bool CheckResult(std::vector<ge::Tensor> &result, const std::vector<uint32_t> &expectOut)
{
    if (result.size() != 1) {
        std::cout << "ERROR=======Fetch data size is expected containing 1 element=" << std::endl;
        return false;
    }
    if (result[0].GetSize() != expectOut.size() * sizeof(uint32_t)) {
        std::cout << "ERROR=======Verify data size failed===========" << std::endl;
        std::cout << "Tensor size:" << result[0].GetSize() << std::endl;
        std::cout << "Expect size:" << expectOut.size() * sizeof(uint32_t) << std::endl;
        return false;    
    }

    uint32_t* outputData = reinterpret_cast<uint32_t*>(result[0].GetData());
    if (outputData != nullptr) {
        for (size_t k = 0; k < expectOut.size(); ++k) {
            if (expectOut[k] != outputData[k]) {
                std::cout << "ERROR=======Verify data failed===========" << std::endl;
                std::cout << "ERROR======expect:" << expectOut[k] << "  real:" << outputData[k] <<std::endl;
                return false;
            }
        }
    }
    return true;
}
}

int32_t main()
{
    // Build dataflow graph
    auto flow_graph = BuildDataFlowGraph();

    // Initialize
    std::map<ge::AscendString, AscendString> config = {{"ge.exec.deviceId", "0"},
                                                       {"ge.graphRunMode", "0"}};
    auto geRet = ge::GEInitialize(config);
    if (geRet != ge::SUCCESS) {
        std::cout << "ERROR=====GeInitialize failed.=======" << std::endl;
        return geRet;
    }

    // Create Session
    std::map<ge::AscendString, ge::AscendString> options;
    std::shared_ptr<ge::Session> session = std::make_shared<ge::Session>(options);
    if (session == nullptr) {
        std::cout << "ERROR=======Create session failed===========" << std::endl;
        ge::GEFinalize();
        return geRet;
    }

    // Add graph
    geRet = session->AddGraph(0, flow_graph.ToGeGraph());
    if (geRet != ge::SUCCESS) {
        std::cout << "ERROR=======Add graph failed===========" << std::endl;
        ge::GEFinalize();
        return geRet;
    }

    // Prepare Inputs
    const int64_t elementNum = 3;
    std::vector<int64_t> shape = {elementNum};
    uint32_t inputData[elementNum] = {4, 7, 5};
    ge::Tensor inputTensor;
    ge::TensorDesc desc(ge::Shape(shape), ge::FORMAT_ND, ge::DT_UINT32);
    inputTensor.SetTensorDesc(desc);
    inputTensor.SetData((uint8_t*)inputData, sizeof(uint32_t) * elementNum);

    ge::DataFlowInfo dataFlowInfo;
    std::vector<ge::Tensor> inputsData = {inputTensor, inputTensor};

    // FeedInput
    const size_t loopNum = 4;
    for (size_t i = 0; i < loopNum; ++i) {
        geRet = session->FeedDataFlowGraph(0, inputsData, dataFlowInfo, kFeedTimeout);
        if (geRet != ge::SUCCESS) {
            std::cout << "ERROR=======Feed data failed===========" << std::endl;
            ge::GEFinalize();
            return geRet;
        }
    }

    // Verify outputs
    std::vector<uint32_t>expectOutput = {8, 14, 10};
    for (size_t i = 0; i < loopNum; ++i) {
        std::vector<ge::Tensor> outputsData;
        geRet = session->FetchDataFlowGraph(0, outputsData, dataFlowInfo, kFetchTimeout);
        if (geRet != ge::SUCCESS) {
            std::cout << "ERROR=======Fetch data failed===========" << std::endl;
            ge::GEFinalize();
            return geRet;
        }
        
        if (!CheckResult(outputsData, expectOutput)) {
            std::cout << "ERROR=======Check result data failed===========" << std::endl;
            ge::GEFinalize();
            return -1;
        }
    }
    std::cout << "TEST=======run case success===========" << std::endl;
    ge::GEFinalize();
    return 0;
}