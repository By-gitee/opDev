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
#include <sys/time.h>
#include <unistd.h>

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
constexpr int32_t kMillion = 1000000;
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

    auto node0 = dflow::FlowNode("node0", 2, 1).SetInput(0, data0).SetInput(1, data1);

    auto pp0 = dflow::FunctionPp("func_pp0").SetCompileConfig("../config/add_func.json").SetInitParam("out_type", ge::DT_INT32);
    node0.AddPp(pp0);

    //set flow graph
    std::vector<FlowOperator> inputsOperator{data0, data1};

    std::vector<FlowOperator> outputsOperator{node0};
    flow_graph.SetInputs(inputsOperator).SetOutputs(outputsOperator);
    return flow_graph;
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
    // warm- up
    geRet = session->FeedDataFlowGraph(0, inputsData, dataFlowInfo, kFeedTimeout);
    if (geRet != ge::SUCCESS) {
        std::cout << "ERROR=======Feed data failed===========" << std::endl;
        ge::GEFinalize();
        return geRet;
    }

    const size_t loopNum = 10;
    timeval s;
    timeval e;
    gettimeofday(&s, nullptr);
    for (size_t i = 0; i < loopNum; ++i) {
        geRet = session->FeedDataFlowGraph(0, inputsData, dataFlowInfo, kFeedTimeout);
        if (geRet != ge::SUCCESS) {
            std::cout << "ERROR=======Feed data failed===========" << std::endl;
            ge::GEFinalize();
            return geRet;
        }
    }
    gettimeofday(&e, nullptr);
    double timeCost = (e.tv_sec - s.tv_sec) * kMillion + static_cast<double>(e.tv_usec - s.tv_usec);
    std::cout << "TEST-TIME:Feed data time: " << timeCost / loopNum << "us per loop" << std::endl;

    // warm-up
    std::vector<ge::Tensor> outputsData;
    geRet = session->FetchDataFlowGraph(0, outputsData, dataFlowInfo, kFetchTimeout);
    if (geRet != ge::SUCCESS) {
        std::cout << "ERROR=======Fetch data failed===========" << std::endl;
        ge::GEFinalize();
        return geRet;
    }

    gettimeofday(&s, nullptr);
    for (size_t i = 0; i < loopNum; ++i) {
        outputsData.clear();
        geRet = session->FetchDataFlowGraph(0, outputsData, dataFlowInfo, kFetchTimeout);
        if (geRet != ge::SUCCESS) {
            std::cout << "ERROR=======Fetch data failed===========" << std::endl;
            ge::GEFinalize();
            return geRet;
        }
    }
    gettimeofday(&e, nullptr);
    timeCost = (e.tv_sec - s.tv_sec) * kMillion + static_cast<double>(e.tv_usec - s.tv_usec);
    std::cout << "TEST-TIME:Fetch data time: " << timeCost / loopNum << "us per loop" << std::endl;
    std::cout << "TEST=======run case success===========" << std::endl;
    ge::GEFinalize();
    return 0;
}