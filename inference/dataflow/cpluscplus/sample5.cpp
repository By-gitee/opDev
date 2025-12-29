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
 * FlowData
 *    |
 *    |
 * FlowNode0 （control which func to  invoke for FlowNode1）
 *    |
 *    |
 * FlowNode1  (contains 3 flow func)
 *    |
 *    |
 * FlowOutput
 *
 * @return DataFlow graph
 * 
*/
dflow::FlowGraph BuildDataFlow()
{
    dflow::FlowGraph flow_graph("flow_graph");
    auto data0 = FlowData("Data0", 0);
    BuildBasicConfig udf1BuildCfg = {.nodeName = "node0", .inputNum = 1, .outputNum = 4,
                                     .compileCfg = "../config/add_func_multi_control.json"};
    auto node0 = BuildFunctionNodeSimple(udf1BuildCfg)
                 .SetInput(0, data0);

    BuildBasicConfig udf2BuildCfg = {.nodeName = "node1", .inputNum = 4, .outputNum = 2,
                                     .compileCfg = "../config/add_func_multi.json"}; 
    auto node1 = BuildFunctionNodeSimple(udf2BuildCfg)
                 .SetInput(0, node0, 0)
                 .SetInput(1, node0, 1)
                 .SetInput(2, node0, 2)
                 .SetInput(3, node0, 3);

    std::vector<FlowOperator> inputsOperator{data0};
    std::vector<FlowOperator> outputsOperator{node1};
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
    auto flow_graph = BuildDataFlow();

    // Initialize
    std::map<ge::AscendString, AscendString> config = {{"ge.exec.deviceId", "0"},
                                                       {"ge.exec.logicalDeviceClusterDeployMode", "SINGLE"},
                                                       {"ge.exec.logicalDeviceId", "[0:0]"},
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
    const int64_t elementNum = 1;
    std::vector<int64_t> shape = {elementNum};
    // invoke proc1
    int32_t inputData = 0;
    ge::Tensor inputTensor;
    ge::TensorDesc desc(ge::Shape(shape), ge::FORMAT_ND, ge::DT_UINT32);
    inputTensor.SetTensorDesc(desc);
    inputTensor.SetData((uint8_t*)&inputData, sizeof(uint32_t) * elementNum);

    ge::DataFlowInfo dataFlowInfo;
    std::vector<ge::Tensor> inputsData = {inputTensor};

    geRet = session->FeedDataFlowGraph(0, inputsData, dataFlowInfo, kFeedTimeout);
    if (geRet != ge::SUCCESS) {
        std::cout << "ERROR=======Feed data failed===========" << std::endl;
        ge::GEFinalize();
        return geRet;
    }
    std::vector<ge::Tensor> outputsData;
    geRet = session->FetchDataFlowGraph(0, {0}, outputsData, dataFlowInfo, kFetchTimeout);
    if (geRet != ge::SUCCESS) {
        std::cout << "ERROR=======Fetch data failed===========" << std::endl;
        ge::GEFinalize();
        return geRet;
    }
    // Verify outputs
    std::vector<uint32_t>expectOut = {2, 4, 6};
    if (!CheckResult(outputsData, expectOut)) {
        std::cout << "ERROR=======Check result data failed===========" << std::endl;
        ge::GEFinalize();
        return -1;
    }

    // Invoke Proc2
    inputData = 1;
    inputTensor.SetData((uint8_t*)&inputData, sizeof(uint32_t) * elementNum);
    inputsData[0] = inputTensor;
    geRet = session->FeedDataFlowGraph(0, inputsData, dataFlowInfo, kFeedTimeout);
    if (geRet != ge::SUCCESS) {
        std::cout << "ERROR=======Feed data failed===========" << std::endl;
        ge::GEFinalize();
        return geRet;
    }
    std::vector<ge::Tensor> outputsData2;
    geRet = session->FetchDataFlowGraph(0, {1}, outputsData2, dataFlowInfo, kFetchTimeout);
    if (geRet != ge::SUCCESS) {
        std::cout << "ERROR=======Fetch data failed===========" << std::endl;
        ge::GEFinalize();
        return geRet;
    }
    // Verify outputs
    std::vector<uint32_t>expectOut2 = {3, 6, 9};
    if (!CheckResult(outputsData2, expectOut2)) {
        std::cout << "ERROR=======Check result data failed===========" << std::endl;
        ge::GEFinalize();
        return -1;
    }
    std::cout << "TEST=======run case success===========" << std::endl;
    ge::GEFinalize();
    return 0;
}