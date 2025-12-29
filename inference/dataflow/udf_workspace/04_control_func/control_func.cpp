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

#include <vector>
#include <iostream>
#include <mutex>

#include "flow_func/meta_multi_func.h"
#include "flow_func/flow_func_log.h"

namespace FlowFunc {
class AddFlowFunc : public MetaMultiFunc {
public:
    AddFlowFunc() = default;
    ~AddFlowFunc() override = default;

    int32_t  Init(const std::shared_ptr<MetaParams> &params) override
    {
        return FLOW_FUNC_SUCCESS;
    }

    int32_t Proc(const std::shared_ptr<MetaRunContext> &runContext,
                 const std::vector<std::shared_ptr<FlowMsg>> &inputFlowMsgs)
    {
        // there should be two input for add func
        if (inputFlowMsgs.size() != 1) {
            FLOW_FUNC_LOG_ERROR("Input size is not 1.");
            return FLOW_FUNC_FAILED;
        }

        const auto &input1 = inputFlowMsgs[0];
        if (input1->GetRetCode() !=0) {
            FLOW_FUNC_LOG_ERROR("Input ret code is not 0.");
            return FLOW_FUNC_FAILED;
        }
        auto inputTensor1 = input1->GetTensor();
        auto inputDataType1 = inputTensor1->GetDataType();
        auto inputShape1 = inputTensor1->GetShape();
        if (inputShape1.size() != 1) {
            FLOW_FUNC_LOG_ERROR("Input shape is expected as {1}.");
            return FLOW_FUNC_FAILED;
        }
        if (inputDataType1 != TensorDataType::DT_UINT32) {
            FLOW_FUNC_LOG_ERROR("Input data type is expected as uint32.");
            return FLOW_FUNC_FAILED;
        }
        const uint32_t elementNum = 3;
        auto outputMsg = runContext->AllocTensorMsg({elementNum}, TensorDataType::DT_UINT32);
        if (outputMsg == nullptr) {
            FLOW_FUNC_LOG_ERROR("Fail to alloc tensor msg.");
            return FLOW_FUNC_FAILED;
        }
        auto outputTensor = outputMsg->GetTensor();

        auto dataSize1 = outputTensor->GetDataSize();
        if (dataSize1 != elementNum * sizeof(uint32_t)) {
            FLOW_FUNC_LOG_ERROR("Alloc tensor data size in invalid.");
            return FLOW_FUNC_FAILED;      
        }
        auto *outputData = static_cast<uint32_t *>(outputTensor->GetData());
        for (uint32_t i = 0; i < elementNum; ++i) {
            outputData[i] = i + 1;
            // element value : 1,2,3
        }

        int32_t inputData1 = *(static_cast<uint32_t *>(inputTensor1->GetData()));
        if (inputData1 == 0) {
            FLOW_FUNC_LOG_INFO("invoke Proc0");
            return SetOutputByIndex(runContext, {0, 1}, outputMsg);
        } else {
            FLOW_FUNC_LOG_INFO("invoke Proc1");
            return SetOutputByIndex(runContext, {2, 3}, outputMsg);
        }

        return FLOW_FUNC_SUCCESS;
    }

private:
    int SetOutputByIndex(const std::shared_ptr<MetaRunContext> &runContext,
        const std::vector<uint32_t> &ids, std::shared_ptr<FlowMsg> outputMsg) const
    {
        for (uint32_t id : ids) {
            const auto ret = runContext->SetOutput(id, outputMsg);
            if (ret != FLOW_FUNC_SUCCESS) {
                FLOW_FUNC_LOG_ERROR("Fail to set output.");
                return FLOW_FUNC_FAILED;
            }
        }
        return FLOW_FUNC_SUCCESS;
    }
};

FLOW_FUNC_REGISTRAR(AddFlowFunc)
    .RegProcFunc("control_func", &AddFlowFunc::Proc);
}