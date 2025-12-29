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
#include "flow_func/meta_flow_func.h"
#include "flow_func/flow_func_log.h"

namespace FlowFunc {
class CallNnFlowFunc : public MetaFlowFunc {
public:
    int32_t Init() override
    {
        (void)context_->GetAttr("enableExceptionCatch", enableCatchException_);
        return FLOW_FUNC_SUCCESS;
    }

    int32_t Proc(const std::vector<std::shared_ptr<FlowMsg>> &inputMsgs) override
    {
        int32_t expCode = -1;
        uint64_t userContextId = 0;
        if (context_->GetException(expCode, userContextId)) {
            // the following process is a sample. UDF can stop or do other things while got an exception.
            FLOW_FUNC_LOG_ERROR("Get exception raised by normal UDF. expCode[%d] userContextId[%lu].",
                                 expCode, userContextId);
            return FLOW_FUNC_SUCCESS;
        }
        if (inputMsgs.size() == 0) {
            FLOW_FUNC_LOG_ERROR("Input msg is empty.");
            return FLOW_FUNC_FAILED;
        }
        const uint32_t raiseExceptionTransId = 3;
        const int32_t raiseExpCode = -100;
        if (enableCatchException_ && inputMsgs[0]->GetTransactionId() == raiseExceptionTransId) {
            // udf will print error and stop if call RaiseException interface while dataflow disable exception catch
            context_->RaiseException(raiseExpCode, raiseExceptionTransId);
            FLOW_FUNC_LOG_ERROR("Raise exception in nn for test.");
        }
        std::vector<std::shared_ptr<FlowMsg>> outputMsgs;
        auto ret = context_->RunFlowModel(dependKey_.c_str(), inputMsgs, outputMsgs, 100000);
        if (ret != FLOW_FUNC_SUCCESS) {
            FLOW_FUNC_LOG_ERROR("Run flow model failed.");
            return ret;
        }
        for (size_t i = 0UL; i < outputMsgs.size(); ++i) {
            ret = context_->SetOutput(i, outputMsgs[i]);
            if (ret != FLOW_FUNC_SUCCESS) {
                FLOW_FUNC_LOG_ERROR("Set output msgs failed.");
                return ret;
            }
        }
        FLOW_FUNC_LOG_INFO("Run flow func success.");
        return FLOW_FUNC_SUCCESS;
    }

private:
    std::string dependKey_{"invoke_graph"};
    bool enableCatchException_ = false;
};

REGISTER_FLOW_FUNC("call_nn", CallNnFlowFunc);
}