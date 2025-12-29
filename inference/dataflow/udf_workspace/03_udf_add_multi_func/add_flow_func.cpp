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

    int32_t Init(const std::shared_ptr<MetaParams> &params) override
    {
        const auto getRet = params->GetAttr("out_type", outDataType_);
        if (getRet != FLOW_FUNC_SUCCESS) {
            FLOW_FUNC_LOG_INFO("Attr out_type is not exist.");
            outDataType_ = TensorDataType::DT_UINT32;
        }
        setOutputCount_ = 0U;
        return FLOW_FUNC_SUCCESS;
    }

    template<typename srcT, typename dstT>
    void Add1(srcT *src1, srcT *src2, dstT *dst, size_t count)
    {
        for (size_t i = 0; i < count; i++) {
            dst[i] = src1[i] + src2[i];
        }
    }

    template<typename srcT, typename dstT>
    void Add2(srcT *src1, srcT *src2, dstT *dst, size_t count)
    {
        for (size_t i = 0; i < count; i++) {
            dst[i] = src1[i] + src1[i] + src2[i];
        }
    }

    int32_t Proc1(const std::shared_ptr<MetaRunContext> &runContext,
                  const std::vector<std::shared_ptr<FlowMsg>> &inputFlowMsgs)
    {
        // there should be two input for add func
        const size_t inputNum = 2;
        if (inputFlowMsgs.size() != inputNum) {
            FLOW_FUNC_LOG_ERROR("Input size is not 2.");
            return FLOW_FUNC_FAILED;
        }

        const auto &input1 = inputFlowMsgs[0];
        const auto &input2 = inputFlowMsgs[1];
        FLOW_FUNC_LOG_INFO("input1 ret code is %d", input1->GetRetCode());
        FLOW_FUNC_LOG_INFO("input2 ret code is %d", input2->GetRetCode());
        if (input1->GetRetCode() !=0 || input2->GetRetCode() != 0) {
            FLOW_FUNC_LOG_ERROR("Input ret code is not 0.");
            return FLOW_FUNC_FAILED;
        }
        MsgType msgType1 = input1->GetMsgType();
        MsgType msgType2 = input2->GetMsgType();
        if (msgType1 != MsgType::MSG_TYPE_TENSOR_DATA || msgType2 != MsgType::MSG_TYPE_TENSOR_DATA) {
            FLOW_FUNC_LOG_ERROR("Input msg type is not data.");
            return FLOW_FUNC_FAILED;
        }
        auto inputTensor1 = input1->GetTensor();
        auto inputTensor2 = input2->GetTensor();
        auto inputDataType1 = inputTensor1->GetDataType();
        auto inputDataType2 = inputTensor2->GetDataType();
        if (inputDataType1 != inputDataType2) {
            FLOW_FUNC_LOG_ERROR("Input data type is not same.");
            return FLOW_FUNC_FAILED;
        }
        auto inputShape1 = inputTensor1->GetShape();
        auto inputShape2 = inputTensor2->GetShape();
        if (inputShape1 != inputShape2) {
            FLOW_FUNC_LOG_ERROR("Input shape is not same.");
            return FLOW_FUNC_FAILED;
        }

        auto outputMsg = runContext->AllocTensorMsg(inputShape1, outDataType_);
        if (outputMsg == nullptr) {
            FLOW_FUNC_LOG_ERROR("Fail to alloc tensor msg.");
            return -1;
        }
        auto outputTensor = outputMsg->GetTensor();

        auto dataSize1 = inputTensor1->GetDataSize();
        auto dataSize2 = inputTensor2->GetDataSize();
        if (dataSize1 == 0) {
            return runContext->SetOutput(0, outputMsg);
        }
        auto inputData1 = inputTensor1->GetData();
        auto inputData2 = inputTensor2->GetData();
        auto outputData = outputTensor->GetData();

       switch (inputDataType1) {
            case TensorDataType::DT_INT8:
                Add1<int8_t, int8_t>(static_cast<int8_t *>(inputData1), static_cast<int8_t *>(inputData2),
                                     static_cast<int8_t *>(outputData), dataSize1/sizeof(float));
                break;
            case TensorDataType::DT_UINT16:
                Add1<uint16_t, uint16_t>(static_cast<uint16_t *>(inputData1), static_cast<uint16_t *>(inputData2),
                                         static_cast<uint16_t *>(outputData), dataSize1/sizeof(uint16_t));
                break;
            case TensorDataType::DT_INT16:
                Add1<int16_t, int16_t>(static_cast<int16_t *>(inputData1), static_cast<int16_t *>(inputData2),
                                       static_cast<int16_t *>(outputData), dataSize1/sizeof(int16_t));
                break;
            case TensorDataType::DT_UINT32:
                Add1<uint32_t, uint32_t>(static_cast<uint32_t *>(inputData1), static_cast<uint32_t *>(inputData2),
                                         static_cast<uint32_t *>(outputData), dataSize1/sizeof(uint32_t));
                break;
            case TensorDataType::DT_INT32:
                Add1<int32_t, int32_t>(static_cast<int32_t *>(inputData1), static_cast<int32_t *>(inputData2),
                                       static_cast<int32_t *>(outputData), dataSize1/sizeof(int32_t));
                break;
            case TensorDataType::DT_INT64:
                Add1<int64_t, int64_t>(static_cast<int64_t *>(inputData1), static_cast<int64_t *>(inputData2),
                                       static_cast<int64_t *>(outputData), dataSize1/sizeof(int64_t));
                break;
            case TensorDataType::DT_FLOAT:
                Add1<float, float>(static_cast<float *>(inputData1), static_cast<float *>(inputData2),
                                   static_cast<float *>(outputData), dataSize1/sizeof(float));
                break;  
            default:
                FLOW_FUNC_LOG_ERROR("Unsupported data type.");
                const int32_t retCode = 100;
                outputMsg->SetRetCode(retCode);
                break;
        }
        std::unique_lock<std::mutex> lock(countMutex_);
        setOutputCount_++;
        return runContext->SetOutput(0, outputMsg);
    }

    int32_t Proc2(const std::shared_ptr<MetaRunContext> &runContext,
                  const std::vector<std::shared_ptr<FlowMsg>> &inputFlowMsgs)
    {
        // there should be two input for add func
        const size_t inputNum = 2;
        if (inputFlowMsgs.size() != inputNum) {
            FLOW_FUNC_LOG_ERROR("Input size is not 2.");
            return FLOW_FUNC_FAILED;
        }

        const auto &input1 = inputFlowMsgs[0];
        const auto &input2 = inputFlowMsgs[1];
        FLOW_FUNC_LOG_INFO("input1 ret code is ", input1->GetRetCode());
        FLOW_FUNC_LOG_INFO("input2 ret code is ", input2->GetRetCode());
        if (input1->GetRetCode() !=0 || input2->GetRetCode() != 0) {
            FLOW_FUNC_LOG_ERROR("Input ret code is not 0.");
            return FLOW_FUNC_FAILED;
        }
        MsgType msgType1 = input1->GetMsgType();
        MsgType msgType2 = input2->GetMsgType();
        if (msgType1 != MsgType::MSG_TYPE_TENSOR_DATA || msgType2 != MsgType::MSG_TYPE_TENSOR_DATA) {
            FLOW_FUNC_LOG_ERROR("Input msg type is not data.");
            return FLOW_FUNC_FAILED;
        }
        auto inputTensor1 = input1->GetTensor();
        auto inputTensor2 = input2->GetTensor();
        auto inputDataType1 = inputTensor1->GetDataType();
        auto inputDataType2 = inputTensor2->GetDataType();
        if (inputDataType1 != inputDataType2) {
            FLOW_FUNC_LOG_ERROR("Input data type is not same.");
            return FLOW_FUNC_FAILED;
        }
        auto inputShape1 = inputTensor1->GetShape();
        auto inputShape2 = inputTensor2->GetShape();
        if (inputShape1 != inputShape2) {
            FLOW_FUNC_LOG_ERROR("Input shape is not same.");
            return FLOW_FUNC_FAILED;
        }

        auto outputMsg = runContext->AllocTensorMsg(inputShape1, outDataType_);
        if (outputMsg == nullptr) {
           FLOW_FUNC_LOG_ERROR("Fail to alloc tensor msg.");
            return -1;
        }
        auto outputTensor = outputMsg->GetTensor();

        auto dataSize1 = inputTensor1->GetDataSize();
        auto dataSize2 = inputTensor2->GetDataSize();
        if (dataSize1 == 0) {
            return runContext->SetOutput(0, outputMsg);
        }
        auto inputData1 = inputTensor1->GetData();
        auto inputData2 = inputTensor2->GetData();
        auto outputData = outputTensor->GetData();

       switch (inputDataType1) {
            case TensorDataType::DT_INT8:
                Add2<int8_t, int8_t>(static_cast<int8_t *>(inputData1), static_cast<int8_t *>(inputData2),
                                     static_cast<int8_t *>(outputData), dataSize1/sizeof(float));
                break;
            case TensorDataType::DT_UINT16:
                Add2<uint16_t, uint16_t>(static_cast<uint16_t *>(inputData1), static_cast<uint16_t *>(inputData2),
                                         static_cast<uint16_t *>(outputData), dataSize1/sizeof(uint16_t));
                break;
            case TensorDataType::DT_INT16:
                Add2<int16_t, int16_t>(static_cast<int16_t *>(inputData1), static_cast<int16_t *>(inputData2),
                                       static_cast<int16_t *>(outputData), dataSize1/sizeof(int16_t));
                break;
            case TensorDataType::DT_UINT32:
                Add2<uint32_t, uint32_t>(static_cast<uint32_t *>(inputData1), static_cast<uint32_t *>(inputData2),
                                         static_cast<uint32_t *>(outputData), dataSize1/sizeof(uint32_t));
                break;
            case TensorDataType::DT_INT32:
                Add2<int32_t, int32_t>(static_cast<int32_t *>(inputData1), static_cast<int32_t *>(inputData2),
                                       static_cast<int32_t *>(outputData), dataSize1/sizeof(int32_t));
                break;
            case TensorDataType::DT_INT64:
                Add2<int64_t, int64_t>(static_cast<int64_t *>(inputData1), static_cast<int64_t *>(inputData2),
                                       static_cast<int64_t *>(outputData), dataSize1/sizeof(int64_t));
                break;
            case TensorDataType::DT_FLOAT:
                Add2<float, float>(static_cast<float *>(inputData1), static_cast<float *>(inputData2),
                                   static_cast<float *>(outputData), dataSize1/sizeof(float));
                break;  
            default:
                FLOW_FUNC_LOG_ERROR("Unsupported data type.");
                const int32_t retCode = 100;
                outputMsg->SetRetCode(retCode);
                break;
        }
        std::unique_lock<std::mutex> lock(countMutex_);
        setOutputCount_++;
        return runContext->SetOutput(1, outputMsg);
    }
private:
    TensorDataType outDataType_;
    std::mutex countMutex_;
    uint32_t setOutputCount_;
};

FLOW_FUNC_REGISTRAR(AddFlowFunc)
    .RegProcFunc("Proc1", &AddFlowFunc::Proc1)
    .RegProcFunc("Proc2", &AddFlowFunc::Proc2);
}