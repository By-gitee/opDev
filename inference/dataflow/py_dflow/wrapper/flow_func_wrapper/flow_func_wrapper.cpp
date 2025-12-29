/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
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

#include <sstream>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>
#include <pybind11/stl_bind.h>
#include <pybind11/iostream.h>
#include <pybind11/detail/common.h>
#include <sys/syscall.h>
#include "utils.h"
#include "flow_func/ascend_string.h"
#include "flow_func/attr_value.h"
#include "flow_func/flow_func_defines.h"
#include "flow_func/flow_func_log.h"
#include "flow_func/flow_msg.h"
#include "flow_func/flow_msg_queue.h"
#include "flow_func/meta_multi_func.h"
#include "flow_func/meta_run_context.h"
#include "flow_func/meta_params.h"
#include "flow_func/tensor_data_type.h"
#include "graph/tensor.h"
#include "toolchain/slog.h"

namespace {
namespace py = pybind11;

inline int64_t GetTid() {
    thread_local static const int64_t tid = static_cast<int64_t>(syscall(__NR_gettid));
    return tid;
}

constexpr int32_t MODULE_ID_UDF = static_cast<int32_t>(UDF);
#define UDF_LOG_ERROR(fmt, ...) \
    dlog_error(MODULE_ID_UDF, "[%s][tid:%ld]: " fmt, __FUNCTION__, GetTid(), ##__VA_ARGS__)

template <typename... Args>
using overload_cast_ = pybind11::detail::overload_cast_impl<Args...>;

const std::map<FlowFunc::TensorDataType, ge::DataType> kFuncDataTypeToGeDataType{
    {FlowFunc::TensorDataType::DT_FLOAT, ge::DataType::DT_FLOAT},
    {FlowFunc::TensorDataType::DT_FLOAT16, ge::DataType::DT_FLOAT16},
    {FlowFunc::TensorDataType::DT_INT8, ge::DataType::DT_INT8},
    {FlowFunc::TensorDataType::DT_INT16, ge::DataType::DT_INT16},
    {FlowFunc::TensorDataType::DT_UINT16, ge::DataType::DT_UINT16},
    {FlowFunc::TensorDataType::DT_UINT8, ge::DataType::DT_UINT8},
    {FlowFunc::TensorDataType::DT_INT32, ge::DataType::DT_INT32},
    {FlowFunc::TensorDataType::DT_INT64, ge::DataType::DT_INT64},
    {FlowFunc::TensorDataType::DT_UINT32, ge::DataType::DT_UINT32},
    {FlowFunc::TensorDataType::DT_UINT64, ge::DataType::DT_UINT64},
    {FlowFunc::TensorDataType::DT_BOOL, ge::DataType::DT_BOOL},
    {FlowFunc::TensorDataType::DT_DOUBLE, ge::DataType::DT_DOUBLE},
    {FlowFunc::TensorDataType::DT_QINT8, ge::DataType::DT_QINT8},
    {FlowFunc::TensorDataType::DT_QINT16, ge::DataType::DT_QINT16},
    {FlowFunc::TensorDataType::DT_QINT32, ge::DataType::DT_QINT32},
    {FlowFunc::TensorDataType::DT_QUINT8, ge::DataType::DT_QUINT8},
    {FlowFunc::TensorDataType::DT_QUINT16, ge::DataType::DT_QUINT16},
    {FlowFunc::TensorDataType::DT_DUAL, ge::DataType::DT_DUAL},
    {FlowFunc::TensorDataType::DT_INT4, ge::DataType::DT_INT4},
    {FlowFunc::TensorDataType::DT_UINT1, ge::DataType::DT_UINT1},
    {FlowFunc::TensorDataType::DT_INT2, ge::DataType::DT_INT2},
    {FlowFunc::TensorDataType::DT_UINT2, ge::DataType::DT_UINT2},
    {FlowFunc::TensorDataType::DT_UNDEFINED, ge::DataType::DT_UNDEFINED}
};

ge::DataType TransFuncDataTypeToGeDataType(const FlowFunc::TensorDataType &data_type)
{
    const auto iter = kFuncDataTypeToGeDataType.find(data_type);
    if (iter == kFuncDataTypeToGeDataType.cend()) {
        return ge::DataType::DT_UNDEFINED;
    }
    return iter->second;
}

const std::map<ge::DataType, FlowFunc::TensorDataType> kGeDataTypeToFlowFuncDataType{
    {ge::DataType::DT_FLOAT, FlowFunc::TensorDataType::DT_FLOAT},
    {ge::DataType::DT_FLOAT16, FlowFunc::TensorDataType::DT_FLOAT16},
    {ge::DataType::DT_INT8, FlowFunc::TensorDataType::DT_INT8},
    {ge::DataType::DT_INT16, FlowFunc::TensorDataType::DT_INT16},
    {ge::DataType::DT_UINT16, FlowFunc::TensorDataType::DT_UINT16},
    {ge::DataType::DT_UINT8, FlowFunc::TensorDataType::DT_UINT8},
    {ge::DataType::DT_INT32, FlowFunc::TensorDataType::DT_INT32},
    {ge::DataType::DT_INT64, FlowFunc::TensorDataType::DT_INT64},
    {ge::DataType::DT_UINT32, FlowFunc::TensorDataType::DT_UINT32},
    {ge::DataType::DT_UINT64, FlowFunc::TensorDataType::DT_UINT64},
    {ge::DataType::DT_BOOL, FlowFunc::TensorDataType::DT_BOOL},
    {ge::DataType::DT_DOUBLE, FlowFunc::TensorDataType::DT_DOUBLE},
    {ge::DataType::DT_QINT8, FlowFunc::TensorDataType::DT_QINT8},
    {ge::DataType::DT_QINT16, FlowFunc::TensorDataType::DT_QINT16},
    {ge::DataType::DT_QINT32, FlowFunc::TensorDataType::DT_QINT32},
    {ge::DataType::DT_QUINT8, FlowFunc::TensorDataType::DT_QUINT8},
    {ge::DataType::DT_QUINT16, FlowFunc::TensorDataType::DT_QUINT16},
    {ge::DataType::DT_DUAL, FlowFunc::TensorDataType::DT_DUAL},
    {ge::DataType::DT_INT4, FlowFunc::TensorDataType::DT_INT4},
    {ge::DataType::DT_UINT1, FlowFunc::TensorDataType::DT_UINT1},
    {ge::DataType::DT_INT2, FlowFunc::TensorDataType::DT_INT2},
    {ge::DataType::DT_UINT2, FlowFunc::TensorDataType::DT_UINT2},
    {ge::DataType::DT_UNDEFINED, FlowFunc::TensorDataType::DT_UNDEFINED}
};

FlowFunc::TensorDataType TransGeDataTypeToFuncDataType(const ge::DataType &data_type)
{
    const auto iter = kGeDataTypeToFlowFuncDataType.find(data_type);
    if (iter == kGeDataTypeToFlowFuncDataType.cend()) {
        return FlowFunc::TensorDataType::DT_UNDEFINED;
    }
    return iter->second;
}

py::memoryview ToReadonlyMemoryView(const FlowFunc::Tensor &tensor)
{
    const auto elementSize = tensor.GetDataSize() / tensor.GetElementCnt();
    std::vector<uint64_t> strides;
    strides.push_back(elementSize);
    auto strideNum = tensor.GetShape().size();
    if (strideNum > 1) {
        uint64_t stride = elementSize;
        for (auto it = tensor.GetShape().rbegin(); it != tensor.GetShape().rend() && strides.size() < strideNum; ++it) {
            stride *= (*it);
            strides.push_back(stride);
        }
        std::reverse(strides.begin(), strides.end());
    }
    switch (tensor.GetDataType()) {
        case FlowFunc::TensorDataType::DT_BOOL: {
            return py::memoryview::from_buffer(tensor.GetData(), elementSize, py::format_descriptor<bool>::value,
                tensor.GetShape(), strides, true);
        }
        case FlowFunc::TensorDataType::DT_INT8: {
            return py::memoryview::from_buffer(tensor.GetData(), elementSize, py::format_descriptor<int8_t>::value,
                tensor.GetShape(), strides, true);
        }
        case FlowFunc::TensorDataType::DT_UINT8: {
            return py::memoryview::from_buffer(tensor.GetData(), elementSize, py::format_descriptor<uint8_t>::value,
                tensor.GetShape(), strides, true);
        }
        case FlowFunc::TensorDataType::DT_INT16: {
            return py::memoryview::from_buffer(tensor.GetData(), elementSize, py::format_descriptor<int16_t>::value,
                tensor.GetShape(), strides, true);
        }
        case FlowFunc::TensorDataType::DT_UINT16: {
            return py::memoryview::from_buffer(tensor.GetData(), elementSize, py::format_descriptor<uint16_t>::value,
                tensor.GetShape(), strides, true);
        }
        case FlowFunc::TensorDataType::DT_INT32: {
            return py::memoryview::from_buffer(tensor.GetData(), elementSize, py::format_descriptor<int32_t>::value,
                tensor.GetShape(), strides, true);
        }
        case FlowFunc::TensorDataType::DT_UINT32: {
            return py::memoryview::from_buffer(tensor.GetData(), elementSize, py::format_descriptor<uint32_t>::value,
                tensor.GetShape(), strides, true);
        }
        case FlowFunc::TensorDataType::DT_INT64: {
            return py::memoryview::from_buffer(tensor.GetData(), elementSize, py::format_descriptor<int64_t>::value,
                tensor.GetShape(), strides, true);
        }
        case FlowFunc::TensorDataType::DT_UINT64: {
            return py::memoryview::from_buffer(tensor.GetData(), elementSize, py::format_descriptor<uint64_t>::value,
                tensor.GetShape(), strides, true);
        }
        case FlowFunc::TensorDataType::DT_FLOAT16:
        case FlowFunc::TensorDataType::DT_FLOAT: {
            return py::memoryview::from_buffer(tensor.GetData(), elementSize, py::format_descriptor<float>::value,
                tensor.GetShape(), strides, true);
        }
        case FlowFunc::TensorDataType::DT_DOUBLE: {
            return py::memoryview::from_buffer(tensor.GetData(), elementSize, py::format_descriptor<double>::value,
                tensor.GetShape(), strides, true);
        }
        default: {
            return py::memoryview::from_memory(tensor.GetData(), tensor.GetDataSize(), true);
        }
    }
}

class PyFlowFuncLogger : public FlowFunc::FlowFuncLogger {
public:
    bool IsLogEnable(FlowFunc::FlowFuncLogLevel level) override
    {
        PYBIND11_OVERRIDE_PURE(bool, FlowFunc::FlowFuncLogger, IsLogEnable,);
    }
    void Error(const char *fmt, ...) override
    {
        PYBIND11_OVERRIDE_PURE(void, FlowFunc::FlowFuncLogger, Error,);
    }
    void Warn(const char *fmt, ...) override
    {
        PYBIND11_OVERRIDE_PURE(void, FlowFunc::FlowFuncLogger, Warn,);
    }
    void Info(const char *fmt, ...) override
    {
        PYBIND11_OVERRIDE_PURE(void, FlowFunc::FlowFuncLogger, Info,);
    }
    void Debug(const char *fmt, ...) override
    {
        PYBIND11_OVERRIDE_PURE(void, FlowFunc::FlowFuncLogger, Debug,);
    }
    void Error(const char *locationMessage, const char *userMessage)
    {
        auto &logger = GetLogger(FlowFunc::FlowFuncLogType::DEBUG_LOG);
        logger.Error("%s%s: %s", locationMessage, GetLogExtHeader(), userMessage);
    }
    void Warn(const char *locationMessage, const char *userMessage)
    {
        auto &logger = GetLogger(FlowFunc::FlowFuncLogType::DEBUG_LOG);
        logger.Warn("%s%s: %s", locationMessage, GetLogExtHeader(), userMessage);
    }
    void Info(const char *locationMessage, const char *userMessage)
    {
        auto &logger = GetLogger(FlowFunc::FlowFuncLogType::DEBUG_LOG);
        logger.Info("%s%s: %s", locationMessage, GetLogExtHeader(), userMessage);
    }
    void Debug(const char *locationMessage, const char *userMessage)
    {
        auto &logger = GetLogger(FlowFunc::FlowFuncLogType::DEBUG_LOG);
        logger.Debug("%s%s: %s", locationMessage, GetLogExtHeader(), userMessage);
    }
    void RunTypeError(const char *locationMessage, const char *userMessage)
    {
        auto &logger = GetLogger(FlowFunc::FlowFuncLogType::RUN_LOG);
        logger.Error("%s%s[RUN]: %s", locationMessage, GetLogExtHeader(), userMessage);
    }
    void RunTypeInfo(const char *locationMessage, const char *userMessage)
    {
        auto &logger = GetLogger(FlowFunc::FlowFuncLogType::RUN_LOG);
        logger.Info("%s%s[RUN]: %s", locationMessage, GetLogExtHeader(), userMessage);
    }
};

class PyFlowMsg : public FlowFunc::FlowMsg {
public:
    FlowFunc::MsgType GetMsgType() const override
    {
        PYBIND11_OVERRIDE_PURE(FlowFunc::MsgType, FlowFunc::FlowMsg, GetMsgType,);
    }

    FlowFunc::Tensor *GetTensor() const override
    {
        PYBIND11_OVERRIDE_PURE(FlowFunc::Tensor*, FlowFunc::FlowMsg, GetTensor,);
    }

    int32_t GetRetCode() const override
    {
        PYBIND11_OVERRIDE_PURE(int32_t, FlowMsg, GetRetCode,);
    }

    void SetRetCode(int32_t retCode) override
    {
        PYBIND11_OVERRIDE_PURE(void, FlowFunc::FlowMsg, SetRetCode, retCode);
    }

    void SetStartTime(uint64_t startTime) override
    {
        PYBIND11_OVERRIDE_PURE(void, FlowFunc::FlowMsg, SetStartTime, startTime);
    }

    uint64_t GetStartTime() const override
    {
        PYBIND11_OVERRIDE_PURE(uint64_t, FlowFunc::FlowMsg, GetStartTime,);
    }

    void SetEndTime(uint64_t endTime) override
    {
        PYBIND11_OVERRIDE_PURE(void, FlowFunc::FlowMsg, SetEndTime, endTime);
    }

    uint64_t GetEndTime() const override
    {
        PYBIND11_OVERRIDE_PURE(uint64_t, FlowFunc::FlowMsg, GetEndTime,);
    }

    void SetFlowFlags(uint32_t flags) override
    {
        PYBIND11_OVERRIDE_PURE(void, FlowFunc::FlowMsg, SetFlowFlags, flags);
    }

    uint32_t GetFlowFlags() const override
    {
        PYBIND11_OVERRIDE_PURE(uint32_t, FlowFunc::FlowMsg, GetFlowFlags,);
    }

    void SetRouteLabel(uint32_t routeLabel) override
    {
        PYBIND11_OVERRIDE_PURE(void, FlowFunc::FlowMsg, SetRouteLabel,);
    }

    uint64_t GetTransactionId() const override
    {
        PYBIND11_OVERRIDE_PURE(uint64_t, FlowFunc::FlowMsg, GetTransactionId,);
    }
};

class PyFlowMsgQueue : public FlowFunc::FlowMsgQueue {
public:
    int32_t Dequeue(std::shared_ptr<FlowFunc::FlowMsg> &flowMsg, int32_t timeout) override
    {
        PYBIND11_OVERRIDE_PURE(int32_t, FlowFunc::FlowMsgQueue, Dequeue,);
    }
 
    int32_t Depth() const override
    {
        PYBIND11_OVERRIDE_PURE(int32_t, FlowFunc::FlowMsgQueue, Depth,);
    }
 
    int32_t Size() const override
    {
        PYBIND11_OVERRIDE_PURE(int32_t, FlowFunc::FlowMsgQueue, Size,);
    }
};

class PyTensor : public FlowFunc::Tensor {
public:
    const std::vector<int64_t> &GetShape() const override
    {
        PYBIND11_OVERRIDE_PURE(const std::vector<int64_t>&, FlowFunc::Tensor, GetShape,);
    }

    FlowFunc::TensorDataType GetDataType() const override
    {
        PYBIND11_OVERRIDE_PURE(FlowFunc::TensorDataType, FlowFunc::Tensor, GetDataType,);
    }

    void *GetData() const override
    {
        PYBIND11_OVERRIDE_PURE(void*, FlowFunc::Tensor, GetData,);
    }

    uint64_t GetDataSize() const override
    {
        PYBIND11_OVERRIDE_PURE(uint64_t, FlowFunc::Tensor, GetDataSize,);
    }

    int64_t GetElementCnt() const override
    {
        PYBIND11_OVERRIDE_PURE(int64_t, FlowFunc::Tensor, GetElementCnt,);
    }

    int32_t Reshape(const std::vector<int64_t> &shape) override
    {
        PYBIND11_OVERRIDE_PURE(int32_t, FlowFunc::Tensor, Reshape,);
    }
};

class PyMetaParams : public FlowFunc::MetaParams {
public:
    const char *GetName() const override
    {
        PYBIND11_OVERRIDE_PURE(char *, FlowFunc::MetaParams, GetName,);
    }

    std::shared_ptr<const FlowFunc::AttrValue> GetAttr(const char *attrName) const
    {
        PYBIND11_OVERRIDE_PURE(std::shared_ptr<const FlowFunc::AttrValue>, FlowFunc::MetaParams, GetAttr,);
    }

    size_t GetInputNum() const
    {
        PYBIND11_OVERRIDE_PURE(size_t, FlowFunc::MetaParams, GetInputNum,);
    }

    size_t GetOutputNum() const
    {
        PYBIND11_OVERRIDE_PURE(size_t, FlowFunc::MetaParams, GetOutputNum,);
    }
    const char *GetWorkPath() const
    {
        PYBIND11_OVERRIDE_PURE(char *, FlowFunc::MetaParams, GetWorkPath,);
    }

    int32_t GetRunningDeviceId() const
    {
        PYBIND11_OVERRIDE_PURE(int32_t, FlowFunc::MetaParams, GetRunningDeviceId,);
    }

    int32_t GetRunningInstanceId() const
    {
        PYBIND11_OVERRIDE_PURE(int32_t, FlowFunc::MetaParams, GetRunningInstanceId,);
    }

    int32_t GetRunningInstanceNum() const
    {
        PYBIND11_OVERRIDE_PURE(int32_t, FlowFunc::MetaParams, GetRunningInstanceNum,);
    }
};

class PyMetaRunContext : public FlowFunc::MetaRunContext {
public:
    std::shared_ptr<FlowFunc::FlowMsg> AllocTensorMsg(const std::vector<int64_t> &shape,
                                                      FlowFunc::TensorDataType dataType) override
    {
        PYBIND11_OVERRIDE_PURE(std::shared_ptr<FlowFunc::FlowMsg>, FlowFunc::MetaRunContext, AllocTensorMsg,);
    }

    std::shared_ptr<FlowFunc::FlowMsg> AllocTensorMsgWithAlign(const std::vector<int64_t> &shape,
        FlowFunc::TensorDataType dataType, uint32_t align) override
    {
        PYBIND11_OVERRIDE_PURE(std::shared_ptr<FlowFunc::FlowMsg>, FlowFunc::MetaRunContext, AllocTensorMsg,);
    }

    int32_t SetOutput(uint32_t outIdx, std::shared_ptr<FlowFunc::FlowMsg> outMsg) override
    {
        PYBIND11_OVERRIDE_PURE(int32_t, FlowFunc::MetaRunContext, SetOutput,);
    }

    int32_t SetOutput(uint32_t outIdx, std::shared_ptr<FlowFunc::FlowMsg> outMsg,
        const FlowFunc::OutOptions &options) override
    {
        PYBIND11_OVERRIDE_PURE(int32_t, FlowFunc::MetaRunContext, SetOutput,);
    }

    int32_t SetMultiOutputs(uint32_t outIdx, const std::vector<std::shared_ptr<FlowFunc::FlowMsg>> &outMsgs,
        const FlowFunc::OutOptions &options) override
    {
        PYBIND11_OVERRIDE_PURE(int32_t, FlowFunc::MetaRunContext, SetMultiOutputs,);
    }

    std::shared_ptr<FlowFunc::FlowMsg> AllocEmptyDataMsg(FlowFunc::MsgType msgType) override
    {
        PYBIND11_OVERRIDE_PURE(std::shared_ptr<FlowFunc::FlowMsg>, FlowFunc::MetaRunContext, AllocEmptyDataMsg,);
    }

    int32_t RunFlowModel(const char *modelKey, const std::vector<std::shared_ptr<FlowFunc::FlowMsg>> &inputMsgs,
        std::vector<std::shared_ptr<FlowFunc::FlowMsg>> &outputMsgs, int32_t timeout) override
    {
        PYBIND11_OVERRIDE_PURE(int32_t, FlowFunc::MetaRunContext, RunFlowModel,);
    }

    int32_t GetUserData(void *data, size_t size, size_t offset = 0U) const override
    {
        PYBIND11_OVERRIDE_PURE(int32_t, FlowFunc::MetaRunContext, GetUserData);
    }
};

class FuncDataTypeManager {
public:
    static FuncDataTypeManager &GetInstance()
    {
        static FuncDataTypeManager dataTypeManager;
        return dataTypeManager;
    }

    void Init(const std::map<FlowFunc::TensorDataType, py::array> &typeMap)
    {
        for (const auto &item : typeMap) {
            auto const dtype = item.first;
            auto const array = item.second;
            const auto buff = array.request();
            funcDTypeToFormatDesc_[dtype] = buff.format;
        }
    }

    const std::map<FlowFunc::TensorDataType, std::string> &GetFlowFuncDtypeToFormatDesc() const
    {
        return funcDTypeToFormatDesc_;
    }

private:
    FuncDataTypeManager() = default;
    std::map<FlowFunc::TensorDataType, std::string> funcDTypeToFormatDesc_;
};

class PyBalanceConfig {
public:
    void SetAffinityPolicy(FlowFunc::AffinityPolicy policy)
    {
        policy_ = policy;
    }

    void SetBalanceWeight(int32_t rowNum, int32_t colNum)
    {
        rowNum_ = rowNum;
        colNum_ = colNum;
    }

    void SetDataPos(const std::vector<std::pair<int32_t, int32_t>> &dataPos)
    {
        dataPos_ = dataPos;
    }

    FlowFunc::AffinityPolicy GetAffinityPolicy() const
    {
        return policy_;
    }

    int32_t GetRowNum() const
    {
        return rowNum_;
    }

    int32_t GetColNum() const
    {
        return colNum_;
    }

    std::vector<std::pair<int32_t, int32_t>> GetDataPos() const
    {
        return dataPos_;
    }

private:
    FlowFunc::AffinityPolicy policy_;
    int32_t rowNum_ = 0;
    int32_t colNum_ = 0;
    std::vector<std::pair<int32_t, int32_t>> dataPos_;
};
constexpr uint32_t MAX_DIM_SIZE = 32U;
struct RuntimeTensorDesc {
    uint64_t dataAddr;
    int64_t dataOffsetSize;
    int64_t dtype;
    int64_t shape[MAX_DIM_SIZE + 1U];
    int64_t originalShape[MAX_DIM_SIZE + 1U];
    int64_t format;
    int64_t subFormat;
    uint64_t dataSize;
    uint8_t reserved[448]; // padding to 1024 bytes.
};

class RuntimeTensorDescMsgProcessor {
public:
    static int32_t GetRuntimeTensorDescs(const std::shared_ptr<FlowFunc::FlowMsg> &inputFlowMsg,
                                         std::vector<RuntimeTensorDesc> &runtimeTensorDescs, int64_t inputNum)
    {
        void *dataPtr = nullptr;
        uint64_t dataSize = 0UL;
        auto ret = inputFlowMsg->GetRawData(dataPtr, dataSize);
        if (ret != FlowFunc::FLOW_FUNC_SUCCESS) {
            UDF_LOG_ERROR("Failed to get raw data, ret = %d.", ret);
            return ret;
        }
        uint64_t offset = 0UL;
        for (int64_t i = 0; i < inputNum; ++i) {
            if (dataSize - offset < sizeof(RuntimeTensorDesc)) {
                UDF_LOG_ERROR("Failed to check flow msg size, data size = %lu, input num = %ld.", dataSize, inputNum);
                return FlowFunc::FLOW_FUNC_ERR_PARAM_INVALID;
            }
            auto desc = reinterpret_cast<RuntimeTensorDesc *>(static_cast<uint8_t *>(dataPtr) + offset);
            runtimeTensorDescs.emplace_back(*desc);
            offset += sizeof(RuntimeTensorDesc);
        }
        return FlowFunc::FLOW_FUNC_SUCCESS;
    }

    static std::shared_ptr<FlowFunc::FlowMsg> CreateRuntimeTensorDescMsg(
        const std::shared_ptr<FlowFunc::MetaRunContext> &runContext,
        const std::vector<RuntimeTensorDesc> &runtimeTensorDescs)
    {
        size_t size = runtimeTensorDescs.size() * sizeof(RuntimeTensorDesc);
        auto msg = runContext->AllocRawDataMsg(size);
        if (msg == nullptr) {
            UDF_LOG_ERROR("Failed to allocate raw data msg, size=%zu.", size);
            return nullptr;
        }

        void *dataPtr = nullptr;
        uint64_t dataSize = 0UL;
        auto ret = msg->GetRawData(dataPtr, dataSize);
        if (ret != FlowFunc::FLOW_FUNC_SUCCESS) {
            UDF_LOG_ERROR("Failed to get raw data, ret = %d.", ret);
            return nullptr;
        }
        if (size > dataSize) {
            UDF_LOG_ERROR("Failed to check flow msg size, alloc data size = %lu, but get data size = %lu.",
                          size, dataSize);
            return nullptr;
        }

        uint64_t offset = 0UL;
        for (const auto &runtimeTensorDesc : runtimeTensorDescs) {
            auto desc = reinterpret_cast<RuntimeTensorDesc *>(static_cast<uint8_t *>(dataPtr) + offset);
            *desc = runtimeTensorDesc;
            offset += sizeof(RuntimeTensorDesc);
        }
        return msg;
    }
};
}

PYBIND11_MODULE(flowfunc_wrapper, m) {
    m.doc() = "pybind11 flowfunc_wrapper plugin"; // optional module docstring
    m.attr("FLOW_FUNC_SUCCESS") = FlowFunc::FLOW_FUNC_SUCCESS;
    m.attr("FLOW_FUNC_FAILED") = FlowFunc::FLOW_FUNC_FAILED;
    m.attr("FLOW_FUNC_ERR_PARAM_INVALID") = FlowFunc::FLOW_FUNC_ERR_PARAM_INVALID;
    m.attr("FLOW_FUNC_ERR_ATTR_NOT_EXITS") = FlowFunc::FLOW_FUNC_ERR_ATTR_NOT_EXITS;
    m.attr("FLOW_FUNC_ERR_ATTR_TYPE_MISMATCH") = FlowFunc::FLOW_FUNC_ERR_ATTR_TYPE_MISMATCH;
    m.attr("FLOW_FUNC_ERR_TIME_OUT_ERROR") = FlowFunc::FLOW_FUNC_ERR_TIME_OUT_ERROR;
    m.attr("FLOW_FUNC_STATUS_REDEPLOYING") = FlowFunc::FLOW_FUNC_STATUS_REDEPLOYING;
    m.attr("FLOW_FUNC_STATUS_EXIT") = FlowFunc::FLOW_FUNC_STATUS_EXIT;
    m.attr("FLOW_FUNC_ERR_DRV_ERROR") = FlowFunc::FLOW_FUNC_ERR_DRV_ERROR;
    m.attr("FLOW_FUNC_ERR_QUEUE_ERROR") = FlowFunc::FLOW_FUNC_ERR_QUEUE_ERROR;
    m.attr("FLOW_FUNC_ERR_MEM_BUF_ERROR") = FlowFunc::FLOW_FUNC_ERR_MEM_BUF_ERROR;
    m.attr("FLOW_FUNC_ERR_EVENT_ERROR") = FlowFunc::FLOW_FUNC_ERR_EVENT_ERROR;
    m.attr("FLOW_FUNC_ERR_USER_DEFINE_START") = FlowFunc::FLOW_FUNC_ERR_USER_DEFINE_START;
    m.attr("FLOW_FUNC_ERR_USER_DEFINE_END") = FlowFunc::FLOW_FUNC_ERR_USER_DEFINE_END;
    py::enum_<FlowFunc::FlowFuncLogType>(m, "FlowFuncLogType", py::arithmetic())
        .value("DEBUG_LOG", FlowFunc::FlowFuncLogType::DEBUG_LOG)
        .value("RUN_LOG", FlowFunc::FlowFuncLogType::RUN_LOG)
        .export_values();
    py::enum_<FlowFunc::FlowFuncLogLevel>(m, "FlowFuncLogLevel", py::arithmetic())
        .value("DEBUG", FlowFunc::FlowFuncLogLevel::DEBUG)
        .value("INFO", FlowFunc::FlowFuncLogLevel::INFO)
        .value("WARN", FlowFunc::FlowFuncLogLevel::WARN)
        .value("ERROR", FlowFunc::FlowFuncLogLevel::ERROR)
        .export_values();
    py::enum_<FlowFunc::MsgType>(m, "MsgType", py::arithmetic())
        .value("MSG_TYPE_TENSOR_DATA", FlowFunc::MsgType::MSG_TYPE_TENSOR_DATA)
        .value("MSG_TYPE_RAW_MSG", FlowFunc::MsgType::MSG_TYPE_RAW_MSG)
        .value("MSG_TYPE_TORCH_TENSOR_MSG", static_cast<FlowFunc::MsgType>(1023))
        .value("MSG_TYPE_USER_DEFINE_START", static_cast<FlowFunc::MsgType>(1024))
        .value("MSG_TYPE_PICKLED_MSG", static_cast<FlowFunc::MsgType>(65535))
        .export_values();
    py::enum_<FlowFunc::FlowFlag>(m, "FlowFlag", py::arithmetic())
        .value("FLOW_FLAG_EOS", FlowFunc::FlowFlag::FLOW_FLAG_EOS)
        .value("FLOW_FLAG_SEG", FlowFunc::FlowFlag::FLOW_FLAG_SEG)
        .export_values();

    py::class_<FlowFunc::FlowFuncLogger, std::shared_ptr<FlowFunc::FlowFuncLogger>,
               PyFlowFuncLogger>(m, "FlowFuncLogger")
        .def(py::init())
        .def("get_log_header", [](FlowFunc::FlowFuncLogger &self) {
            const std::string LogHeader(self.GetLogExtHeader());
            return LogHeader;
        })
        .def("is_log_enable", [](FlowFunc::FlowFuncLogger &self, const FlowFunc::FlowFuncLogType &LogType,
                                 const FlowFunc::FlowFuncLogLevel &LogLevel) {
            FlowFunc::FlowFuncLogger &logger = self.GetLogger(LogType);
            return logger.IsLogEnable(LogLevel);
        })
        .def("debug_log_error", [](PyFlowFuncLogger &self, const char *locationMessage, const char *userMessage) {
            self.Error(locationMessage, userMessage);
        })
        .def("debug_log_info", [](PyFlowFuncLogger &self, const char *locationMessage, const char *userMessage) {
            self.Info(locationMessage, userMessage);
        })
        .def("debug_log_warn", [](PyFlowFuncLogger &self, const char *locationMessage, const char *userMessage) {
            self.Warn(locationMessage, userMessage);
        })
        .def("debug_log_debug", [](PyFlowFuncLogger &self, const char *locationMessage, const char *userMessage) {
            self.Debug(locationMessage, userMessage);
        })
        .def("run_log_error", [](PyFlowFuncLogger &self, const char *locationMessage, const char *userMessage) {
            self.RunTypeError(locationMessage, userMessage);
        })
        .def("run_log_info", [](PyFlowFuncLogger &self, const char *locationMessage, const char *userMessage) {
            self.RunTypeInfo(locationMessage, userMessage);
        })
        .def("__repr__", [](FlowFunc::FlowFuncLogger &self) {
            std::stringstream repr;
            repr << "FlowFuncLogger(LogHeader=" << self.GetLogExtHeader() << ")";
            return repr.str();
        });

    py::class_<FlowFunc::FlowMsg, std::shared_ptr<FlowFunc::FlowMsg>, PyFlowMsg>(m, "FlowMsg")
        .def(py::init<>())
        .def("get_msg_type", &FlowFunc::FlowMsg::GetMsgType)
        .def("set_msg_type", [](FlowFunc::FlowMsg &self, uint16_t msg_type) {
            return self.SetMsgType(static_cast<FlowFunc::MsgType>(msg_type));
        })
        .def("get_tensor", &FlowFunc::FlowMsg::GetTensor, py::return_value_policy::reference)
        .def("get_raw_data", [](FlowFunc::FlowMsg &self) {
            void *data = nullptr;
            uint64_t data_size = 0U;
            (void)self.GetRawData(data, data_size);
            return py::memoryview::from_memory(data, data_size, false);
        })
        .def("get_ret_code", &FlowFunc::FlowMsg::GetRetCode)
        .def("set_ret_code", &FlowFunc::FlowMsg::SetRetCode)
        .def("get_start_time", &FlowFunc::FlowMsg::GetStartTime)
        .def("set_start_time", &FlowFunc::FlowMsg::SetStartTime)
        .def("get_end_time", &FlowFunc::FlowMsg::GetEndTime)
        .def("set_end_time", &FlowFunc::FlowMsg::SetEndTime)
        .def("get_flow_flags", &FlowFunc::FlowMsg::GetFlowFlags)
        .def("set_flow_flags", &FlowFunc::FlowMsg::SetFlowFlags)
        .def("set_route_label", &FlowFunc::FlowMsg::SetRouteLabel)
        .def("get_transaction_id", &FlowFunc::FlowMsg::GetTransactionId)
        .def("set_transaction_id", &FlowFunc::FlowMsg::SetTransactionId)
        .def("__repr__", [](FlowFunc::FlowMsg &self) {
            std::stringstream repr;
            repr << "FlowMsg(msg_type=" << static_cast<int32_t>(self.GetMsgType());
            repr << ", tensor=...";
            repr << ", ret_code=" << self.GetRetCode();
            repr << ", start_time=" << self.GetStartTime();
            repr << ", end_time=" << self.GetEndTime();
            repr << ", flow_flags=" << self.GetFlowFlags() << ")";
            return repr.str();
        });

    py::class_<FlowFunc::FlowMsgQueue, std::shared_ptr<FlowFunc::FlowMsgQueue>, PyFlowMsgQueue>(m, "FlowMsgQueue")
        .def(py::init<>())
        .def("dequeue", [](FlowFunc::FlowMsgQueue &self, int32_t timeout) {
            std::shared_ptr<FlowFunc::FlowMsg> flowMsg;
            const auto ret = self.Dequeue(flowMsg, timeout);
            return std::make_tuple(ret, flowMsg);
        }, py::call_guard<py::gil_scoped_release>())
        .def("depth", &FlowFunc::FlowMsgQueue::Depth)
        .def("size", &FlowFunc::FlowMsgQueue::Size);

    py::class_<FlowFunc::FlowBufferFactory>(m, "FlowBufferFactory")
        .def_static("alloc_tensor", [](const std::vector<int64_t> &shapes,
                                       const ge::DataType &dtype, uint32_t align) {
            const auto func_dtype = TransGeDataTypeToFuncDataType(dtype);
            return FlowFunc::FlowBufferFactory::AllocTensor(shapes, func_dtype, align);
        }, py::return_value_policy::reference);

    py::class_<FlowFunc::Tensor, std::shared_ptr<FlowFunc::Tensor>, PyTensor>(m, "Tensor", py::buffer_protocol())
        .def(py::init<>())
        .def("get_shape", &FlowFunc::Tensor::GetShape)
        .def("get_dtype", [](FlowFunc::Tensor &self) {
            const auto f_dtype = self.GetDataType();
            const auto ge_dtype = TransFuncDataTypeToGeDataType(f_dtype);
            return ge_dtype;
        })
        .def("get_data", &ToReadonlyMemoryView)
        .def("get_writable_data", [](FlowFunc::Tensor &self) {
            return py::memoryview::from_memory(self.GetData(), self.GetDataSize(), false);
        })
        .def("get_data_size", &FlowFunc::Tensor::GetDataSize)
        .def("get_element_cnt", &FlowFunc::Tensor::GetElementCnt)
        .def("reshape", &FlowFunc::Tensor::Reshape)
        .def("__repr__", [](FlowFunc::Tensor &self) {
            std::stringstream repr;
            repr << "Tensor(shape=[";
            for (auto shapeItem : self.GetShape()) {
                repr << shapeItem << ", ";
            }
            repr << "], data_type=" << static_cast<int32_t>(self.GetDataType());
            repr << ", data_size=" << self.GetDataSize();
            repr << ", element_cnt=" << self.GetElementCnt();
            repr << ", data=...)";
            return repr.str();
        })
        .def_buffer([](const FlowFunc::Tensor& tensor) -> py::buffer_info {
                const auto dType = tensor.GetDataType();
                auto const &formatDescs = FuncDataTypeManager::GetInstance().GetFlowFuncDtypeToFormatDesc();
                const auto it = formatDescs.find(dType);
                if (it == formatDescs.cend()) {
                    throw std::runtime_error("Unsupported data type: " + std::to_string(static_cast<int32_t>(dType)));
                }
                const auto itemSize = static_cast<ssize_t>((tensor.GetDataSize() / tensor.GetElementCnt()));
                const auto shape = tensor.GetShape();
                std::vector<ssize_t> strides;
                const std::string errMsg = wrapper::ComputeStrides(itemSize, shape, strides);
                if (!errMsg.empty()) {
                    throw std::runtime_error(errMsg);
                }
                return py::buffer_info(tensor.GetData(),
                                       itemSize,
                                       it->second,
                                       static_cast<ssize_t>(shape.size()),
                                       shape,
                                       strides
                );
            });

    py::class_<FlowFunc::MetaParams, std::shared_ptr<FlowFunc::MetaParams>, PyMetaParams>(m, "MetaParams")
        .def(py::init<>())
        .def("get_name", &FlowFunc::MetaParams::GetName)
        .def("get_int64", [](FlowFunc::MetaParams &self, const char *name) {
            int64_t value = -1L;
            const auto ret = self.GetAttr<int64_t>(name, value);
            if (ret == FlowFunc::FLOW_FUNC_SUCCESS) {
                return std::make_tuple(FlowFunc::FLOW_FUNC_SUCCESS, value);
            }
            return  std::make_tuple(FlowFunc::FLOW_FUNC_ERR_ATTR_NOT_EXITS, value);
        })
        .def("get_int64_vector", [](FlowFunc::MetaParams &self, const char *name) {
            std::vector<int64_t> value;
            const auto ret = self.GetAttr<std::vector<int64_t>>(name, value);
            if (ret == FlowFunc::FLOW_FUNC_SUCCESS) {
                return std::make_tuple(FlowFunc::FLOW_FUNC_SUCCESS, value);
            }
            return  std::make_tuple(FlowFunc::FLOW_FUNC_ERR_ATTR_NOT_EXITS, value);
        })
        .def("get_int64_vector_vector", [](FlowFunc::MetaParams &self, const char *name) {
            std::vector<std::vector<int64_t>> value;
            const auto ret = self.GetAttr<std::vector<std::vector<int64_t>>>(name, value);
            if (ret == FlowFunc::FLOW_FUNC_SUCCESS) {
                return std::make_tuple(FlowFunc::FLOW_FUNC_SUCCESS, value);
            }
            return  std::make_tuple(FlowFunc::FLOW_FUNC_ERR_ATTR_NOT_EXITS, value);
        })
        .def("get_bool", [](FlowFunc::MetaParams &self, const char *name) {
            bool value;
            const auto ret = self.GetAttr<bool>(name, value);
            if (ret == FlowFunc::FLOW_FUNC_SUCCESS) {
                return std::make_tuple(FlowFunc::FLOW_FUNC_SUCCESS, value);
            }
            return  std::make_tuple(FlowFunc::FLOW_FUNC_ERR_ATTR_NOT_EXITS, value);
        })
        .def("get_bool_list", [](FlowFunc::MetaParams &self, const char *name) {
            std::vector<bool> value;
            const auto ret = self.GetAttr<std::vector<bool>>(name, value);
            if (ret == FlowFunc::FLOW_FUNC_SUCCESS) {
                return std::make_tuple(FlowFunc::FLOW_FUNC_SUCCESS, value);
            }
            return  std::make_tuple(FlowFunc::FLOW_FUNC_ERR_ATTR_NOT_EXITS, value);
        })
        .def("get_float", [](FlowFunc::MetaParams &self, const char *name) {
            float value;
            const auto ret = self.GetAttr<float>(name, value);
            if (ret == FlowFunc::FLOW_FUNC_SUCCESS) {
                return std::make_tuple(FlowFunc::FLOW_FUNC_SUCCESS, value);
            }
            return  std::make_tuple(FlowFunc::FLOW_FUNC_ERR_ATTR_NOT_EXITS, value);
        })
        .def("get_float_list", [](FlowFunc::MetaParams &self, const char *name) {
            std::vector<float> value;
            const auto ret = self.GetAttr<std::vector<float>>(name, value);
            if (ret == FlowFunc::FLOW_FUNC_SUCCESS) {
                return std::make_tuple(FlowFunc::FLOW_FUNC_SUCCESS, value);
            }
            return std::make_tuple(FlowFunc::FLOW_FUNC_ERR_ATTR_NOT_EXITS, value);
        })
        .def("get_tensor_dtype", [](FlowFunc::MetaParams &self, const char *name) {
            FlowFunc::TensorDataType value;
            const auto ret = self.GetAttr<FlowFunc::TensorDataType>(name, value);
            if (ret != FlowFunc::FLOW_FUNC_SUCCESS) {
                return std::make_tuple(ret, ge::DataType::DT_UNDEFINED);
            }
            return  std::make_tuple(ret, TransFuncDataTypeToGeDataType(value));
        })
        .def("get_tensor_dtype_list", [](FlowFunc::MetaParams &self, const char *name) {
            std::vector<FlowFunc::TensorDataType> value;
            std::vector<ge::DataType> GeDType;
            const auto ret = self.GetAttr<std::vector<FlowFunc::TensorDataType>>(name, value);
            if (ret != FlowFunc::FLOW_FUNC_SUCCESS) {
                return std::make_tuple(ret, GeDType);
            }
            for (const auto &f_dtype : value) {
                GeDType.emplace_back(TransFuncDataTypeToGeDataType(f_dtype));
            }
            return  std::make_tuple(ret, GeDType);
        })
        .def("get_string", [](FlowFunc::MetaParams &self, const char *name) {
            FlowFunc::AscendString value;
            const auto ret = self.GetAttr<FlowFunc::AscendString>(name, value);
            if (ret != FlowFunc::FLOW_FUNC_SUCCESS) {
                return std::make_tuple(ret, "");
            }
            return  std::make_tuple(ret, value.GetString());
        })
        .def("get_string_list", [](FlowFunc::MetaParams &self, const char *name) {
            std::vector<FlowFunc::AscendString> value;
            std::vector<std::string> StrList;
            const auto ret = self.GetAttr< std::vector<FlowFunc::AscendString>>(name, value);
            if (ret != FlowFunc::FLOW_FUNC_SUCCESS) {
                return std::make_tuple(ret, StrList);
            }
            for (const auto &str : value) {
                StrList.emplace_back(str.GetString());
            }
            return  std::make_tuple(ret, StrList);
        })
        .def("get_input_number", &FlowFunc::MetaParams::GetInputNum)
        .def("get_output_number", &FlowFunc::MetaParams::GetOutputNum)
        .def("get_work_path", &FlowFunc::MetaParams::GetWorkPath)
        .def("get_running_device_id", &FlowFunc::MetaParams::GetRunningDeviceId)
        .def("get_running_instance_id", &FlowFunc::MetaParams::GetRunningInstanceId)
        .def("get_running_instance_num", &FlowFunc::MetaParams::GetRunningInstanceNum)
        .def("__repr__", [](FlowFunc::MetaParams &self) {
            std::stringstream repr;
            repr << "MetaParams(name= " << self.GetName();
            repr << " , input_number=" << self.GetInputNum();
            repr << " , output_number=" << self.GetOutputNum();
            repr << ", working_path=" << self.GetWorkPath();
            repr << ", running_device_id=" << self.GetRunningDeviceId();
            repr << ", running_instance_id=" << self.GetRunningInstanceId();
            repr << ", running_instance_num=" << self.GetRunningInstanceNum() <<")";
            return repr.str();
        });

    py::enum_<FlowFunc::AffinityPolicy>(m, "AffinityPolicy")
        .value("NO_AFFINITY", FlowFunc::AffinityPolicy::NO_AFFINITY)
        .value("ROW_AFFINITY", FlowFunc::AffinityPolicy::ROW_AFFINITY)
        .value("COL_AFFINITY", FlowFunc::AffinityPolicy::COL_AFFINITY)
        .export_values();

    py::class_<PyBalanceConfig>(m, "BalanceConfig")
        .def(py::init<>())
        .def("set_data_pos", &PyBalanceConfig::SetDataPos)
        .def("set_balance_weight", &PyBalanceConfig::SetBalanceWeight)
        .def("set_affinity_policy", &PyBalanceConfig::SetAffinityPolicy);

    py::class_<FlowFunc::MetaRunContext, std::shared_ptr<FlowFunc::MetaRunContext>,
               PyMetaRunContext>(m, "MetaRunContext")
        .def(py::init<>())
        .def("alloc_tensor_msg", [](FlowFunc::MetaRunContext &self, const std::vector<int64_t> &shapes,
                                    const ge::DataType &dtype, uint32_t align) {
            const auto func_dtype = TransGeDataTypeToFuncDataType(dtype);
            return self.AllocTensorMsgWithAlign(shapes, func_dtype, align);
        }, py::return_value_policy::reference)
        .def("alloc_raw_data_msg", [](FlowFunc::MetaRunContext &self, int64_t size, uint32_t align) {
            return self.AllocRawDataMsg(size, align);
         }, py::return_value_policy::reference)
         .def("to_flow_msg", [](FlowFunc::MetaRunContext &self, std::shared_ptr<FlowFunc::Tensor> tensor) {
            return self.ToFlowMsg(tensor);
         }, py::return_value_policy::reference)
        .def("set_output",
            overload_cast_<uint32_t, std::shared_ptr<FlowFunc::FlowMsg>>()(&FlowFunc::MetaRunContext::SetOutput))
        .def("set_output", [](FlowFunc::MetaRunContext &self, uint32_t outIdx,
            std::shared_ptr<FlowFunc::FlowMsg> outMsg, const PyBalanceConfig &config) {
            FlowFunc::OutOptions options;
            auto *balanConfig = options.MutableBalanceConfig();
            balanConfig->SetAffinityPolicy(config.GetAffinityPolicy());
            balanConfig->SetDataPos(config.GetDataPos());
            FlowFunc::BalanceWeight balanceWeight;
            balanceWeight.rowNum = config.GetRowNum();
            balanceWeight.colNum = config.GetColNum();
            balanConfig->SetBalanceWeight(balanceWeight);
            return self.SetOutput(outIdx, outMsg, options);
        })
        .def("set_multi_outputs", [](FlowFunc::MetaRunContext &self, uint32_t outIdx,
            const std::vector<std::shared_ptr<FlowFunc::FlowMsg>> &outMsg, const PyBalanceConfig &config) {
            FlowFunc::OutOptions options;
            auto *balanConfig = options.MutableBalanceConfig();
            balanConfig->SetAffinityPolicy(config.GetAffinityPolicy());
            balanConfig->SetDataPos(config.GetDataPos());
            FlowFunc::BalanceWeight balanceWeight;
            balanceWeight.rowNum = config.GetRowNum();
            balanceWeight.colNum = config.GetColNum();
            balanConfig->SetBalanceWeight(balanceWeight);
            return self.SetMultiOutputs(outIdx, outMsg, options);
        })
        .def("alloc_empty_msg", &FlowFunc::MetaRunContext::AllocEmptyDataMsg, py::return_value_policy::reference)
        .def("run_flow_model", [](FlowFunc::MetaRunContext &self, const char *modelKey,
            std::vector<std::shared_ptr<FlowFunc::FlowMsg>> inputMsgs, int32_t timeout) {
            std::vector<std::shared_ptr<FlowFunc::FlowMsg>> outputMsgs;
            if (self.RunFlowModel(modelKey, inputMsgs, outputMsgs, timeout) == FlowFunc::FLOW_FUNC_SUCCESS) {
                return std::make_tuple(FlowFunc::FLOW_FUNC_SUCCESS, outputMsgs);
            }
            return std::make_tuple(FlowFunc::FLOW_FUNC_FAILED, std::vector<std::shared_ptr<FlowFunc::FlowMsg>>());
        }, py::return_value_policy::reference_internal)
        .def("get_user_data", [](FlowFunc::MetaRunContext &self, py::buffer user_data, size_t size, size_t offset) {
            void *data = reinterpret_cast<void *>(user_data.request().ptr);
            return self.GetUserData(data, size, offset);
        })
        .def("raise_exception", &PyMetaRunContext::RaiseException)
        .def("get_exception", [](FlowFunc::MetaRunContext &self) {
            int32_t expCode = 0;
            uint64_t userContextId = 0;
            bool ret = self.GetException(expCode, userContextId);
            return std::make_tuple(ret, expCode, userContextId);
        })
        .def("__repr__", [](FlowFunc::MetaRunContext &self) {
            std::stringstream repr;
            repr << "MetaRunContext()";
            return repr.str();
        });

    m.def("init_func_datatype_manager", [](const std::map<FlowFunc::TensorDataType, py::array> &typeMap) {
        FuncDataTypeManager::GetInstance().Init(typeMap);
    });

    py::class_<RuntimeTensorDesc>(m, "RuntimeTensorDesc")
        .def(py::init<>())
        .def_static("from_memory", [](py::buffer &buf) {
            py::buffer_info info = buf.request();
            if (static_cast<size_t>(info.size) < sizeof(RuntimeTensorDesc)) {
                throw std::runtime_error("Buffer size is less than sizeof(RuntimeTensorDesc)");
            }
            auto desc_view = static_cast<RuntimeTensorDesc *>(info.ptr);
            return *desc_view;
        })
        .def_readwrite("address", &RuntimeTensorDesc::dataAddr)
        .def_readwrite("dtype", &RuntimeTensorDesc::dtype)
        .def_readwrite("size", &RuntimeTensorDesc::dataSize)
        .def_property("shape", [](RuntimeTensorDesc &s)
                        { return std::vector<int64_t>(&s.shape[1], &s.shape[1 + s.shape[0]]); },
                        [](RuntimeTensorDesc &s, const std::vector<int64_t> &v) {
                            s.shape[0] =  v.size() > MAX_DIM_SIZE ? MAX_DIM_SIZE : v.size();
                            for (size_t i = 0; i < v.size(); ++i) {
                                s.shape[i + 1] = v[i];
                            } 
                        })
        .def("to_bytes", [](RuntimeTensorDesc &desc) {
            return py::bytes(reinterpret_cast<char *>(&desc), sizeof(RuntimeTensorDesc));
        });

    py::class_<RuntimeTensorDescMsgProcessor>(m, "RuntimeTensorDescMsgProcessor")
        .def_static("get_runtime_tensor_descs", [](const std::shared_ptr<FlowFunc::FlowMsg> &inputFlowMsg,
                                                   int64_t inputNum) {
            std::vector<RuntimeTensorDesc> runtimeTensorDescs;
            auto ret = RuntimeTensorDescMsgProcessor::GetRuntimeTensorDescs(inputFlowMsg, runtimeTensorDescs, inputNum);
            return std::make_tuple(ret, runtimeTensorDescs);
        })
        .def_static("create_runtime_tensor_desc_msg", &RuntimeTensorDescMsgProcessor::CreateRuntimeTensorDescMsg);
}