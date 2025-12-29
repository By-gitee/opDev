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

#include <algorithm>
#include <cmath>
#include <map>
#include <regex>
#include <string>
#include <tuple>
#include <vector>
#include <sys/syscall.h>
#include "utils.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "toolchain/slog.h"
#include "flow_graph/data_flow.h"
#include "ge/ge_api.h"
#include "parser/onnx_parser.h"
#include "parser/tensorflow_parser.h"

namespace {
namespace py = pybind11;
constexpr size_t kMaxUserDataSize = 64U;
using namespace ge;

struct ReturnMessage {
    uint32_t retCode;
    std::string errorMsg;
};

int64_t GetTid()
{
    thread_local static const int64_t tid = static_cast<int64_t>(syscall(__NR_gettid));
    return tid;
}
#define DFLOW_MODULE_NAME static_cast<int32_t>(GE)

#define DFLOW_LOGE(fmt, ...)    \
    dlog_error(DFLOW_MODULE_NAME, "[%s][tid:%ld]: " fmt, __FUNCTION__, GetTid(), ##__VA_ARGS__)

class DFlowDataTypeManager {
public:
    static DFlowDataTypeManager &GetInstance()
    {
        static DFlowDataTypeManager dataTypeManager;
        return dataTypeManager;
    }

    void Init(const std::map<ge::DataType, py::array> &typeMap)
    {
        for (const auto &item : typeMap) {
            auto const dtype = item.first;
            auto const array = item.second;
            numpyDTypeToGeDType_[array.dtype().char_()] = dtype;
            const auto buff = array.request();
            geDTypeToFormatDesc_[dtype] = buff.format;
        }
    }

    const std::map<char, ge::DataType> &GetNumpyDtypeToGeDType() const
    {
        return numpyDTypeToGeDType_;
    }

    const std::map<ge::DataType, std::string> &GetGeDtypeToFormatDesc() const
    {
        return geDTypeToFormatDesc_;
    }

private:
    DFlowDataTypeManager() = default;
    std::map<char, ge::DataType> numpyDTypeToGeDType_;
    std::map<ge::DataType, std::string> geDTypeToFormatDesc_;
};

std::string ConvertNumpyDataTypeToGeDataType(const py::dtype &npDataType, ge::DataType &geDataType)
{
    const auto &numpyDTypeToGeDType = DFlowDataTypeManager::GetInstance().GetNumpyDtypeToGeDType();
    const std::map<char, ge::DataType>::const_iterator it = numpyDTypeToGeDType.find(npDataType.char_());
    if (it != numpyDTypeToGeDType.cend()) {
        geDataType = it->second;
        return "";
    }
    return "Unsupported data type:" + npDataType.char_();
}

bool IsStringDataType(const std::string &data_type)
{
    const std::regex r("([^a-zA-Z])(S|U)[0-9]+");
    return std::regex_match(data_type, r);
};

const std::string ERR_MSG = "for details about the error information, see the ascend log.";

struct UserDataInfo {
    void *userDataPtr = nullptr;
    size_t dataSize = 0UL;
    size_t offset = 0UL;
};

struct FlowInfo {
    uint64_t startTime = 0UL;
    uint64_t endTime = 0UL;
    uint64_t transactionId = 0UL;
    uint32_t flowFlags = 0U;
    UserDataInfo userData;
};

struct DflowStringHead {
    int64_t addr;
    int64_t len;
};

std::vector<ge::AscendString> SplitToStrVector(const char* dataPtr, const size_t &dataSize, const size_t &elementNum)
{
    std::vector<ge::AscendString> res;
    const size_t byteNumPerElement = dataSize / elementNum;
    if (byteNumPerElement == 0UL) {
        return res;
    }
    for (size_t i = 0UL; i < elementNum; ++i) {
        res.emplace_back(dataPtr + i * byteNumPerElement);
    }
    return res;
}

template <typename... Args>
using overload_cast_ = pybind11::detail::overload_cast_impl<Args...>;

class PyFlowMsg : public ge::FlowMsg {
public:
    ge::MsgType GetMsgType() const override
    {
        PYBIND11_OVERRIDE_PURE(ge::MsgType, ge::FlowMsg, GetMsgType,);
    }

    void SetMsgType(ge::MsgType msgType) override
    {
        PYBIND11_OVERRIDE_PURE(void, ge::FlowMsg, SetMsgType, msgType);
    }

    ge::Tensor *GetTensor() const override
    {
        PYBIND11_OVERRIDE_PURE(ge::Tensor*, ge::FlowMsg, GetTensor,);
    }

    ge::Status GetRawData(void *&data_ptr, uint64_t &data_size) const override
    {
        PYBIND11_OVERRIDE_PURE(ge::Status, ge::FlowMsg, GetRawData,);
    }

    int32_t GetRetCode() const override
    {
        PYBIND11_OVERRIDE_PURE(int32_t, FlowMsg, GetRetCode,);
    }

    void SetRetCode(int32_t retCode) override
    {
        PYBIND11_OVERRIDE_PURE(void, ge::FlowMsg, SetRetCode, retCode);
    }

    void SetStartTime(uint64_t startTime) override
    {
        PYBIND11_OVERRIDE_PURE(void, ge::FlowMsg, SetStartTime, startTime);
    }

    uint64_t GetStartTime() const override
    {
        PYBIND11_OVERRIDE_PURE(uint64_t, ge::FlowMsg, GetStartTime,);
    }

    void SetEndTime(uint64_t endTime) override
    {
        PYBIND11_OVERRIDE_PURE(void, ge::FlowMsg, SetEndTime, endTime);
    }

    uint64_t GetEndTime() const override
    {
        PYBIND11_OVERRIDE_PURE(uint64_t, ge::FlowMsg, GetEndTime,);
    }

    void SetFlowFlags(uint32_t flags) override
    {
        PYBIND11_OVERRIDE_PURE(void, ge::FlowMsg, SetFlowFlags, flags);
    }

    uint32_t GetFlowFlags() const override
    {
        PYBIND11_OVERRIDE_PURE(uint32_t, ge::FlowMsg, GetFlowFlags,);
    }

    void SetTransactionId(uint64_t transactionId) override
    {
        PYBIND11_OVERRIDE_PURE(void, ge::FlowMsg, SetTransactionId, transactionId);
    }

    uint64_t GetTransactionId() const override
    {
        PYBIND11_OVERRIDE_PURE(uint64_t, ge::FlowMsg, GetTransactionId,);
    }

    ge::Status GetUserData(void *data, size_t size, size_t offset = 0U) const override
    {
        PYBIND11_OVERRIDE_PURE(ge::Status, FlowMsg, GetUserData,);
    }

    ge::Status SetUserData(const void *data, size_t size, size_t offset = 0U) override
    {
        PYBIND11_OVERRIDE_PURE(ge::Status, ge::FlowMsg, SetUserData, data, size, offset);
    }
};
}

PYBIND11_MODULE(dflow_wrapper, m) {
    m.attr("PARAM_INVALID") = ACL_ERROR_GE_PARAM_INVALID;
    m.attr("SHAPE_INVALID") = ACL_ERROR_GE_SHAPE_INVALID;
    m.attr("DATATYPE_INVALID") = ACL_ERROR_GE_DATATYPE_INVALID;
    m.attr("NOT_INIT") = ACL_ERROR_GE_EXEC_NOT_INIT;
    m.attr("INNER_ERROR") = ACL_ERROR_GE_INTERNAL_ERROR;
    m.attr("SUBHEALTHY") = ACL_ERROR_GE_SUBHEALTHY;
    py::enum_<ge::MsgType>(m, "MsgType", py::arithmetic())
        .value("MSG_TYPE_TENSOR_DATA", ge::MsgType::MSG_TYPE_TENSOR_DATA)
        .value("MSG_TYPE_RAW_MSG", ge::MsgType::MSG_TYPE_RAW_MSG)
        .export_values();

    py::class_<ReturnMessage>(m, "ReturnMessage")
        .def(py::init<uint32_t, std::string>())
        .def_readwrite("ret_code", &ReturnMessage::retCode)
        .def_readwrite("error_msg", &ReturnMessage::errorMsg);

    m.def("ge_initialize", [](const std::map<std::string, std::string> &options) {
        const auto ret = ge::GEInitialize(options);
        return ret;
    }, py::call_guard<py::gil_scoped_release>());

    m.def("ge_finalize", []() {
        const auto ret = ge::GEFinalize();
        return ret;
    }, py::call_guard<py::gil_scoped_release>());

    py::class_<dflow::ProcessPoint>(m, "ProcessPoint");

    py::class_<dflow::FunctionPp, dflow::ProcessPoint>(m, "FunctionPp")
        .def(py::init<const char *>())
        .def("set_compile_config", &dflow::FunctionPp::SetCompileConfig)
        // DataType和bool要定义在int64_前面，否则按照pybind11的匹配规则，会将DataType和bool匹配到int64_t
        .def("set_init_param", overload_cast_<const char *, const DataType &>()(&dflow::FunctionPp::SetInitParam))
        .def("set_init_param",
            overload_cast_<const char *, const std::vector<ge::DataType> &>()(&dflow::FunctionPp::SetInitParam))
        .def("set_init_param", overload_cast_<const char *, const char *>()(&dflow::FunctionPp::SetInitParam))
        .def("set_init_param",
            [](dflow::FunctionPp &self, const char *attrName, const std::vector<std::string> &values) {
                std::vector<AscendString> strValues;
                for (auto &value : values) {
                    strValues.emplace_back(AscendString(value.c_str()));
                }
                self.SetInitParam(attrName, strValues);
            })
        .def("set_init_param", overload_cast_<const char *, const bool &>()(&dflow::FunctionPp::SetInitParam))
        .def("set_init_param",
            overload_cast_<const char *, const std::vector<bool> &>()(&dflow::FunctionPp::SetInitParam))
        .def("set_init_param", overload_cast_<const char *, const int64_t &>()(&dflow::FunctionPp::SetInitParam))
        .def("set_init_param",
            overload_cast_<const char *, const std::vector<int64_t> &>()(&dflow::FunctionPp::SetInitParam))
        .def("set_init_param",
            overload_cast_<const char *, const std::vector<std::vector<int64_t>> &>()(&dflow::FunctionPp::SetInitParam))
        .def("set_init_param", overload_cast_<const char *, const float &>()(&dflow::FunctionPp::SetInitParam))
        .def("set_init_param",
            overload_cast_<const char *, const std::vector<float> &>()(&dflow::FunctionPp::SetInitParam))
        .def("add_invoked_closure",
            overload_cast_<const char *, const dflow::GraphPp &>()(&dflow::FunctionPp::AddInvokedClosure))
        .def("add_invoked_closure",
            overload_cast_<const char *, const dflow::FlowGraphPp &>()(&dflow::FunctionPp::AddInvokedClosure));

    m.def("load_graph_pp", [](const std::string &framework,
                               const std::string &graphFile,
                               const std::map<std::string, std::string> &loadParams,
                               const std::string &compileConfigPath,
                               const std::string &name) {
        std::map<ge::AscendString, ge::AscendString> params;
        if (!loadParams.empty()) {
            for (auto it = loadParams.cbegin(); it != loadParams.cend(); ++it) {
                AscendString key{it->first.data()};
                AscendString value{it->second.data()};
                params[key] = value;
            }
        }
        dflow::GraphBuilder errGraphBuild = []() {
            ge::Graph graph;
            return graph;
        };
        dflow::GraphPp errGraphPp{name.data(), errGraphBuild};

        static const std::set<std::string> supportFrameworks = {"tensorflow", "onnx", "mindspore"};
        if (supportFrameworks.find(framework) == supportFrameworks.cend()) {
            ReturnMessage returnMsg = {.retCode = ACL_ERROR_GE_PARAM_INVALID,
                .errorMsg = "Unsupported framework: " + framework};
            return std::make_tuple(returnMsg, errGraphPp);
        }

        dflow::GraphBuilder graphBuild = [framework, graphFile, params]() {
            ge::Graph graph;
            if (framework == "tensorflow") {
                const auto ret = aclgrphParseTensorFlow(graphFile.data(), params, graph);
                if (ret != ge::GRAPH_SUCCESS) {
                    DFLOW_LOGE("Failed to parse tensorflow model, file=%s, ret=%u", graphFile.c_str(), ret);
                }
            } else if (framework == "onnx") {
                const auto ret = aclgrphParseONNX(graphFile.data(), params, graph);
                if (ret != ge::GRAPH_SUCCESS) {
                    DFLOW_LOGE("Failed to parse onnx model, file=%s, ret=%u", graphFile.c_str(), ret);
                }
            } else if (framework == "mindspore") {
                const auto ret = graph.LoadFromFile(graphFile.data());
                if (ret != ge::GRAPH_SUCCESS) {
                    DFLOW_LOGE("Failed to parse mindspore model, file=%s, ret=%u", graphFile.c_str(), ret);
                }
            } else {
                DFLOW_LOGE("Unsupported framework, framework=%s, file=%s", framework.c_str(), graphFile.c_str());
            }
            return graph;
        };
        dflow::GraphPp graphPp{name.data(), graphBuild};
        (void) graphPp.SetCompileConfig(compileConfigPath.data());
        ReturnMessage returnMsg = {.retCode = ge::SUCCESS, .errorMsg = "success"};
        return std::make_tuple(returnMsg, graphPp);
    });

    m.def("load_flow_graph_pp", [](dflow::FlowGraph &flowGraph,
                                   const std::string &compileConfigPath,
                                   const std::string &name) {
        dflow::FlowGraphBuilder graphBuild = [flowGraph]() {
            return flowGraph;
        };
        dflow::FlowGraphPp flowGraphPp{name.data(), graphBuild};
        (void) flowGraphPp.SetCompileConfig(compileConfigPath.data());
        ReturnMessage returnMsg = {.retCode = ge::SUCCESS, .errorMsg = "success"};
        return std::make_tuple(returnMsg, flowGraphPp);
    });

    py::class_<dflow::GraphPp, dflow::ProcessPoint>(m, "GraphPp")
        .def(py::init<const char*, const dflow::GraphBuilder>())
        .def("set_compile_config", &dflow::GraphPp::SetCompileConfig);

    py::class_<dflow::FlowGraphPp, dflow::ProcessPoint>(m, "FlowGraphPp")
        .def(py::init<const char*, const dflow::FlowGraphBuilder>())
        .def("set_compile_config", &dflow::FlowGraphPp::SetCompileConfig);

    py::class_<Operator>(m, "Operator")
        .def("set_attr", overload_cast_<const char *, bool>()(&Operator::SetAttr))
        .def("set_attr", overload_cast_<const char *, int64_t>()(&Operator::SetAttr))
        .def("set_attr", overload_cast_<const std::string &, const std::string &>()(&Operator::SetAttr));

    py::class_<dflow::FlowOperator, Operator>(m, "FlowOperator");

    py::class_<dflow::FlowData, dflow::FlowOperator>(m, "FlowData")
        .def(py::init<const char *, int64_t>());

    py::class_<dflow::TimeBatch>(m, "TimeBatch")
        .def(py::init())
        .def_readwrite("time_window", &dflow::TimeBatch::time_window)
        .def_readwrite("batch_dim", &dflow::TimeBatch::batch_dim)
        .def_readwrite("drop_remainder", &dflow::TimeBatch::drop_remainder);

    py::class_<dflow::CountBatch>(m, "CountBatch")
        .def(py::init())
        .def_readwrite("batch_size", &dflow::CountBatch::batch_size)
        .def_readwrite("slide_stride", &dflow::CountBatch::slide_stride)
        .def_readwrite("timeout", &dflow::CountBatch::timeout)
        .def_readwrite("padding", &dflow::CountBatch::padding);

    py::enum_<dflow::DataFlowAttrType>(m, "DataFlowAttrType")
        .value("COUNT_BATCH", dflow::DataFlowAttrType::COUNT_BATCH)
        .value("TIME_BATCH", dflow::DataFlowAttrType::TIME_BATCH)
        .export_values();

    py::class_<dflow::DataFlowInputAttr>(m, "DataFlowInputAttr")
        .def(py::init())
        .def_readwrite("attr_type", &dflow::DataFlowInputAttr::attr_type)
        .def_readwrite("attr_value", &dflow::DataFlowInputAttr::attr_value);

    py::class_<dflow::FlowNode, dflow::FlowOperator>(m, "FlowNode")
        .def(py::init<const char *, uint32_t, uint32_t>())
        .def("set_input", &dflow::FlowNode::SetInput)
        .def("add_pp", &dflow::FlowNode::AddPp)
        .def("map_input", &dflow::FlowNode::MapInput)
        .def("map_output", &dflow::FlowNode::MapOutput)
        .def("set_balance_scatter", &dflow::FlowNode::SetBalanceScatter)
        .def("set_balance_gather", &dflow::FlowNode::SetBalanceGather);

    py::class_<dflow::FlowGraph>(m, "FlowGraph")
        .def(py::init<const char *>())
        .def("set_inputs", &dflow::FlowGraph::SetInputs)
        .def("set_outputs", overload_cast_<const std::vector<dflow::FlowOperator> &>()(&dflow::FlowGraph::SetOutputs))
        .def("set_outputs", overload_cast_<const std::vector<std::pair<dflow::FlowOperator,
                            std::vector<size_t>>> &>()(&dflow::FlowGraph::SetOutputs))
        .def("set_contains_n_mapping_node", &dflow::FlowGraph::SetContainsNMappingNode)
        .def("set_inputs_align_attrs", &dflow::FlowGraph::SetInputsAlignAttrs)
        .def("set_exception_catch", &dflow::FlowGraph::SetExceptionCatch)
        .def("set_graphpp_builder_async", &dflow::FlowGraph::SetGraphPpBuilderAsync);

    py::class_<FlowInfo>(m, "FlowInfo")
        .def(py::init())
        .def_readwrite("start_time", &FlowInfo::startTime)
        .def_readwrite("end_time", &FlowInfo::endTime)
        .def_readwrite("flow_flags", &FlowInfo::flowFlags)
        .def_readwrite("transaction_id", &FlowInfo::transactionId)
        .def("set_user_data", [](FlowInfo &self, py::buffer user_data, size_t data_size, size_t offset) {
            self.userData.userDataPtr = reinterpret_cast<void *>(user_data.request().ptr);
            self.userData.dataSize = data_size;
            self.userData.offset = offset;
        });

    py::class_<ge::Tensor>(m, "Tensor", py::buffer_protocol())
        .def(py::init([](py::array &npArray) {
            auto flags = static_cast<unsigned int>(npArray.flags());
            if ((flags & pybind11::detail::npy_api::NPY_ARRAY_C_CONTIGUOUS_) == 0) {
                throw std::runtime_error("Numpy array is not C Contiguous");
            }
            ge::DataType dtype = ge::DataType::DT_FLOAT;
            if (IsStringDataType(py::str(npArray.dtype()))) {
                dtype = ge::DataType::DT_STRING;
            } else {
                const auto retMsg = ConvertNumpyDataTypeToGeDataType(npArray.dtype(), dtype);
                if (!retMsg.empty()) {
                    throw std::runtime_error(retMsg);
                }
            }
            std::vector<int64_t> dims;
            for (ssize_t i = 0; i < npArray.ndim(); ++i) {
                dims.emplace_back(static_cast<int64_t>(npArray.shape(i)));
            }
            ge::TensorDesc desc(ge::Shape(dims), ge::FORMAT_ND, dtype);
            ge::Tensor tensor;
            tensor.SetTensorDesc(desc);

            if (dtype == ge::DataType::DT_STRING) {
                const int64_t shapeSize = desc.GetShape().GetShapeSize();
                const size_t elementNumber = shapeSize <= 0L ? 1UL : static_cast<size_t>(shapeSize);
                const auto stringVec = SplitToStrVector(reinterpret_cast<const char *>(npArray.data()),
                                                        npArray.nbytes(), elementNumber);
                if (stringVec.empty()) {
                    throw std::runtime_error("Split string to vector failed.");
                }
                tensor.SetData(stringVec);
            } else {
                tensor.SetData(reinterpret_cast<const uint8_t *>(npArray.data()), npArray.nbytes());
            }
            return tensor;
        }))
        .def("get_dtype", [](const ge::Tensor &self) {
            const auto tensorDesc = self.GetTensorDesc();
            return tensorDesc.GetDataType();
        })
        .def("get_shape", [](const ge::Tensor &self) {
            const auto tensorDesc = self.GetTensorDesc();
            return tensorDesc.GetShape().GetDims();
        })
        .def("clone", [](const ge::Tensor &self) {
            return self.Clone();
        })
        .def("get_string_tensor", [](const ge::Tensor &tensor) {
            const auto tensorDesc = tensor.GetTensorDesc();
            const int64_t shapeSize = tensorDesc.GetShape().GetShapeSize();
            const size_t elementNumber = shapeSize <= 0L ? 1UL : static_cast<size_t>(shapeSize);
            if (wrapper::CheckInt64MulOverflow(elementNumber, sizeof(DflowStringHead))) {
                throw std::runtime_error("element number " + std::to_string(elementNumber) +
                    " mul DflowStringHead size " + std::to_string(sizeof(DflowStringHead)) + " is overflow.");
            }
            uint64_t totalHeaderSize = elementNumber * sizeof(DflowStringHead);
            if (totalHeaderSize > tensor.GetSize()) {
                throw std::runtime_error("Total ptr size " + std::to_string(totalHeaderSize) +
                                         " is greater than data size " + std::to_string(tensor.GetSize()));
            }
            if (tensor.GetData() == nullptr) {
                throw std::runtime_error("Data tensor nullptr is invalid.");
            }
            std::vector<std::string> tensorStrs;
            for (size_t i = 0; i < elementNumber; ++i) {
                auto header = reinterpret_cast<const DflowStringHead *>(tensor.GetData()) + i;
                tensorStrs.emplace_back(reinterpret_cast<const char *>(tensor.GetData() + header->addr));
            }
            return tensorStrs;
        })
        .def_buffer([](ge::Tensor &tensor) -> py::buffer_info {
            const auto tensorDesc = tensor.GetTensorDesc();
            const auto dType = tensorDesc.GetDataType();
            auto const &formatDescs = DFlowDataTypeManager::GetInstance().GetGeDtypeToFormatDesc();
            const std::map<ge::DataType, std::string>::const_iterator it = formatDescs.find(dType);
            if (it == formatDescs.cend()) {
                throw std::runtime_error("Unsupported data type: " + std::to_string(static_cast<int32_t>(dType)));
            }
            const ssize_t itemSize = static_cast<ssize_t>(ge::GetSizeByDataType(dType));
            const auto shape = tensorDesc.GetShape();
            const auto dims = shape.GetDims();
            std::vector<ssize_t> strides;
            const std::string errMsg = wrapper::ComputeStrides(itemSize, dims, strides);
            if (!errMsg.empty()) {
                throw std::runtime_error(errMsg);
            }
            return py::buffer_info(tensor.GetData(),
                itemSize,
                it->second,
                static_cast<ssize_t>(shape.GetDimNum()),
                dims,
                strides
            );
        });

    py::class_<ge::FlowMsg, std::shared_ptr<ge::FlowMsg>, PyFlowMsg>(m, "FlowMsg")
        .def(py::init<>())
        .def("get_msg_type", &ge::FlowMsg::GetMsgType)
        .def("set_msg_type", [](ge::FlowMsg &self, uint16_t msg_type) {
            return self.SetMsgType(static_cast<ge::MsgType>(msg_type));
        })
        .def("get_tensor", &ge::FlowMsg::GetTensor, py::return_value_policy::reference)
        .def("get_raw_data", [](ge::FlowMsg &self) {
            void *data = nullptr;
            uint64_t data_size = 0U;
            (void)self.GetRawData(data, data_size);
            return py::memoryview::from_memory(data, data_size, false);
        })
        .def("get_ret_code", &ge::FlowMsg::GetRetCode)
        .def("set_ret_code", &ge::FlowMsg::SetRetCode)
        .def("get_start_time", &ge::FlowMsg::GetStartTime)
        .def("set_start_time", &ge::FlowMsg::SetStartTime)
        .def("get_end_time", &ge::FlowMsg::GetEndTime)
        .def("set_end_time", &ge::FlowMsg::SetEndTime)
        .def("get_flow_flags", &ge::FlowMsg::GetFlowFlags)
        .def("set_flow_flags", &ge::FlowMsg::SetFlowFlags)
        .def("get_transaction_id", &ge::FlowMsg::GetTransactionId)
        .def("set_transaction_id", &ge::FlowMsg::SetTransactionId)
        .def("__repr__", [](ge::FlowMsg &self) {
            std::stringstream repr;
            repr << "FlowMsg(msg_type=" << static_cast<int32_t>(self.GetMsgType());
            repr << ", tensor=...";
            repr << ", ret_code=" << self.GetRetCode();
            repr << ", start_time=" << self.GetStartTime();
            repr << ", end_time=" << self.GetEndTime();
            repr << ", transaction_id=" << self.GetTransactionId();
            repr << ", flow_flags=" << self.GetFlowFlags() << ")";
            return repr.str();
        });

    py::class_<ge::FlowBufferFactory>(m, "FlowBufferFactory")
        .def_static("alloc_tensor_msg", &ge::FlowBufferFactory::AllocTensorMsg)
        .def_static("alloc_raw_data_msg", &ge::FlowBufferFactory::AllocRawDataMsg)
        .def_static("alloc_empty_data_msg", &ge::FlowBufferFactory::AllocEmptyDataMsg)
        .def_static("to_tensor_flow_msg", [](const ge::Tensor &tensor) {
            return ge::FlowBufferFactory::ToFlowMsg(tensor);
        })
        .def_static("to_raw_data_flow_msg", [](py::buffer buffer) {
            py::buffer_info info = buffer.request();
            ge::RawData raw_data{};
            raw_data.addr = static_cast<const void *>(info.ptr);
            raw_data.len = info.size;
            return ge::FlowBufferFactory::ToFlowMsg(raw_data);
        });

    py::class_<Session>(m, "Session")
        .def(py::init<const std::map<std::string, std::string> &>())
        .def("add_flow_graph", ([](ge::Session &self, uint32_t graphId, dflow::FlowGraph &flowGraph,
                                   const std::map<std::string, std::string> &options) {
            ReturnMessage returnMsg = {.retCode = ge::SUCCESS, .errorMsg = "success"};
            const auto ret = self.AddGraph(graphId, flowGraph.ToGeGraph(), options);
            if (ret != 0) {
                returnMsg.retCode = ret;
                returnMsg.errorMsg = "Failed to add flow graph, " + ERR_MSG;
            }
            return returnMsg;
        }))
        .def("feed_data", [](ge::Session &self, uint32_t graphId, std::vector<uint32_t> &indexes,
                             const std::vector<ge::Tensor> &inputs, FlowInfo &info, int32_t timeout) {
            DataFlowInfo flowInfo;
            flowInfo.SetStartTime(info.startTime);
            flowInfo.SetEndTime(info.endTime);
            flowInfo.SetFlowFlags(info.flowFlags);
            flowInfo.SetTransactionId(info.transactionId);
            ReturnMessage returnMsg = {.retCode = ge::SUCCESS, .errorMsg = "success"};
            if (info.userData.dataSize != 0UL) {
                const auto setRet = flowInfo.SetUserData(info.userData.userDataPtr,
                                                         info.userData.dataSize,
                                                         info.userData.offset);
                if (setRet != SUCCESS) {
                    returnMsg.retCode = setRet;
                    returnMsg.errorMsg = "Failed to set user data, " + ERR_MSG;
                    return returnMsg;
                }
            }
            const auto ret = self.FeedDataFlowGraph(graphId, indexes, inputs, flowInfo, timeout);
            if ((ret != ge::SUCCESS)) {
                returnMsg.retCode = ret;
                if (ret == ACL_ERROR_GE_SUBHEALTHY) {
                    returnMsg.errorMsg = "Current system is in subhealth status.";
                } else {
                    returnMsg.errorMsg = "Failed to feed data, "+ ERR_MSG;
                }
            }
            return returnMsg;
        }, py::call_guard<py::gil_scoped_release>())
        .def("feed_flow_msg", [](ge::Session &self, uint32_t graphId, std::vector<uint32_t> &indexes,
                                 const std::vector<ge::FlowMsgPtr> &inputs, int32_t timeout) {
            ReturnMessage returnMsg = {.retCode = ge::SUCCESS, .errorMsg = "success"};
            const auto ret = self.FeedDataFlowGraph(graphId, indexes, inputs, timeout);
            if ((ret != ge::SUCCESS)) {
                returnMsg.retCode = ret;
                if (ret == ACL_ERROR_GE_SUBHEALTHY) {
                    returnMsg.errorMsg = "Current system is in subhealth status.";
                } else {
                    returnMsg.errorMsg = "Failed to feed flow msg, "+ ERR_MSG;
                }
            }
            return returnMsg;
        }, py::call_guard<py::gil_scoped_release>())
        .def("fetch_data", [](ge::Session &self, uint32_t graphId, std::vector<uint32_t> &indexes,
                              int32_t timeout, py::buffer user_data) {
            const size_t userDataSize = user_data.request().size;
            ReturnMessage returnMsg = {.retCode = ge::SUCCESS, .errorMsg = "success"};
            std::vector<ge::Tensor> outputs;
            FlowInfo info;
            if (userDataSize > kMaxUserDataSize) {
                returnMsg.retCode = ACL_ERROR_GE_PARAM_INVALID;
                returnMsg.errorMsg = "The size of user data is greater than limit value." + ERR_MSG;
                return std::make_tuple(returnMsg, outputs, info);
            }
            ge::DataFlowInfo flowInfo;
            const auto ret = self.FetchDataFlowGraph(graphId, indexes, outputs, flowInfo, timeout);

            info.startTime = flowInfo.GetStartTime();
            info.endTime = flowInfo.GetEndTime();
            info.flowFlags = flowInfo.GetFlowFlags();
            info.transactionId = flowInfo.GetTransactionId();
            if (userDataSize > 0) {
                (void)flowInfo.GetUserData(user_data.request().ptr, userDataSize);
            }
            if ((ret != ge::SUCCESS) && (ret != ACL_ERROR_GE_SUBHEALTHY)) {
                returnMsg.retCode = ret;
                returnMsg.errorMsg = "Failed to fetch data, " + ERR_MSG;
                return std::make_tuple(returnMsg, outputs, info);
            }
            if (ret == ACL_ERROR_GE_SUBHEALTHY) {
                returnMsg.retCode = ret;
                returnMsg.errorMsg = "Current system is in subhealth status.";
            }
            return std::make_tuple(returnMsg, outputs, info);
        }, py::call_guard<py::gil_scoped_release>())
        .def("fetch_flow_msg", [](ge::Session &self, uint32_t graphId, std::vector<uint32_t> &indexes,
                                  int32_t timeout) {
            ReturnMessage returnMsg = {.retCode = ge::SUCCESS, .errorMsg = "success"};
            std::vector<ge::FlowMsgPtr> outputs;
            ge::DataFlowInfo flowInfo;
            const auto ret = self.FetchDataFlowGraph(graphId, indexes, outputs, timeout);
            if ((ret != ge::SUCCESS) && (ret != ACL_ERROR_GE_SUBHEALTHY)) {
                returnMsg.retCode = ret;
                returnMsg.errorMsg = "Failed to fetch data, " + ERR_MSG;
                return std::make_tuple(returnMsg, outputs);
            }
            if (ret == ACL_ERROR_GE_SUBHEALTHY) {
                returnMsg.retCode = ret;
                returnMsg.errorMsg = "Current system is in subhealth status.";
            }
            return std::make_tuple(returnMsg, outputs);
        }, py::call_guard<py::gil_scoped_release>());
        m.def("init_datatype_manager", [](const std::map<ge::DataType, py::array> &typeMap) {
            DFlowDataTypeManager::GetInstance().Init(typeMap);
        });
}
