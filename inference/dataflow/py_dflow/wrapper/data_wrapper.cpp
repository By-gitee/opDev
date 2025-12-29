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

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "graph/types.h"
#include "utils.h"
#include "flow_func/tensor_data_type.h"

namespace {
namespace py = pybind11;
}

PYBIND11_MODULE(data_wrapper, m) {
    py::enum_<ge::DataType>(m, "DataType")
        .value("DT_FLOAT", ge::DataType::DT_FLOAT)
        .value("DT_FLOAT16", ge::DataType::DT_FLOAT16)
        .value("DT_BF16", ge::DataType::DT_BF16)
        .value("DT_INT8", ge::DataType::DT_INT8)
        .value("DT_INT16", ge::DataType::DT_INT16)
        .value("DT_UINT16", ge::DataType::DT_UINT16)
        .value("DT_UINT8", ge::DataType::DT_UINT8)
        .value("DT_INT32", ge::DataType::DT_INT32)
        .value("DT_INT64", ge::DataType::DT_INT64)
        .value("DT_UINT32", ge::DataType::DT_UINT32)
        .value("DT_UINT64", ge::DataType::DT_UINT64)
        .value("DT_BOOL", ge::DataType::DT_BOOL)
        .value("DT_DOUBLE", ge::DataType::DT_DOUBLE)
        .value("DT_STRING", ge::DataType::DT_STRING)
        .export_values();
    py::enum_<FlowFunc::TensorDataType>(m, "FuncDataType")
        .value("DT_FLOAT", FlowFunc::TensorDataType::DT_FLOAT)
        .value("DT_FLOAT16", FlowFunc::TensorDataType::DT_FLOAT16)
        .value("DT_BF16", FlowFunc::TensorDataType::DT_BF16)
        .value("DT_INT8", FlowFunc::TensorDataType::DT_INT8)
        .value("DT_INT16", FlowFunc::TensorDataType::DT_INT16)
        .value("DT_UINT16", FlowFunc::TensorDataType::DT_UINT16)
        .value("DT_UINT8", FlowFunc::TensorDataType::DT_UINT8)
        .value("DT_INT32", FlowFunc::TensorDataType::DT_INT32)
        .value("DT_INT64", FlowFunc::TensorDataType::DT_INT64)
        .value("DT_UINT32", FlowFunc::TensorDataType::DT_UINT32)
        .value("DT_UINT64", FlowFunc::TensorDataType::DT_UINT64)
        .value("DT_BOOL", FlowFunc::TensorDataType::DT_BOOL)
        .value("DT_DOUBLE", FlowFunc::TensorDataType::DT_DOUBLE)
        .export_values();
}