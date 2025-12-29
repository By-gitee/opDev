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
#define PYBIND11_DETAILED_ERROR_MESSAGES
#include <functional>
#include "pybind11/stl.h"
#include "pybind11/pybind11.h"
#include "pybind11/embed.h"
#include "meta_multi_func.h"
#include "flow_func_log.h"
__attribute__ ((constructor)) static void SoInit(void);
__attribute__ ((destructor)) static void SoDeinit(void);
void SoInit(void)
{
    Py_Initialize();
}

void SoDeinit(void)
{
    Py_Finalize();
}

namespace py = pybind11;

namespace FlowFunc {
namespace {
class ScopeGuard {
public:
    explicit ScopeGuard(std::function<void()> callback)
        : callback_(callback) {}

    ~ScopeGuard() noexcept
    {
        if (callback_ != nullptr) {
            callback_();
        }    
    }
private:
    std::function<void()> callback_;
};
}

class AddFlowFunc : public MetaMultiFunc {
public:
    AddFlowFunc()
    {
        Py_DECREF(PyImport_ImportModule("threading"));
        threadState_ = PyEval_SaveThread();
    }

    ~AddFlowFunc() override
    {
        pyModule_.release();
        PyEval_RestoreThread(threadState_);
        FLOW_FUNC_LOG_INFO("Finalize python interpreter.");
    }

    int32_t Init(const std::shared_ptr<MetaParams>& params) override
    {
        FLOW_FUNC_LOG_DEBUG("Init enter.");
        PyGILState_STATE gilState = PyGILState_Ensure();
        ScopeGuard gilGuard([&gilState]() {
            PyGILState_Release(gilState);
            FLOW_FUNC_LOG_INFO("PyGILState_Release.");
        });
        FLOW_FUNC_LOG_INFO("PyGILState_Ensure.");
        int32_t result = FLOW_FUNC_SUCCESS;
        try {
            PyRun_SimpleString("import sys");
            std::string append = std::string("sys.path.append('") + params->GetWorkPath() + "')";
            PyRun_SimpleString(append.c_str());

            constexpr const char *pyModuleName = "func_add";
            constexpr const char *pyClzName = "Add";
            auto module = py::module_::import(pyModuleName);
            pyModule_ = module.attr(pyClzName)();
            if (CheckProcExists() != FLOW_FUNC_SUCCESS) {
                FLOW_FUNC_LOG_ERROR("%s %s check proc exists failed.", pyModuleName, pyClzName);
                return FLOW_FUNC_FAILED;
            }
            if (py::hasattr(pyModule_, "init_flow_func")) {
                result = pyModule_.attr("init_flow_func")(params).cast<int32_t>();
                if (result != FLOW_FUNC_SUCCESS) {
                    FLOW_FUNC_LOG_ERROR("%s %s Init failed: %d", pyModuleName, pyClzName, result);
                    return result;
                }
                FLOW_FUNC_LOG_INFO("%s %s Init success.", pyModuleName, pyClzName);
            } else {
                FLOW_FUNC_LOG_INFO("%s %s has no init_flow_func method.", pyModuleName, pyClzName);
            }
        } catch (std::exception &e) {
            FLOW_FUNC_LOG_ERROR("Init failed: %s", e.what());
            result = FLOW_FUNC_FAILED;
        }
        return result;
    }

    int32_t Add1Proc(const std::shared_ptr<MetaRunContext> &runContext,
                     const std::vector<std::shared_ptr<FlowMsg>> &inputFlowMsgs)
    {
        FLOW_FUNC_LOG_DEBUG("Add1Proc enter.");
        PyGILState_STATE gilState = PyGILState_Ensure();
        ScopeGuard gilGuard([&gilState]() {
            PyGILState_Release(gilState);
            FLOW_FUNC_LOG_INFO("PyGILState_Release.");
        });
        FLOW_FUNC_LOG_INFO("PyGILState_Ensure.");
        int32_t result = FLOW_FUNC_SUCCESS;
        try
        {
            result = pyModule_.attr("add1")(runContext, inputFlowMsgs).cast<int32_t>();
            if (result != FLOW_FUNC_SUCCESS) {
                FLOW_FUNC_LOG_ERROR("Add1 failed: %d", result);
                return result;
            }
            FLOW_FUNC_LOG_INFO("Add1 success.");
        } catch(const std::exception& e)
        {
            FLOW_FUNC_LOG_ERROR("Proc failed: %s", e.what());
            result = FLOW_FUNC_FAILED;
        }
        return result;        
    }

    int32_t Add2Proc(const std::shared_ptr<MetaRunContext> &runContext,
                     const std::vector<std::shared_ptr<FlowMsg>> &inputFlowMsgs)
    {
        FLOW_FUNC_LOG_DEBUG("Add2Proc enter.");
        PyGILState_STATE gilState = PyGILState_Ensure();
        ScopeGuard gilGuard([&gilState]() {
            PyGILState_Release(gilState);
            FLOW_FUNC_LOG_INFO("PyGILState_Release.");
        });
        FLOW_FUNC_LOG_INFO("PyGILState_Ensure.");
        int32_t result = FLOW_FUNC_SUCCESS;
        try
        {
            result = pyModule_.attr("add2")(runContext, inputFlowMsgs).cast<int32_t>();
            if (result != FLOW_FUNC_SUCCESS) {
                FLOW_FUNC_LOG_ERROR("Add2 failed: %d", result);
                return result;
            }
            FLOW_FUNC_LOG_INFO("Add2 success.");
        } catch(const std::exception& e)
        {
            FLOW_FUNC_LOG_ERROR("Proc failed: %s", e.what());
            result = FLOW_FUNC_FAILED;
        }
        return result;        
    }
 
private:
    int32_t CheckProcExists() const{
        if(!py::hasattr(pyModule_, "add1")) {
            FLOW_FUNC_LOG_ERROR("There is no method named Add1.");
        }
        if(!py::hasattr(pyModule_, "add2")) {
            FLOW_FUNC_LOG_ERROR("There is no method named Add2.");
        }
        return FLOW_FUNC_SUCCESS;
    }

    py::object pyModule_;
    PyThreadState *threadState_ = nullptr;
};

FLOW_FUNC_REGISTRAR(AddFlowFunc)
    .RegProcFunc("Add1", &AddFlowFunc::Add1Proc)
    .RegProcFunc("Add2", &AddFlowFunc::Add2Proc);
}
