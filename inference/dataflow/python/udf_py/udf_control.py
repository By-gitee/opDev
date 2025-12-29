"""
# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
import dataflow.flow_func as ff
import dataflow.data_type as dt
import numpy as np
import time

class UserFunc0():
    def __init__(self):
        pass

    @ff.init_wrapper()
    def init_flow_func(self, meta_params):
        logger = ff.FlowFuncLogger()
        logger.info("Start init")
        name = meta_params.get_name()
        logger.info("Func name [%s]" % name)
        name_attr = meta_params.get_attr_int("name")
        logger.info("Get attr int [%d]", name_attr[1])
        input_num = meta_params.get_input_num()
        logger.info("Input num [%d]", input_num)

        return ff.FLOW_FUNC_SUCCESS

    @ff.proc_wrapper("i0,o0,o1,o2,o3")
    def control_func(self, run_context, input_flow_msgs):
        logger = ff.FlowFuncLogger()
        for msg in input_flow_msgs:
            logger.info("Get msg code [%d]", msg.get_ret_code())
            if msg.get_ret_code() != ff.FLOW_FUNC_SUCCESS:
                logger.error("Invalid inputs: return code is not success.")
                return ff.FLOW_FUNC_FAILED
            
            if msg.get_msg_type() != ff.MSG_TYPE_TENSOR_DATA:
                logger.error("invalid input: msg type should be tensor data.")
                return ff.FLOW_FUNC_FAILED
            
        if input_flow_msgs.__len__() != 1:
            logger.error("Inputs should be 1.")
            return ff.FLOW_FUNC_FAILED
            
        tensor1 = input_flow_msgs[0].get_tensor()
        np1 = tensor1.numpy()

        out0 = run_context.alloc_tensor_msg([3], dt.DT_INT32)
        out1 = run_context.alloc_tensor_msg([3], dt.DT_INT32)
        data_size = out0.get_tensor().get_data_size()
        logger.info("Output data size is [%d].", data_size)
        ele_cnt = out0.get_tensor().get_element_cnt()
        logger.info("Element count is [%d].", ele_cnt)

        a = out0.get_tensor().numpy()
        a[...] = [1, 2, 3]
        b = out1.get_tensor().numpy()
        b[...] = [1, 1, 1]
        logger.info("Prepare to set output.")
        if np1 == [0]:
            if run_context.set_output(0, out0) != ff.FLOW_FUNC_SUCCESS:
                logger.error("Set output 0 failed.")
                return ff.FLOW_FUNC_FAILED
            if run_context.set_output(1, out1) != ff.FLOW_FUNC_SUCCESS:
                logger.error("Set output 1 failed.")
                return ff.FLOW_FUNC_FAILED
        else :
            if run_context.set_output(2, out0) != ff.FLOW_FUNC_SUCCESS:
                logger.error("Set output 0 failed.")
                return ff.FLOW_FUNC_FAILED
            if run_context.set_output(3, out0) != ff.FLOW_FUNC_SUCCESS:
                logger.error("Set output 1 failed.")
                return ff.FLOW_FUNC_FAILED
        return ff.FLOW_FUNC_SUCCESS