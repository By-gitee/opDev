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

class Add():
    def __init__(self):
        self.count = 0 

    @ff.init_wrapper()
    def init_flow_func(self, meta_params):
        return ff.FLOW_FUNC_SUCCESS

    @ff.proc_wrapper("i0,i1,o0")
    def add1(self, run_context, input_flow_msgs):
        return ff.FLOW_FUNC_SUCCESS
    
    @ff.proc_wrapper("i2,i3,o1")
    def add2(self, run_context, input_flow_msgs):
        return ff.FLOW_FUNC_SUCCESS