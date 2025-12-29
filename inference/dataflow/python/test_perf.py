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
import numpy as np
import dataflow as df
import time

# dataflow 初始化参数设置
options = {
    "ge.exec.deviceId":"0",
    "ge.exec.logicalDeviceClusterDeployMode":"SINGLE",
    "ge.exec.logicalDeviceId":"[0:0]"
}
df.init(options)

# 定义输入
data0 = df.FlowData()
data1 = df.FlowData()

# 定义FuncProcessPoint
pp0 = df.FuncProcessPoint(compile_config_path='config/add_func.json')
pp0.set_init_param("out_type", df.DT_INT32)

# 创建计算节点
flow_node0 = df.FlowNode(input_num=2, output_num=1)
flow_node0.add_process_point(pp0)

# 构建连边关系
flow_node0_out = flow_node0(data0, data1)

# 构建FlowGraph
dag = df.FlowGraph([flow_node0_out])

# feed
feed_data = df.Tensor(np.array([4, 7, 5], dtype=np.int32), tensor_desc=df.TensorDesc(df.DT_INT32, [3]))

flow_info = df.FlowInfo()

# warm up
for i in range(3):
    dag.feed_data({data0:feed_data, data1:feed_data}, flow_info)

s = time.time()
for i in range(10):
    dag.feed_data({data0:feed_data, data1:feed_data}, flow_info)
e = time.time()

print(f"TEST-TIME: fetch cost {((e -s) * 1000000)} us")
print("TEST SUCCESS")

# 释放dataflow资源
df.finalize()
