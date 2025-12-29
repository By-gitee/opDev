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

# dataflow 初始化参数按需设置
options = {
    "ge.exec.deviceId":"0",
    "ge.exec.logicalDeviceClusterDeployMode":"SINGLE",
    "ge.exec.logicalDeviceId":"[0:0]"
}
df.init(options)

# 构图
'''
      FlowData     FlowData
       |    \       /  |
       |     \     /   |
       |      \   /    |
       |       \ /     |
       |        /\     |
       |       /  \    |
       |      /    \   |
      FlowNode     FlowNode
FuncProcessPoint   FuncProcessPoint
       \              /
    timeBatch        /
          \         /
           \       /
           FlowNode
      GraphProcessPoint
'''

# 定义输入
data0 = df.FlowData()
data1 = df.FlowData()

# 定义FuncProcessPoint实现Add功能的FlowNode
pp0 = df.FuncProcessPoint(compile_config_path="config/add_func.json")
pp0.set_init_param("out_type", df.DT_UINT32) #按UDF实际实现设置

flow_node0 = df.FlowNode(input_num=2, output_num=1)
flow_node0.add_process_point(pp0)
# 给flowNode设置flowAttr相关属性，按需设置
flow_node0.set_attr("_flow_attr", True)
flow_node0.set_attr("_flow_attr_depth", 108)
flow_node0.set_attr("_flow_attr_enqueue_policy", "FIFO")

# 定义FuncProcessPoint调用实现add功能的GraphProcessPoint
pp1 = df.FuncProcessPoint(compile_config_path="config/invoke_func.json")
pp2 = df.GraphProcessPoint(df.Framework.TENSORFLOW, "config/add", load_params={"input_data_names":"Placeholder,Placeholder_1"},
                           compile_config_path="config/add_graph.json")
pp1.add_invoked_closure("invoke_graph", pp2)

flow_node1 = df.FlowNode(input_num=2, output_num=1)
flow_node1.add_process_point(pp1)

# 定义GraphProcessPoint实现Add功能的FlowNode
pp3 = df.GraphProcessPoint(df.Framework.TENSORFLOW, "config/add", load_params={"input_data_names":"Placeholder,Placeholder_1"},
                           compile_config_path="config/add_graph.json")
flow_node2 = df.FlowNode(input_num=2, output_num=1)
flow_node2.add_process_point(pp3)

# 为flowNode2的一个输入 添加timeBatch属性 实际场景按需添加，此处仅提供使用样例
time_batch= df.TimeBatch()
time_batch.time_window = 5
time_batch.batch_dim = 0
flow_node2.map_input(0, pp3, 0, [time_batch])
flow_node2.map_input(1, pp3, 1)

# 构建连边关系
flow_node0_out = flow_node0(data0, data1)
flow_node1_out = flow_node1(data0, data1)
flow_node2_out = flow_node2(flow_node0_out, flow_node1_out)

# 构建FlowGraph
dag = df.FlowGraph([flow_node2_out])

# feed data
feed_data0 = np.ones([3], dtype=np.uint32)
feed_data1 = np.array([1, 2, 3], dtype=np.uint32)

flow_info = df.FlowInfo()
flow_info.start_time = 0
flow_info.end_time = 5

# 异步喂数据
dag.feed_data({data0:feed_data0, data1:feed_data1}, flow_info)

# 获取输出
result = dag.fetch_data()

# check output 
print("TEST-OUTPUT:", result)

# 释放
df.finalize()
