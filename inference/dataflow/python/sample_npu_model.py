"""
# Copyright 2025 Huawei Technologies Co., Ltd
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
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import dataflow as df

'''模型执行流程
             ┌─────────────┐    ┌──────────────────┐    ┌────────────────────┐    ┌──────────────┐
  模型输入--->│ preprocess  │--->│ FakeModel1 infer │--->│  FakeModel2 infer  │--->│  postprocess │--->模型输出
             └─────────────┘    └──────────────────┘    └────────────────────┘    └──────────────┘
'''


############## 模型定义 ##############
@df.npu_model(optimize_level=1)
class FakeModel1(nn.Module):
    def __init__(self):
        super().__init__()

    # 模拟模型推理
    @df.method()
    def forward(self, input_image):
        return F.interpolate(input_image, size=(256, 256), mode='bilinear')

@df.npu_model(optimize_level=2, input_descs=[df.TensorDesc(dtype=df.DT_FLOAT, shape=[1, 3, 256, 256])])
class FakeModel2(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean = 0.5
        self.std = 0.5

    # 模拟模型推理
    @df.method()
    def forward(self, input_image):
        return (input_image - self.mean) / self.std


############## UDF定义 ##############
# 预处理
@df.npu_model()
def preprocess(input_image):
    # 模拟图片裁切
    transform = transforms.Compose([transforms.CenterCrop(512)])
    return transform(input_image)

@df.npu_model()
def postprocess(input_image):
    mean = 0.5
    std = 0.5
    img = input_image * std + mean
    return F.interpolate(img, size=(512, 512), mode='bilinear')


if __name__ == '__main__':
    options = {
        "ge.experiment.data_flow_deploy_info_path": "./config/sample_npu_model_deploy_info.json",
    }
    df.init(options)

    ############## FlowGraph构图 ##############
    # 定义图的输入
    data0 = df.FlowData()

    # 构造flow node
    preprocess_node = preprocess.fnode()
    model1_node = FakeModel1.fnode()
    model2_node = FakeModel2.fnode()
    postprocess_node = postprocess.fnode()

    # 节点连边
    preprocess_output = preprocess_node(data0)
    model1_output = model1_node.forward(preprocess_output)
    model2_output = model2_node.forward(model1_output)
    postprocess_output = postprocess_node(model2_output)

    # 构造FlowGraph
    dag = df.FlowGraph([postprocess_output])

    # FlowGraph执行
    image = torch.randn(1, 3, 768, 768)
    dag.feed({data0: image})
    output = dag.fetch()

    print("TEST-OUTPUT:", output)
    df.finalize()
