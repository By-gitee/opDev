# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import acl
import numpy as np
from PIL import Image
from constant import ACL_MEM_MALLOC_HUGE_FIRST, \
    ACL_MEMCPY_HOST_TO_DEVICE, ACL_MEMCPY_DEVICE_TO_HOST


class Net:
    def __init__(self, model_path):
        # 初始化函数
        self.device_id = 0  
        # step1: 初始化
        ret = acl.init()
        # 指定运算的Device
        ret = acl.rt.set_device(self.device_id) 
        # step2: 加载模型，本示例为ResNet-50模型
        # 加载离线模型文件，返回标识模型的ID
        self.model_id, ret = acl.mdl.load_from_file(model_path)
        # 创建空白模型描述信息，获取模型描述信息的指针地址
        self.model_desc = acl.mdl.create_desc()
        # 通过模型的ID，将模型的描述信息填充到model_desc
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)  
        # step3：创建输入输出数据集
        # 创建输入数据集
        self.input_dataset, self.input_data = self.prepare_dataset('input')
        # 创建输出数据集
        self.output_dataset, self.output_data = self.prepare_dataset('output')

    def prepare_dataset(self, io_type):
        # 准备数据集
        if io_type == "input":
            # 获得模型输入的个数
            io_num = acl.mdl.get_num_inputs(self.model_desc)
            acl_mdl_get_size_by_index = acl.mdl.get_input_size_by_index
        else:
            # 获得模型输出的个数
            io_num = acl.mdl.get_num_outputs(self.model_desc)
            acl_mdl_get_size_by_index = acl.mdl.get_output_size_by_index
        # 创建aclmdlDataset类型的数据，描述模型推理的输入。
        dataset = acl.mdl.create_dataset()
        datas = []
        for i in range(io_num):
            # 获取所需的buffer内存大小
            buffer_size = acl_mdl_get_size_by_index(self.model_desc, i)
            # 申请buffer内存
            buffer, ret = acl.rt.malloc(buffer_size, ACL_MEM_MALLOC_HUGE_FIRST)
            # 从内存创建buffer数据
            data_buffer = acl.create_data_buffer(buffer, buffer_size)
            # 将buffer数据添加到数据集
            _, ret = acl.mdl.add_dataset_buffer(dataset, data_buffer)
            datas.append({"buffer": buffer, "data": data_buffer, "size": buffer_size})
        return dataset, datas

    def forward(self, inputs):
        # 执行推理任务
        # 遍历所有输入，拷贝到对应的buffer内存中
        input_num = len(inputs)
        for i in range(input_num):
            bytes_data = inputs[i].tobytes()
            bytes_ptr = acl.util.bytes_to_ptr(bytes_data)
            # 将图片数据从Host传输到Device。
            ret = acl.rt.memcpy(self.input_data[i]["buffer"],   # 目标地址 device
                                self.input_data[i]["size"],     # 目标地址大小
                                bytes_ptr,                      # 源地址 host
                                len(bytes_data),                # 源地址大小
                                ACL_MEMCPY_HOST_TO_DEVICE)      # 模式:从host到device
        # 执行模型推理。
        ret = acl.mdl.execute(self.model_id, self.input_dataset, self.output_dataset)
        # 处理模型推理的输出数据，输出top5置信度的类别编号。
        inference_result = []
        for i, item in enumerate(self.output_data):
            buffer_host, ret = acl.rt.malloc_host(self.output_data[i]["size"])
            # 将推理输出数据从Device传输到Host。
            ret = acl.rt.memcpy(buffer_host,                    # 目标地址 host
                                self.output_data[i]["size"],    # 目标地址大小
                                self.output_data[i]["buffer"],  # 源地址 device
                                self.output_data[i]["size"],    # 源地址大小
                                ACL_MEMCPY_DEVICE_TO_HOST)      # 模式：从device到host
            # 从内存地址获取bytes对象
            bytes_out = acl.util.ptr_to_bytes(buffer_host, self.output_data[i]["size"])
            # 按照float32格式将数据转为numpy数组
            data = np.frombuffer(bytes_out, dtype=np.float32)
            inference_result.append(data)
            # 释放内存
            ret = acl.rt.free_host(buffer_host)
        vals = np.array(inference_result).flatten()
        # 对结果进行softmax转换
        vals = np.exp(vals)
        vals = vals / np.sum(vals)
        
        return vals

    def __del__(self):
        # 析构函数 按照初始化资源的相反顺序释放资源。
        # 销毁输入输出数据集
        for dataset in [self.input_data, self.output_data]:
            while dataset:
                item = dataset.pop()
                ret = acl.destroy_data_buffer(item["data"])    # 销毁buffer数据
                ret = acl.rt.free(item["buffer"])              # 释放buffer内存
        ret = acl.mdl.destroy_dataset(self.input_dataset)      # 销毁输入数据集
        ret = acl.mdl.destroy_dataset(self.output_dataset)     # 销毁输出数据集
        # 销毁模型描述
        ret = acl.mdl.destroy_desc(self.model_desc)
        # 卸载模型
        ret = acl.mdl.unload(self.model_id)
        # 释放device
        ret = acl.rt.reset_device(self.device_id)
        # acl去初始化
        ret = acl.finalize()

def transfer_pic(input_path):
    # 图像预处理
    input_path = os.path.abspath(input_path)
    with Image.open(input_path) as image_file:
        # 缩放为224*224
        img = image_file.resize((224, 224))
        # 转换为float32类型ndarray
        img = np.array(img).astype(np.float32)
    # 根据imageNet图片的均值和方差对图片像素进行归一化
    img -= [123.675, 116.28, 103.53]
    img /= [58.395, 57.12, 57.375]
    # RGB通道交换顺序为BGR
    img = img[:, :, ::-1]
    # resnet50为色彩通道在前
    img = img.transpose((2, 0, 1))
    # 返回并添加batch通道
    return np.array([img])

def print_top_5(data):
    top_5 = data.argsort()[::-1][:5]
    print("======== top5 inference results: =============")
    for j in top_5:
        print("[%d]: %f" % (j, data[j]))

if __name__ == "__main__":
    image_paths = ["./data/dog1_1024_683.jpg", "./data/dog2_1024_683.jpg"]
    model_path = './model/resnet50.om'
    resnet50 = Net(model_path)
    
    for path in image_paths:
        # 图像预处理，此处仅供参考，用户按照自己需求进行预处理
        image = transfer_pic(path)
        # 将数据按照每个输入的顺序构造list传入，当前示例的ResNet-50模型只有一个输入
        result = resnet50.forward([image])
        # 输出top_5
        print_top_5(result)

    del resnet50
