#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2022-2023 Huawei Technologies Co., Ltd
import numpy as np
import os
import torch
import torch.nn as nn

def gen_golden_data_simple():
    input_predict = np.random.uniform(-10, 10, [1024, 1024]).astype(np.float32)
    input_label  = np.random.uniform(-10, 10, [1024, 1024]).astype(np.float32)
    # 初始化损失函数
    mse_loss = nn.MSELoss(reduction='mean')

    # 假设我们有一些预测值和目标值
    predictions = torch.tensor(input_predict)
    targets = torch.tensor(input_label)

    # 计算损失
    loss = mse_loss(predictions, targets )

    golden = loss.numpy()

    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_predict.tofile("./input/input_predict.bin")
    input_label.tofile("./input/input_label.bin")
    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()
