#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2022-2023 Huawei Technologies Co., Ltd
import torch
import numpy as np
import os


def gen_golden_data_simple():
    input_x = np.random.uniform(-10, 10, [2000]).astype(np.int32)
    bins = 100
    min = 0
    max = 0
    res = torch.histc(torch.Tensor(input_x), bins=bins, min=min, max=max)
    golden = res.numpy().astype(np.int32)
    print(golden)
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_x.tofile("./input/input_x.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple()
