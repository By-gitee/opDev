#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2022-2023 Huawei Technologies Co., Ltd
import numpy as np
import os
import torch


def gen_golden_data_simple():
    input_x = np.random.uniform(-10, 10, [3, 43,117]).astype(np.float16)
    diagonal = 10
    golden = torch.tril(torch.Tensor(input_x), diagonal=diagonal).numpy().astype(np.float16)
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    print(golden)
    input_x.tofile("./input/input_x.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple()
