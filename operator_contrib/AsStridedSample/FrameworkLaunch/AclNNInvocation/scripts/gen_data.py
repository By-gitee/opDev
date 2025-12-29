#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2022-2023 Huawei Technologies Co., Ltd
import numpy as np
import os
import torch


def gen_golden_data_simple():
    input_x = np.random.uniform(-10, 10, [50,400]).astype(np.float32)
    input_size = np.array([64,64]).astype(np.int32)
    input_stride = np.array([3,7]).astype(np.int32)
    input_storage_offset = np.array([5]).astype(np.int32)
    res = torch.as_strided(torch.Tensor(input_x),
                           tuple(input_size),
                           tuple(input_stride),
                           input_storage_offset[0])

    golden = res.numpy().astype(np.float32)
    print(golden)
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_x.tofile("./input/input_x.bin")
    input_size.tofile("./input/input_size.bin")
    input_stride.tofile("./input/input_stride.bin")
    input_storage_offset.tofile("./input/input_storage_offset.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple()
