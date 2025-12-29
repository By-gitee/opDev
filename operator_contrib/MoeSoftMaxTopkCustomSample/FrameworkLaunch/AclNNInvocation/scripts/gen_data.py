#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2022-2023 Huawei Technologies Co., Ltd
import numpy as np
import torch
import sys
import os

def gen_golden_data_simple(case_id):
    if case_id == "case1":
        x_shape = (1024,16)
        k = 4
    elif case_id == "case2":
        x_shape = (2048,32)
        k = 6
    dtype = torch.float32
    device = "cpu"
    x = torch.randn(*x_shape, dtype=dtype, device=device)
    y, indices = torch.topk(torch.softmax(x,dim=1), k)

    input_x = x.detach().numpy().astype(np.float32)
    output_y = y.detach().numpy().astype(np.float32)
    output_indices = indices.detach().numpy().astype(np.int32)
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_x.tofile("./input/input_x.bin")
    output_y.tofile("./output/golden_y.bin")
    output_indices.tofile("./output/golden_indices.bin")

if __name__ == "__main__":
    gen_golden_data_simple(sys.argv[1])
