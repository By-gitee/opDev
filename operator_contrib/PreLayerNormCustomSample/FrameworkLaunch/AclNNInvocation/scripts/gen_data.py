#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2022-2023 Huawei Technologies Co., Ltd
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


def gen_golden_data_simple(CaseId):
    if CaseId == "case1":
        x_shape = (4980, 4 , 2048)
        eps = 1e-5
    elif CaseId == "case2":
        x_shape = (512, 4 , 20480)
        eps = 1e-7
    else:
        print("[ERROR] The caseId is error, please input [case1 / case2]")
        exit(1)
    dtype = torch.float32
    device = "cpu"
    x = torch.randn(*x_shape, dtype=dtype, device=device)
    y = torch.randn(*x_shape, dtype=dtype, device=device)
    gamma = nn.Parameter(torch.randn(x_shape[2], dtype=dtype, device=device))
    beta = nn.Parameter(torch.randn(x_shape[2], dtype=dtype, device=device))
    normalized_shape = (x_shape[-1],)
    add = torch.add(x,y)
    res = F.layer_norm(add,normalized_shape, gamma, beta, eps)
    input_x = x.detach().numpy().astype(np.float32)
    input_y = y.detach().numpy().astype(np.float32)
    input_gamma = gamma.detach().numpy().astype(np.float32)
    input_beta = beta.detach().numpy().astype(np.float32)
    golden = res.detach().numpy().astype(np.float32)
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_x.tofile("./input/input_x.bin")
    input_y.tofile("./input/input_y.bin")
    input_gamma.tofile("./input/input_gamma.bin")
    input_beta.tofile("./input/input_beta.bin")
    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple(sys.argv[1])
