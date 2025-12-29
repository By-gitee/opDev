#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2024 Huawei Technologies Co., Ltd
import numpy as np
import os

rank_dim = 8
rank_m = 512
rank_k = 5120
rank_n = 640

def gen_golden_data_simple():
    if not os.path.exists("input"):
        os.mkdir("input")
    if not os.path.exists("output"):
        os.mkdir("output")

    input_a = []
    input_b = []
    for i in range(rank_dim):
        a = np.random.uniform(-3, 3, [rank_m, rank_k]).astype(np.float16)
        b = np.random.uniform(-3, 3, [rank_k, rank_n]).astype(np.float16)
        a.tofile("./input/input_a_{}.bin".format(i))
        b.tofile("./input/input_b_{}.bin".format(i))
        input_a.append(a)
        input_b.append(b)

    golden_gather_out = np.concatenate(input_a, axis=0)
    for i in range(rank_dim):
        golden_gather_out.tofile("./output/golden_gather_out_{}.bin".format(i))
        out = np.matmul(golden_gather_out.astype(np.float32), input_b[i].astype(np.float32)).astype(np.float16)
        out.tofile("./output/golden_out_{}.bin".format(i))

if __name__ == "__main__":
    gen_golden_data_simple()
