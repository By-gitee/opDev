#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2024-2025. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================

import numpy as np


def gen_golden_data():

    input_a = np.random.randint(1, 10, size=(1024, 256,)).astype(np.float16)
    input_b = np.random.randint(1, 10, size=(256, 640,)).astype(np.float16)
    input_bias = np.random.randint(1, 10, size=(640,)).astype(np.float32)
    golden = (np.matmul(input_a.astype(np.float32), input_b.astype(np.float32)) + input_bias).astype(np.float32)

    alpha = 0.001
    golden = (np.matmul(input_a.astype(np.float32), input_b.astype(np.float32)) + input_bias).astype(np.float32)
    golden = np.where(golden >= 0, golden * 0, golden * alpha * 0)
    input_a.tofile("input_0.bin")
    input_b.tofile("input_1.bin")
    input_bias.tofile("input_2.bin")
    golden.tofile("golden.bin")


if __name__ == "__main__":
    gen_golden_data()
