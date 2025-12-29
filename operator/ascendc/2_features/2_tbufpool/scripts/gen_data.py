#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2024-2025. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
import os
import numpy as np

def gen_golden_data_simple():
    dtype = np.float32

    input_shape = [8, 256]
    input_x = np.random.randint(0, np.nextafter(1000, np.inf), input_shape).astype(dtype)
    input_y = np.random.randint(0, np.nextafter(1000, np.inf), input_shape).astype(dtype)
    rows = input_shape[0]
    mid = rows // 2
    top_half = input_x[:mid] + input_y[:mid]
    bottom_half = input_x[mid:] - input_y[mid:]
    golden = np.vstack((top_half, bottom_half))

    os.system("mkdir -p ./input")
    input_x.tofile("./input/input_x.bin")
    input_y.tofile("./input/input_y.bin")
    os.system("mkdir -p ./output")
    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()