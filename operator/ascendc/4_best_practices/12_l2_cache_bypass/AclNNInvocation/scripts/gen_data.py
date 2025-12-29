#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================

import numpy as np


def gen_golden_data_simple():
    row = 5120
    col = 5120
    input_x = np.random.uniform(1, 10, [row, col]).astype(np.float32)
    input_y = np.random.uniform(1, 10, [row, col * 3]).astype(np.float32)
    y_blocks = np.split(input_y, 3, axis=1)
    result_blocks = [input_x + block for block in y_blocks]
    golden = np.hstack(result_blocks)
    input_x.tofile("./input/input_x.bin")
    input_y.tofile("./input/input_y.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple()
