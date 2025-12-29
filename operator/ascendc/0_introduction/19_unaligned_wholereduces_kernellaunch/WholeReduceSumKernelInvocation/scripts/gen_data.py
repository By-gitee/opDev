#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================

import numpy as np


def gen_golden_data_simple():
    datatype = np.float16
    shape = [13, 123]
    input_x = np.random.uniform(1, 100, shape).astype(datatype)
    golden = np.sum(input_x, axis=1).astype(datatype)
    tiling = [shape[0] * shape[1]] + shape
    tiling = np.array(tiling, dtype=np.uint32)

    input_x.tofile("./input/input_x.bin")
    tiling.tofile("./input/tiling.bin")
    golden.tofile("./output/golden.bin")
    

if __name__ == "__main__":
    gen_golden_data_simple()
