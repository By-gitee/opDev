#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2023-2025. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================

import numpy as np
import tensorflow as tf
bfloat16 = tf.bfloat16.as_numpy_dtype
dtype_emu = {bfloat16: 0, np.float16: 1, np.float32: 2, np.int8: 3, np.int16: 4, np.int32: 5}

def gen_golden_data_simple():
    dtype = np.int8
    # dtype = bfloat16

    ## 核间均分，单核计算量对齐:
    # input_shape = [32, 1024]

    ## 核间均分，单核计算量非对齐:
    # input_shape = [8, 1023]

    ## 核间不均分，单核计算量对齐:
    # input_shape = [32, 1023]

    ## 核间不均分，单核计算量非对齐:
    input_shape = [17, 1023]

    input_x = np.random.uniform(-50, 50, input_shape).astype(dtype)
    input_y = np.random.uniform(-50, 50, input_shape).astype(dtype)
    golden = (input_x + input_y).astype(dtype)
    tiling = np.array([input_shape[0] * input_shape[1], dtype_emu[dtype]], dtype=np.uint32)

    tiling.tofile("./input/input_tiling.bin")
    input_x.tofile("./input/input_x.bin")
    input_y.tofile("./input/input_y.bin")
    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()
