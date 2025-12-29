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

def gen_golden_data_simple():
    input_shape_x = [1, 2048]
    input_shape_y = [1, 2048]
    dtype = np.float16

    input_x = np.random.uniform(-50, 50, input_shape_x).astype(dtype)
    input_y = np.random.uniform(-50, 50, input_shape_y).astype(dtype)
    golden = (input_x + input_y).astype(dtype)

    input_x.tofile("./input/input_x.bin")
    input_y.tofile("./input/input_y.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple()
