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
    # dtype = np.float32
    # dtype = bfloat16
    dtype = np.int8

    ## Broadcast场景 axis = 0时, 核间均分, 单核计算量对齐
    # input_shape_x = [8, 1024]
    # input_shape_y = [1, 1024]

    ## Broadcast场景 axis = 0时, 核间均分, 单核计算量非对齐
    # input_shape_x = [8, 1022]
    # input_shape_y = [1, 1022]

    ## Broadcast场景 axis = 0时, 核间不均分, 单核计算量对齐
    # input_shape_x = [17, 1024]
    # input_shape_y = [1, 1024]

    ## Broadcast场景 axis = 0时, 核间不均分, 单核计算量非对齐
    input_shape_x = [17, 1022]
    input_shape_y = [1, 1022]

    ## Broadcast场景 axis = 1时, 核间均分, 单核计算量对齐
    # input_shape_x = [16, 1]
    # input_shape_y = [16, 256]

    ## Broadcast场景 axis = 1时, 核间均分, 单核计算量非对齐
    # input_shape_x = [16, 1]
    # input_shape_y = [16, 255]

    ## Broadcast场景 axis = 1时, 核间不均分, 单核计算量对齐
    # input_shape_x = [20, 1]
    # input_shape_y = [20, 256]

    ## Broadcast场景 axis = 1时, 核间不均分, 单核计算量非对齐
    # input_shape_x = [20, 1]
    # input_shape_y = [20, 255]

    input_x = np.random.uniform(-50, 50, input_shape_x).astype(dtype)
    input_y = np.random.uniform(-50, 50, input_shape_y).astype(dtype)
    golden = (input_x + input_y).astype(dtype)

    if np.size(input_x) > np.size(input_y):
        if input_shape_y[0] == 1:
            axis = 0
            coef = np.size(input_y)
        elif input_shape_y[1] == 1:
            axis = 1
            coef = np.size(input_x) / np.size(input_y)
    else:
        if input_shape_x[0] == 1:
            axis = 0
            coef = np.size(input_x)
        elif input_shape_x[1] == 1:
            axis = 1
            coef = np.size(input_y) / np.size(input_x)
    tiling = np.array([input_shape_x[0] * input_shape_x[1],
                       input_shape_y[0] * input_shape_y[1],
                       coef,
                       axis,
                       dtype_emu[dtype]],
                       dtype=np.uint32)

    tiling.tofile("./input/input_tiling.bin")
    input_x.tofile("./input/input_x.bin")
    input_y.tofile("./input/input_y.bin")
    golden.tofile("./output/golden.bin")

if __name__ == "__main__":
    gen_golden_data_simple()
