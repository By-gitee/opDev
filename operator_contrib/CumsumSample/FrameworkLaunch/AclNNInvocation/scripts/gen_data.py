#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2022-2023 Huawei Technologies Co., Ltd
import numpy as np
import os
import tensorflow as tf


def gen_golden_data_simple():
    input_x = np.random.uniform(-10, 10, [3, 3,3,32]).astype(np.float32)
    dim_num = len(input_x.shape)
    input_axis = np.random.uniform(2, 2, [1]).astype(np.int32)
    exclusive = False
    reverse = False
    golden = tf.math.cumsum(input_x, input_axis[0], exclusive=exclusive, reverse=reverse).numpy()
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_x.tofile("./input/input_x.bin")
    input_axis.tofile("./input/input_axis.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple()
