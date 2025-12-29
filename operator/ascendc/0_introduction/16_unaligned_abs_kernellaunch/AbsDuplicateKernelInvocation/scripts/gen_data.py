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
    input_x_ = np.arange(11*16*4).astype(np.float16)*(-1)
    zero_tensor = np.zeros([16-11]).astype(np.float16)
    input_x = np.concatenate((input_x_, zero_tensor),axis=None)
    golden = np.absolute(input_x).astype(np.float16)
    sync_tensor = np.zeros([8*4]).astype(np.int32)
    input_x.tofile("./input/input_x.bin")
    golden.tofile("./output/golden.bin")
    sync_tensor.tofile("./input/sync.bin")

if __name__ == '__main__':
    gen_golden_data_simple()