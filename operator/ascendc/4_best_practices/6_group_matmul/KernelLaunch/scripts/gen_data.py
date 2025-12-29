#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================

import numpy as np
import os


def FastGelu(x):
    x2 = np.exp(0.851 * (x - np.abs(x)))
    x3 = 1 + np.exp(-1.702 * np.abs(x))
    return x * x2 / x3


def gen_golden_data():
    groupNum = 8
    m = 1024
    k = 1024
    n = 8192

    groupList = np.array([128] * groupNum, dtype=np.int64)

    x = np.random.randint(-10, 10, [m, k]).astype(np.int8)
    weight = np.random.randint(-10, 10, [groupNum, k, n]).astype(np.int8)
    scale = np.random.normal(0, 0.01, (groupNum, n)).astype(np.float32)
    perTokenScale = np.random.normal(0, 0.01, (m, 1)).astype(np.float32)
    index = np.cumsum(groupList)
    xSplit = np.split(x, index, axis=0)
    perTokenScaleSplit = np.split(perTokenScale, index, axis=0)
    mmOuts = []
    for i in range(groupNum):
        mm = np.matmul(xSplit[i].astype(np.int32), weight[i].astype(np.int32))
        mm = mm.astype(np.float32) * scale[i].astype(np.float32) * perTokenScaleSplit[i]
        mmOuts.append(mm)
    golden = np.concatenate(mmOuts, axis=0).astype(np.float16)
    golden = FastGelu(golden)
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    x.tofile("./input/x.bin")
    weightNz = weight.reshape([groupNum, k // 16, 16, n // 32, 32]).transpose([0, 3, 1, 2, 4])
    weightNz.tofile("./input/weight.bin")
    groupList.tofile("./input/groupList.bin")
    scale.tofile("./input/scale.bin")
    perTokenScale.tofile("./input/perTokenScale.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data()
