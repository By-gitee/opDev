#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2022-2023 Huawei Technologies Co., Ltd
import numpy as np
import os
import torch

def run_three_nn(b, n, m, xyz1, xyz2):
    dist = np.zeros_like(xyz1, dtype=np.float32)
    idx = np.zeros_like(xyz1, dtype=np.int32)

    for ii in range(b):
        for jj in range(n):
            x1 = xyz1[ii, jj, 0]
            y1 = xyz1[ii, jj, 1]
            z1 = xyz1[ii, jj, 2]
            best1 = 1e40
            best2 = 1e40
            best3 = 1e40
            besti1 = 0
            besti2 = 0
            besti3 = 0
            for kk in range(m):
                x2 = xyz2[ii, kk, 0]
                y2 = xyz2[ii, kk, 1]
                z2 = xyz2[ii, kk, 2]
                d = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1)
                if (d < best1):
                    best3 = best2
                    besti3 = besti2
                    best2 = best1
                    besti2 = besti1
                    best1 = d
                    besti1 = kk
                elif (d < best2):
                    best3 = best2
                    besti3 = besti2
                    best2 = d
                    besti2 = kk
                elif (d < best3):
                    best3 = d
                    besti3 = kk
            dist[ii, jj, 0] = best1
            idx[ii, jj, 0] = besti1
            dist[ii, jj, 1] = best2
            idx[ii, jj, 1] = besti2
            dist[ii, jj, 2] = best3
            idx[ii, jj, 2] = besti3

    return dist, idx

def gen_golden_data_simple():
    b = 5
    n = 160
    m = 256
    xyz1_shape = [b,n,3]
    xyz2_shape = [b,m,3]
    input_xyz1 = np.random.uniform(-10, 10, xyz1_shape).astype(np.float32)
    input_xyz2 = np.random.uniform(-10, 10, xyz2_shape).astype(np.float32)
    dist, idx = run_three_nn(b, n, m, input_xyz1, input_xyz2)
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    print("dist = ",dist)
    print("idx = ",idx)
    input_xyz1.tofile("./input/input_xyz1.bin")
    input_xyz2.tofile("./input/input_xyz2.bin")
    dist.tofile("./output/golden_dist.bin")
    idx.tofile("./output/golden_idx.bin")


if __name__ == "__main__":
    gen_golden_data_simple()
