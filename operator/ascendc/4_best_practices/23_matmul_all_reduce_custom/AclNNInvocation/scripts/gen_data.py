#!/usr/bin/python3
# -*- coding:utf-8 -*-
# Copyright 2022-2024 Huawei Technologies Co., Ltd
import numpy as np
import os
import random
import torch
import torch_npu
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.distributed_c10d import ReduceOp

rank_dim = 8
rank_m = 16384
rank_k = 640
rank_n = 5120

# fp16
def gen_cpu_data(rank, port):
    input_x1 = torch.tensor(np.fromfile("./input/input_x1_{}.bin".format(rank), np.float16).reshape([rank_m, rank_k]))
    input_x2 = torch.tensor(np.fromfile("./input/input_x2_{}.bin".format(rank), np.float16).reshape([rank_k, rank_n]))
    # torch_npu.npu.set_device(rank)
    dist.init_process_group(backend='gloo', rank=rank, world_size=rank_dim, init_method=f'tcp://127.0.0.1:{port}')
    print('[INFO] device_{} 构造cpu_out数据'.format(rank))
    cpu_input = input_x1.to(torch.float32)
    cpu_weight = input_x2.to(torch.float32)
    cpu_mm_out = torch.matmul(cpu_input, cpu_weight)
    dist.all_reduce(cpu_mm_out, op=dist.ReduceOp.SUM)
    # cpu_scatter_out = cpu_mm_out.narrow(0, rank * rank_m // rank_dim, rank_m // rank_dim)
    np.array(cpu_mm_out.cpu()).tofile('./output/cpu_out_{}.bin'.format(rank))


def gen_cpu():
    from torch.multiprocessing import Process
    p_list = []
    port = 29500 + random.randint(0, 10000)
    mp.set_start_method("forkserver", force=True)
    for rank in range(rank_dim):
        p = Process(target=gen_cpu_data, args=(rank, port))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()


def gen_gpu_data(rank, port=50001):
    input_x1 = torch.tensor(np.fromfile("./input/input_x1_{}.bin".format(rank), np.float16).reshape([rank_m, rank_k])).npu()
    input_x2 = torch.tensor(np.fromfile("./input/input_x2_{}.bin".format(rank), np.float16).reshape([rank_k, rank_n])).npu()
    torch_npu.npu.set_device(rank)
    dist.init_process_group(backend="hccl", rank=rank, world_size=rank_dim, init_method=f'tcp://127.0.0.1:{port}')
    print('[INFO] device_{} 构造gpu_out数据'.format(rank))
    gpu_out = torch.zeros([rank_m, rank_n], dtype=torch.float16).npu()
    gpu_mm_out = torch.matmul(input_x1, input_x2)
    dist.all_reduce(gpu_mm_out.npu(), op=ReduceOp.SUM)
    np.array(gpu_out.cpu()).tofile('./output/gpu_out_{}.bin'.format(rank))


def gen_gpu():
    from torch.multiprocessing import Process
    p_list = []
    mp.set_start_method("forkserver", force=True)
    port = 50001
    for rank in range(rank_dim):
        p = Process(target=gen_gpu_data, args=(rank, port))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()


if __name__ == "__main__":
    if not os.path.exists("input"):
        os.mkdir("input")
    if not os.path.exists("output"):
        os.mkdir("output")

    # get x1 x2
    for rank in range(rank_dim):
        np.random.seed(rank)
        x1 = np.random.uniform(-3, 3, [rank_m, rank_k]).astype(np.float16)
        x2 = np.random.uniform(-3, 3, [rank_k, rank_n]).astype(np.float16)
        x1.tofile("./input/input_x1_{}.bin".format(rank))
        x2.tofile("./input/input_x2_{}.bin".format(rank))

    gen_cpu()
    gen_gpu()