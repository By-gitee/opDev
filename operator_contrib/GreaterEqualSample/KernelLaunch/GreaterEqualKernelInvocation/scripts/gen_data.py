#!/usr/bin/python3
# -*- coding:utf-8 -*-
import torch
import numpy as np

def is_integer_dtype(dtype):
    return dtype in [torch.int8, torch.int16, torch.int32, torch.int64]

def is_float_dtype(dtype):
    return dtype in [torch.float16, torch.float32, torch.float64]

def gen_golden_data_simple():
    dtype = torch.float16
    flen = 1024
    
    if(is_float_dtype(dtype)):
        x1 = torch.randn(flen).to(dtype)
        x2 = torch.randn(flen).to(dtype)
    else:
        x1 = torch.randint(-100, 100, (flen,)).to(dtype)
        x2 = torch.randint(-100, 100, (flen,)).to(dtype)
    golden = torch.greater_equal(x1, x2)
    x1.to(dtype).detach().cpu().numpy().tofile("./input/input_x1.bin")
    x2.to(dtype).detach().cpu().numpy().tofile("./input/input_x2.bin")
    golden.detach().cpu().numpy().astype(np.int8).tofile("./output/golden.bin")
    
    
if __name__ == "__main__":
    gen_golden_data_simple()


