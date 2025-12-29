import numpy as np
import os

def gen_golden_data_simple():
    zero_tensor = np.zeros([16-4]).astype(np.float16)
    sync_tensor = np.zeros([8*4]).astype(np.int32)
    input_ = np.arange(4*4*4).astype(np.float16)*(-1)
    input_x = np.concatenate((input_, zero_tensor),axis=None)
    zero_tensor = np.zeros([16*4]).astype(np.float16)
    for i in range(0,61):
        if i%4==0:
            zero_tensor[i] = i * (-1) -3
    input_x = zero_tensor
    zero_gm = np.zeros([16 - 4]).astype(np.float16)

    golden = np.concatenate((input_x, zero_gm),axis=None)
    input_x.tofile("./input/input_x.bin")
    golden.tofile("./output/golden.bin")
    sync_tensor.tofile("./input/sync.bin")

if __name__ == '__main__':
    gen_golden_data_simple()