import torch
import numpy as np
import os


def gen_golden_data_simple():
    start = np.random.uniform(-10, 10, [32]).astype(np.float16)
    end = np.random.uniform(-10, 10, [32]).astype(np.float16)
    weight = np.random.uniform(0, 1, [32]).astype(np.float16)
    res = torch.lerp(torch.Tensor(start), torch.Tensor(end), torch.Tensor(weight))

    golden = res.numpy().astype(np.float16)
    print(golden)
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    start.tofile("./input/input_start.bin")
    end.tofile("./input/input_end.bin")
    weight.tofile("./input/input_weight.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple()
