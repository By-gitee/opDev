import numpy as np

def gen_golden_data_simple():
    input_x = np.arange(14*16*8*8*2).astype(np.float16)
    golden = np.absolute(input_x).astype(np.float16)

    input_x.tofile("./input/input_x.bin")
    golden.tofile("./output/golden.bin")

if __name__ == '__main__':
    gen_golden_data_simple()