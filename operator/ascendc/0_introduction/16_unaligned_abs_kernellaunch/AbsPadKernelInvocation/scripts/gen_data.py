import numpy as np

def gen_golden_data_simple():
    COLS = 7
    ROWS = 16
    CORE_NUM = 8
    TILE_NUM = 16
    input_x = np.arange(COLS * ROWS * CORE_NUM * TILE_NUM).astype(np.float16)*(-1) - 1
    golden = np.absolute(input_x).astype(np.float16)
    input_x.tofile("./input/input_x.bin")
    golden.tofile("./output/golden.bin")

if __name__ == '__main__':
    gen_golden_data_simple()