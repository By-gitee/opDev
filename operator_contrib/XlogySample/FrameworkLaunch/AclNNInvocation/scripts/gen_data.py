import tensorflow as tf
import numpy as np
import os
import tensorflow as tf


def gen_golden_data_simple():
    test_type = np.float32
    shape = [128, 64]
    input_x1 = np.random.uniform(-10, 10, shape).astype(test_type)
    input_x2 = np.random.uniform(0, 10, shape).astype(test_type)
    res = tf.math.xlogy(input_x1, input_x2)
    golden = np.array(res).astype(test_type).reshape(shape)
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_x1.tofile("./input/input_x1.bin")
    input_x2.tofile("./input/input_x2.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple()
