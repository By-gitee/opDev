import os
import sys
import numpy as np

loss = 1e-4


def verify_result(real_result, golden):
    result = np.fromfile(real_result, dtype=np.float32)
    golden = np.fromfile(golden, dtype=np.float32)
    for i in range(len(result)):
        diff = abs(result[i] - golden[i])
        if (diff > loss) and (diff / golden[i] > loss):
            error_message = "output[{}] is {}, expect {}".format(i, result[i], golden[i])
            print(error_message)
            return False
    
    print("test pass")
    return True


if __name__ == '__main__':
    verify_result(sys.argv[1], sys.argv[2])
