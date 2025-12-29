import os
import sys
import numpy as np

loss = 1e-4
minimum = 10e-10

def verify_result(real_result_y, real_result_indices, golden_y, golden_indices):
    real_result_y = np.fromfile(real_result_y, dtype=np.float32)
    golden_y = np.fromfile(golden_y, dtype=np.float32)
    real_result_indices = np.fromfile(real_result_indices, dtype=np.int32)
    golden_indices = np.fromfile(golden_indices, dtype=np.int32)

    result_y = np.abs(real_result_y - golden_y)
    deno_y = np.maximum(np.abs(real_result_y), np.abs(golden_y))
    result_atol_y = np.less_equal(result_y, loss)
    result_rtol_y = np.less_equal(result_y / np.add(deno_y, minimum), loss)

    result_indices = np.abs(real_result_indices - golden_indices)
    deno_indices = np.maximum(np.abs(real_result_indices), np.abs(golden_indices))
    result_atol_indices = np.less_equal(result_indices, loss)
    result_rtol_indices = np.less_equal(result_indices / np.add(deno_y, minimum), loss)

    if not result_rtol_y.all() and not result_atol_y.all():
        if np.sum(result_rtol_y == False) > real_result_y.size * loss and np.sum(result_atol_y == False) > real_result_y.size * loss:
            print("[ERROR] result_rtol_y error")
            return False

    if not result_rtol_indices.all() and not result_atol_indices.all():
        if np.sum(result_rtol_indices == False) > result_rtol_indices.size * loss and np.sum(result_rtol_indices == False) > result_rtol_indices.size * loss:
            print("[ERROR] result_rtol_indices error")
            return False           
    print("test pass")
    return True

if __name__ == '__main__':
    verify_result(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
