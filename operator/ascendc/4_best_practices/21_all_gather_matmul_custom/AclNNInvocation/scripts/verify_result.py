import os
import sys
import numpy as np

def cal_relativediff_numpy(data_check, data_exepect, diff_thd):
    a = np.abs(np.subtract(data_check, data_exepect))
    b1 = np.maximum(np.abs(data_check), (np.abs(data_exepect)))
    b2 = float((1.0 / (1 << 14)) / diff_thd)
    b = np.add(np.maximum(b1, b2), 10e-10)
    result = np.where(a < diff_thd, a, a / b)
    return result

def data_compare(data_check, data_exepect, diff_thd=0.005, pct_thd=0.005):
    npu_shape = data_check.shape
    expect_shape = data_exepect.shape
    if npu_shape != expect_shape:
        print("============ out_shape is not equal expect!")
        return False
    data_check = data_check.flatten()
    data_exepect = data_exepect.flatten()
    start = 0
    end = data_check.size - 1
    diff = cal_relativediff_numpy(data_check, data_exepect, diff_thd)
    split_count = int(end - start + 1) if end != start else 1
    lt_num = diff[diff < diff_thd].size
    lt_num = lt_num + data_exepect[np.isinf(data_exepect)].size + data_exepect[np.isnan(data_exepect)].size
    lt_pct = float(lt_num) / float(split_count) * 100
    pct_thd = (1 - pct_thd) * 100.0
    return (lt_pct >= pct_thd)

def verify_result(out_dir):
    for i in range(8):
        gather_out = np.fromfile("{}/gather_out_{}.bin".format(out_dir, i), dtype=np.float16)
        out = np.fromfile("{}/out_{}.bin".format(out_dir, i), dtype=np.float16)
        golden_gather_out = np.fromfile("{}/golden_gather_out_{}.bin".format(out_dir, i), dtype=np.float16)
        golden_out = np.fromfile("{}/golden_out_{}.bin".format(out_dir, i), dtype=np.float16)

        if not data_compare(gather_out, golden_gather_out):
            print("============ gather_out_{} precession check failed", i)
            return False
        if not data_compare(out, golden_out):
            print("============ out_{} precession check failed", i)
            return False

    print("test pass")
    return True


if __name__ == '__main__':
    verify_result(sys.argv[1])
