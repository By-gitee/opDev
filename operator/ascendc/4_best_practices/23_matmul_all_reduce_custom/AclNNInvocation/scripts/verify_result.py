import os
import sys
import time
import numpy as np
import torch
import torch_npu
from new_compare import *

def data_compare(cpu_data, gpu_data, npu_data, rank):
    rst_npu = checkResult(npu_data, cpu_data, "{}_dq_npu".format(rank))
    # rst_npu.print_result()
    rst_gpu = checkResult(gpu_data, cpu_data, "{}_dq_gpu".format(rank))
    # rst_gpu.print_result()
    str1, str2 = rst_npu.check_result(rst_gpu)
    if 'error' in str1:
        res = False
    else:
        res = True
    # print('[INFO] device_{} 精度结果为：{}'.format(rank, res))
    # print('[INFO] device_{} 精度计算结束：{} '.format(rank, time.strftime('%H:%M:%S', time.localtime())))
    return res


def verify_result(out_dir, output_dtype='fp16'):
    for rank in range(8):
        cpu_out_path = f"{out_dir}/cpu_out_{rank}.bin"
        gpu_out_path = f"{out_dir}/gpu_out_{rank}.bin"
        npu_out_path = f"{out_dir}/out_{rank}.bin"
        if output_dtype == 'fp16':
            np_dtype = np.float16
        else:
            import tensorflow as tf
            np_dtype = tf.bfloat16.as_numpy_dtype
        cpu_out = torch.tensor(np.fromfile(cpu_out_path, np.float32))
        gpu_out = torch.tensor(np.fromfile(gpu_out_path, np_dtype).astype(np.float32))
        gpu_out = gpu_out.cpu().view(-1)
        npu_out = torch.tensor(np.fromfile(npu_out_path, np_dtype).astype(np.float32))
        npu_out = npu_out.cpu().view(-1)

        if not data_compare(cpu_out, gpu_out, npu_out, rank):
            print("============ out_{} precession check failed", rank)
            return False

    print("test pass")
    return True


if __name__ == '__main__':
    verify_result(sys.argv[1])
