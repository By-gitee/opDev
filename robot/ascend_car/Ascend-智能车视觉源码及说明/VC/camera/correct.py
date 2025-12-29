'''
去畸变 （处理1个文件夹内图片）

'''

import sys
import os
# os.environ['DISPLAY']='192.168.31.167:0.0'
import numpy as np
import cv2
import time
import glob

if __name__ == '__main__':

    glob_str = "/root/VC/20230822/*.jpg"
    save_dir = '/root/VC/20230822_undist/'  # 图片保存路径
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # 定义相机的内参（焦距和光心坐标）和畸变系数
    camera_matrix = np.array(
        [[1.19829297e+03, 0.00000000e+00, 1.00315842e+03],
         [0.00000000e+00, 1.20912180e+03, 5.51069623e+02],
         [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
    )
    distortion_coeffs = np.array([-0.50932019, 0.3163081, -0.00543584, -0.01006447, -0.10689112])

    ls = glob.glob(glob_str)
    for i in ls:
        image = cv2.imread(i)
        image = cv2.undistort(image, camera_matrix, distortion_coeffs)
        file_name = i.split("/")[-1]
        save_path = f"{save_dir}{file_name}"
        cv2.imwrite(save_path, image)
        print(f'saved to {save_path}')

