'''
opencv 相机图像采集

'''

import sys
import os
# os.environ['DISPLAY']='192.168.31.167:0.0'
import numpy as np
import cv2
import time


if __name__ == '__main__':

    save_dir = 'imgs/'  # 图片保存路径
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    id = 0
    print('Enter键: 保存图片\nEsc键：退出')
    #初始化相机
    print("查找相机id...")
    for i in range(15):
        try:
            print(i)
            cap = cv2.VideoCapture(0)  # 杰瑞GZ60
            flag, frame = cap.read()
            if flag:
                print(f"camera id {i} is valid， using {i} ")
                break
        except:
            print(f"camera id {i} is invalid")
            continue

    while True:
        flag, frame = cap.read()  # 获取相机图像
        image = frame
        # 定义相机的内参（焦距和光心坐标）和畸变系数
        camera_matrix = np.array(
            [[1.19829297e+03, 0.00000000e+00, 1.00315842e+03],
             [0.00000000e+00, 1.20912180e+03, 5.51069623e+02],
             [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
 )
        distortion_coeffs = np.array([-0.50932019,  0.3163081,  -0.00543584, -0.01006447, -0.10689112])

        image = cv2.undistort(image, camera_matrix, distortion_coeffs)
        image = cv2.resize(image, (1920//2,1080//2))
        cv2.imshow('image', image)
        k = cv2.waitKey(1)
        if k == 27:  # esc
            cv2.destroyAllWindows()
            cap.release()
            sys.exit()
        if k == 13:  # enter
            save_path = f'{save_dir}{time.time()}_{id}.jpg'
            cv2.imwrite(save_path, image)
            print(f'saved to {save_path}')
            id += 1
