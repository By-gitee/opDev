# coding=utf-8
import sys

sys.path.append('/usr/local/miniconda3/lib/python3.9/site-packages/')
import cv2  # 图片处理三方库，用于对图片进行前后处理 # opencv-python-headless-4.7.0.72
import numpy as np  # 用于对多维数组进行计算

from objDet_yolov5 import *
from det_utils import draw_bbox
from udp_server import UdpServer

# 相机及目标物参数
CAM_WID, CAM_HGT = 640, 480  # 深度图img的图像尺寸
CAM_FX, CAM_FY = 4.7253557850154391e+02, 4.7159764707015580e+02  # 相机的fx/fy参数
CAM_CX, CAM_CY = 3.2371276037928334e+02, 2.3384312872362858e+02  # 相机的cx/cy参数
REAL_W = 0.025  # 目标真实宽度
REAL_H = 0.061  # 目标真实高度
PIXEL_CX = CAM_WID / 2
PIXEL_CY = CAM_HGT / 2  # 图像中心点

MID_THRESH = 20  # 图像中心偏离阈值
DEEP_THRESH = 0.26 # 摄像头距离目标物体距离阈值
CAM_END_BAIS = 10 # 相机安装在末端时因安装产生的偏移量

# 内参（焦距和光心坐标）和畸变系数，位于vision_calibrate/config/intrinsic_calibration.yaml
camera_matrix = np.array(
    [[CAM_FX, 0.00000000e+00, CAM_CX],
     [0.00000000e+00, CAM_FY, CAM_CY],
     [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]
) # 内参3行3列
distortion_coeffs = np.array(
    [-4.4626334451673577e-01, 2.4743317318853048e-01, -8.1511029675525293e-04, -6.3153671239013617e-04,
     -7.7325005728512336e-02]) # 1行5列


# 推理相关
trained_model_path = "/root/VC/yolo_acl_sample/yolov5s_cylinder_cls3_20230822_640640_f32.om"
context = init_acl(DEVICE_ID)  # 初始化acl相关资源
det_model = YoloV5(model_path=trained_model_path)  # 初始化模型

# 车辆控制及机械臂坐标相关参数
ROT_SPEED = 0.04  # 旋转速度
LINE_SPEED = 0.05 # 直线速度
Coordinate_car_x = -0.04   # 初始0.04
Coordinate_car_y = 0
Coordinate_car_z = 0.2  # 相机到机械臂坐标偏移量


def get_xyz(obj):
    # 通过实际物体宽高估计尺度时，当中心点靠近图像中心时，会导致归于中心的坐标差值较小，与估计的尺度之间拉不开足够的数量级，导致误差放大，估计尺寸结果错误
    # 因此要选择目标上远离图像中心的点进行尺寸估计
    # 目标像素宽高
    x1, y1, x2, y2 = obj[0], obj[1], obj[2], obj[3]

    target_pixel_w = x2 - x1
    target_pixel_h = y2 - y1

    # 通过实际长与宽分别计算比例关系
    scale_w = REAL_W / target_pixel_w  # 长度比例
    scale_h = REAL_H / target_pixel_h  # 长度比例
    # print(scale_w, scale_h)
    # 取比例均值降低误差
    scale = (scale_w + scale_h) / 2
    # print("center: {}  {}".format(target_pixel_center_x, target_pixel_center_y))
    # 目标像素中心（中心化后的）
    target_pixel_center_x = (x1 + target_pixel_w * 0.5) - PIXEL_CX
    target_pixel_center_y = (y1 + target_pixel_h * 0.5) - PIXEL_CY

    cam_x1, cam_x2, cam_y1, cam_y2 = None, None, None, None,
    z_x1, z_x2, z_y1, z_y2 = None, None, None, None,
    if abs(x1 - PIXEL_CX) > 35:  # 拉开足够数量级
        cam_x1 = (x1 - PIXEL_CX) * scale
        z_x1 = (cam_x1 * CAM_FX) / (x1 - CAM_CX)

    if abs(x2 - PIXEL_CX) > 35:  # 拉开足够数量级
        cam_x2 = (x2 - PIXEL_CX) * scale
        z_x2 = (cam_x2 * CAM_FX) / (x2 - CAM_CX)

    if abs(y1 - PIXEL_CY) > 35:  # 拉开足够数量级
        cam_y1 = (y1 - PIXEL_CY) * scale
        z_y1 = (cam_y1 * CAM_FY) / (y1 - CAM_CY)

    if abs(y2 - PIXEL_CY) > 35:  # 拉开足够数量级
        cam_y2 = (y2 - PIXEL_CY) * scale
        z_y2 = (cam_y2 * CAM_FY) / (y2 - CAM_CY)

    depth_arr = np.array([z_x1, z_x2, z_y1, z_y2])
    num_depth = np.sum(depth_arr != None)
    depth_sum = 0
    for i in depth_arr:
        if i != None:
            depth_sum += i
    depth = depth_sum / num_depth
    print(depth)

    # 通过相机内参计算深度
    cam_x_new = depth * ((target_pixel_center_x + PIXEL_CX) - CAM_CX) / CAM_FX
    cam_y_new = depth * ((target_pixel_center_y + PIXEL_CY) - CAM_CY) / CAM_FY
    print("xyz{} {} {}".format(cam_x_new, cam_y_new, depth))
    return [cam_x_new, cam_y_new, depth], target_pixel_center_x, depth


def contrl_car(target_pixel_center_x):
    if target_pixel_center_x > MID_THRESH - CAM_END_BAIS:
        return -ROT_SPEED
    elif target_pixel_center_x < -MID_THRESH- CAM_END_BAIS:
        return ROT_SPEED
    else:
        return 0


def contrl_car_line(target_pixel_center_z):
    if target_pixel_center_z > DEEP_THRESH:
        return LINE_SPEED
    else:
        return 0


if __name__ == '__main__':
    udp = UdpServer()
    # 初始化相机
    print("get cam id...")
    for i in range(15):
        try:
            print(i)
            cap = cv2.VideoCapture(i)  # 杰瑞GZ60
            flag, frame = cap.read()
            if flag:
                print(f"camera id {i} is valid    using {i} ")
                break
        except:
            print(f"camera id {i} is invalid")
            continue

    # 标志位
    client_addr = None   # 客户端地址
    car_contrl = 0       # 车辆旋转到位的次数
    car_contrl_line = 0  # 车辆直线修正到位的次数
    line_over = False    # 是否完成直线运动距离修正
    rotate_over = False  # 是否完成旋转
    begin = False        # 是否开始检测
    while True:
        flag, img = cap.read()  # 获取相机图像
        img = cv2.undistort(img, camera_matrix, distortion_coeffs) #去畸变
        start, addr = udp.run_once_rec()
        print("start: ", start)
        if start == "start":
            client_addr = addr
            begin = True
        # 推理
        if begin:
            pred_all = det_model.infer(img)  # 前处理、推理、后处理,
            if pred_all == []:
                print('No Pred')
                continue
            
            target_xyzs = []
            pixel_center_xs = []
            depth_center_zs = []
            for obj in pred_all:
                target_xyz, pixel_center_x, depth = get_xyz(obj)
                target_xyzs.append(target_xyz)
                pixel_center_xs.append(pixel_center_x)
                depth_center_zs.append(depth)
                
                
            # 调整车体定位距离
            if line_over is not True:
                print("depth_center_zs: ", depth_center_zs)
                if depth_center_zs == []:
                    print("no target Line")
                    continue
                contrl_data_line = contrl_car_line(depth_center_zs[0])
                udp.run_once_send("L {}".format(contrl_data_line), client_addr)  # 开始发送消息
                print("L {}".format(contrl_data_line))
                if contrl_data_line == 0:
                    car_contrl_line += 1
                if car_contrl_line >= 8:    # 经历20个循环确认完成直线距离修正
                    line_over = True     
                    car_contrl_line = 0  # 重置车辆直线修正到位次数
            else:         
                # 完成车体离目标物距离修正后   
                # 调整车体方向
                if rotate_over is not True:  # 未完成车体方向调整
                    if pixel_center_xs == []:
                        print("no target Rotate")
                        continue
                    contrl_data = contrl_car(pixel_center_xs[0])
                    udp.run_once_send("R {}".format(contrl_data), client_addr)  # 开始发送消息
                    print("R {}".format(contrl_data))
                    if contrl_data == 0:  # 已完成
                        car_contrl += 1
                    if car_contrl >= 30:   # 经历30个循环确认完成
                        rotate_over = True
                        car_contrl = 0    # 重置次数
                                 
            if rotate_over:
                # 发送坐标，调试时需加偏置，并转换xyz轴方向
                for i in range(3):
                    # Z:Coordinate_car_z - target_xyzs[0][1]
                    udp.run_once_send("P {} {} {}".format(Coordinate_car_x - target_xyzs[0][2], target_xyzs[0][0], 0.18), client_addr)
                    print("robotxyz:{},{},{},修正：{}".format(Coordinate_car_x - target_xyzs[0][2], target_xyzs[0][0], Coordinate_car_z - target_xyzs[0][1], 0.18))
                    
                    
                begin = False #重置检测状态，不在进行检测仅获取视频流
                rotate_over = False # 重置旋转状态
                line_over = False # 重置直线距离修正状态

            labels_dict = {0: "0_yellow", 1: "0_blue", 2: "0_green"}
            img = draw_bbox(pred_all, img, (0, 255, 0), 1, labels_dict)  # 画出检测框、类别、概率，得到最终推理图片
            img = cv2.resize(img, (1920 // 2, 1080 // 2))
        cv2.imshow("image", img)
        k1 = cv2.waitKey(1)
        if k1 == 27:  # esc
            det_model.release()  # 释放 acl 模型相关资源, 包括输入数据、输出数据、模型等
            deinit_acl(context, 0)  # acl 去初始化
            cv2.destroyAllWindows()
            cap.release()
            sys.exit()