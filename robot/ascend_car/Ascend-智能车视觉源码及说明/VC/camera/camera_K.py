'''
杰瑞GZ60单目相机内参标定
'''
import sys
import cv2

# import pyrealsense2 as rs
import numpy as np

class Camera(object): # JR_GZ60
    def __init__(self):
        #
        for i in range(15):
            self.cap = cv2.VideoCapture(i)  # 杰瑞GZ60
            flag, frame = self.cap.read()
            if (flag):
                print(f"used camera id {i}")
                break
        # self.cap = cv2.VideoCapture(0)  # 杰瑞GZ60
        fps, total = self.cap.get(cv2.CAP_PROP_FPS), int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 帧率，总帧数
        self.width, self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 帧宽，帧高
        print(f'fps:{fps} total{total} w{self.width} h{self.height}')

    def get_frame(self):
        if self.cap.isOpened():
            flag, frame = self.cap.read()  # 获取相机图像
            return frame

    def release(self):
        self.cap.release()


class MarkCamera:
    def __init__(self):
        self.w = 4
        self.h = 5 # 行数
        self.b = 0.0106 # 小板=0.0106  15mm/1.414  大板=0.025
        self.minArea = 100
        self.maxArea = 5000000
        self.mark_points = []
        self.cam_corners = []

        self.camera_matrix, self.camera_distort = np.zeros((3,3), np.uint8), np.zeros((1, 5), np.uint8)
        print('Start Internal reference calibration...')


    def get_mark_points(self):
        """
        获取标定版为世界坐标系的圆心世界坐标
        :return: 圆心世界坐标
        """
        object_points = []
        for i in range(self.h):
            for j in range(self.w):
                object_points.append([(i % 2 + j * 2) * self.b, i * self.b, 0])
        return np.float32(object_points)

    def find_circles_grid(self, color_image):
        """
        查找圆心坐标
        :param color_image:
        :return: 是否找到True or False, 图像上的圆心坐标
        """
        params = cv2.SimpleBlobDetector_Params()
        params.maxArea = self.maxArea
        params.minArea = self.minArea
        blob_detector = cv2.SimpleBlobDetector_create(params)
        ret_find_grid, corners = cv2.findCirclesGrid(image=color_image, patternSize=(self.w, self.h),
                                           flags=cv2.CALIB_CB_ASYMMETRIC_GRID,
                                           blobDetector=blob_detector)
        self.cam_corners = corners
        return ret_find_grid


    def show(self, color_image, ret_find_grid):
        cv2.drawChessboardCorners(color_image, (self.w, self.h), self.cam_corners, ret_find_grid)
        cv2.imshow('camera image', 255 - color_image)
        cv2.waitKey(1000)


class CameraIntrinsicCalibrate:
    '''
    相机内参标定
    '''
    def __init__(self):
        self.cam = Camera() # 相机
        self.mark2cam = MarkCamera() # 标定板
        self.cam_plate_coners = [] # 像素坐标系坐标
        self.plate_point = [] # 标定板(世界)坐标系坐标
        self.camera_intrinsic_path = 'jr_K.yaml'

    def main(self):
        while True:
            color_image = self.cam.get_frame()
            cv2.imshow('camera image', color_image)
            kvalue = cv2.waitKey(50) & 0xFF
            if kvalue == 27:
                break
            elif kvalue == 10 or kvalue == 13:
                ret_find_grid = self.mark2cam.find_circles_grid(color_image)
                print(f'ret_find_grid: {ret_find_grid}')
                if ret_find_grid:
                    self.mark2cam.show(color_image, ret_find_grid)
                    self.cam_plate_coners.append(self.mark2cam.cam_corners)
                    plate_point = self.mark2cam.get_mark_points()
                    self.plate_point.append(plate_point)
        cv2.destroyAllWindows()
        self.cam.release()  # 相机释放
        if len(self.cam_plate_coners) < 4:
            print(f'Camera Calibration Need More Data, Current: {len(self.cam_plate_coners)}')
        else:
            ret, mtx, dist, revcs, tvecs = cv2.calibrateCamera(self.plate_point, self.cam_plate_coners,
                                                              (self.cam.width, self.cam.height), None, None )
            # ret, mtx, dist, revcs, tvecs = cv2.calibrateCamera(self.plate_point, self.cam_plate_coners,
            #                                                   (self.cam.height, self.cam.width), None, None )
            print('matrix :' + '\n', mtx)
            print('distort :' + '\n', dist)
            print(f'ret: {ret}' )
            if ret < 1:
                fs = cv2.FileStorage(self.camera_intrinsic_path, cv2.FileStorage_WRITE)
                fs.write('cameraMatrix', mtx)
                fs.write('distCoeffs', dist)
                print('File Saved {}'.format(self.camera_intrinsic_path))


if __name__ == '__main__':

    CameraIntrinsicCalibrate().main()

