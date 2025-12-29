#!/usr/bin/env python
# coding=utf-8
import cv2
import numpy as np


class Camera(object):
    def __init__(self, width=640, height=480, id=0):
        self.width = width
        self.height = height
        self.cap = cv2.VideoCapture(id)
        # cv2.namedWindow("frame", 0)

    def get_frame(self):
        ret, frame = self.cap.read()
        return ret, frame


if __name__ == '__main__':
    Cam = Camera()

    while True:
        ret, frame = Cam.get_frame()
        print(frame.shape)
        cv2.imshow('frame', frame)
        k = cv2.waitKey(3)
        if k == 27:
            cv2.destroyAllWindows()
            break




