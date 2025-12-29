# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import glob

import cv2
import yaml
import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class BatchDataLoader:
    def __init__(self, data_path='coco', img_size=(640, 640), batch_size=4):
        self.img_path = glob.glob(os.path.join(data_path, 'val2017/*.jpg'))
        self.img_path.sort()
        self.img_num = len(self.img_path)
        self.img_size = img_size
        self.batch_size = batch_size

    def __len__(self):
        return self.img_num // self.batch_size + int(self.img_num % self.batch_size > 0)

    @staticmethod
    def read_data(img_path, img_size):
        img_name = os.path.basename(img_path)
        img0 = cv2.imread(img_path)
        imgh, imgw = img0.shape[:2]
        img, ratio, pad = letterbox(img0, new_shape=img_size)  # padding resize
        shape = (imgh, imgw), ((img0.shape[0] / imgh, img0.shape[1] / imgw), pad)  # for COCO mAP rescaling
        img_info = np.array([img_size[0], img_size[1], imgh, imgw], dtype=np.float32)
        return img0, img.astype(np.float32), img_info, img_name, shape

    def __getitem__(self, item):
        if (item + 1) * self.batch_size <= self.img_num:
            slice_end = (item + 1) * self.batch_size
            pad_num = 0
        else:
            slice_end = self.img_num
            pad_num = (item + 1) * self.batch_size - self.img_num

        img0 = []
        img = []
        img_info = []
        img_name = []
        shapes = []
        for path in self.img_path[item * self.batch_size:slice_end]:
            im0, im, info, name, shape = self.read_data(path, self.img_size)
            im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR tp RGB
            im /= 255.0
            im = np.ascontiguousarray(im)
            img.append(im)
            img_info.append(info)
            img0.append(im0)
            img_name.append(name)
            shapes.append(shape)
        valid_num = len(img)
        for _ in range(pad_num):
            img.append(img[0])
            img_info.append(img_info[0])
        return valid_num, np.stack(img, axis=0), np.stack(img_info, axis=0), img0, img_name, shapes


def evaluate(cocoGt_file, cocoDt_file):
    cocoGt = COCO(cocoGt_file)
    cocoDt = cocoGt.loadRes(cocoDt_file)
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


def coco80_to_coco91_class():
    # converts 80-index (val2014/val2017) to 91-index (paper)
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return x


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def make_grid(anchors, nx=20, ny=20):
    na = len(anchors) // 2  # number of anchors

    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    grid = torch.stack((xv, yv), 2).expand((1, na, ny, nx, 2)).float()
    anchor_grid = anchors.view((1, na, 1, 1, 2)).expand((1, na, ny, nx, 2)).float()

    return grid, anchor_grid


def correct_bbox(result, anchors, stride, cls_num, out):
    result = torch.tensor(result)
    bs, _, ny, nx, _ = result.shape
    grid, anchor_grid = make_grid(anchors, nx, ny)
    y = result.float().sigmoid()
    y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + grid) * stride  # xy
    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid  # wh
    out.append(y.view(bs, -1, cls_num+5))


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def save_coco_json(predn, pred_dict, image_id, class_map):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        pred_dict.append({'image_id': image_id,
                          'category_id': class_map[int(p[5])],
                          'bbox': [round(x, 3) for x in b],
                          'score': round(p[4], 5)})
