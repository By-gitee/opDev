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

import torch
from tqdm import tqdm
import numpy as np

from pathlib import Path
from common.util.dataset import coco80_to_coco91_class, correct_bbox, save_coco_json

try:
    from utils.general import non_max_suppression, scale_coords  # tag > 2.0
except:
    from utils.utils import non_max_suppression, scale_coords  # tag = 2.0


def forward_nms_op(model, dataloader):
    pred_results = []
    for i in tqdm(range(len(dataloader))):
        # load and preprocess dataset
        valid_num, img, img_info, img0, img_name, shapes = dataloader[i]

        # om infer
        result = model.infer([img.astype(np.float16), img_info.astype(np.float16)])
        box_out = result[0]
        box_out_num = result[1]

        for idx in range(valid_num):
            # coordinate change
            num_det = int(box_out_num[idx][0])
            boxout = box_out[idx][:num_det * 6].reshape(6, -1).transpose().astype(np.float32)  # 6xN -> Nx6
            # append to COCO-JSON dictionary
            image_id = int(img_name[idx].split('.')[0])
            save_coco_json(boxout, pred_results, image_id, coco80_to_coco91_class())

    return pred_results


def forward_nms_script(model, dataloader, cfg):
    pred_results = []
    for (img, targets, paths, shapes) in tqdm(dataloader):
        img = img.half()
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        nb, _, height, width = img.shape  # batch size, channels, height, width

        padding = False
        batch_size = model.get_inputs()[0].shape[0]
        if nb != batch_size:
            img = np.pad(img, ((0, batch_size - nb), (0,0), (0,0),(0,0)), 'constant', constant_values=0)
            padding = True
        else:
            img = img.numpy()

        # om infer
        result = model.infer([img])
        if len(result) == 3:  # number of output nodes is 3, each shape is (bs, na, no, ny, nx)
            out = []
            for i in range(len(result)):
                anchors = torch.tensor(cfg['anchors'])
                stride = torch.tensor(cfg['stride'])
                cls_num = cfg['class_num']
                if padding == True:
                    result[i] = result[i][:nb]
                correct_bbox(result[i], anchors[i], stride[i], cls_num, out)
            box_out = torch.cat(out, 1)
        else:  # only use the first output node, which shape is (bs, -1, no)
            if padding == True:
                result[0] = result[0][:nb]
            box_out = torch.tensor(result[0])

        # non_max_suppression
        boxout = nms(box_out, conf_thres=cfg["conf_thres"], iou_thres=cfg["iou_thres"])
        for idx, pred in enumerate(boxout):
            try:
                scale_coords(img[idx].shape[1:], pred[:, :4], shapes[idx][0], shapes[idx][1])  # native-space pred
            except:
                pred = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
            # append to COCO-JSON dictionary
            path = Path(paths[idx])
            image_id = int(path.stem) if path.stem.isnumeric() else path.stem
            save_coco_json(pred, pred_results, image_id, coco80_to_coco91_class())

    return pred_results


def nms(box_out, conf_thres=0.4, iou_thres=0.5):
    try:
        boxout = non_max_suppression(box_out, conf_thres=conf_thres, iou_thres=iou_thres, multi_label=True)
    except:
        boxout = non_max_suppression(box_out, conf_thres=conf_thres, iou_thres=iou_thres)

    return boxout
