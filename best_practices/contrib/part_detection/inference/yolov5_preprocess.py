# Copyright 2023 Huawei Technologies Co., Ltd
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

import yaml
import json
import argparse
from tqdm import tqdm
import os
import numpy as np

try:
    from utils.datasets import create_dataloader
except:
    from utils.dataloaders import create_dataloader

from common.util.dataset import BatchDataLoader, evaluate
from common.util.model import forward_nms_op, forward_nms_script


def main(opt, cfg):
    if not os.path.exists(opt.prep_data):
        os.mkdir(opt.prep_data)

    if opt.nms_mode == "nms_op":
        if not os.path.exists(opt.img_info):
            os.mkdir(opt.img_info)
        # load dataset
        dataloader = BatchDataLoader(opt.data_path, batch_size=opt.batch_size)
        img_name_list = []
        for idx in tqdm(range(len(dataloader))):
            # load and preprocess dataset
            _, img, img_info, _, img_name, _ = dataloader[idx]
            img_name_list.append(img_name)
            # save img and img shape
            img.astype(np.float16).tofile("{}/{}.bin".format(opt.prep_data, idx))
            img_info.astype(np.float32).tofile("{}/{}.bin".format(opt.img_info, idx))
        np.save("img_name.npy", img_name_list)
        print(
            "The dataset has been processed.The image name is stored in img_name.npy,the shape of the image is stored in the img_info folder")

    elif opt.nms_mode == "nms_script":
        # load dataset
        single_cls = False if opt.tag >= 6.0 else opt.single_cls
        dataloader = create_dataloader(f"{opt.data_path}/val2017.txt", opt.img_size, opt.batch_size,
                                       max(cfg["stride"]), single_cls, pad=0.5)[0]
        path_list = []
        shapes_list = []
        i = 0
        for (img, targets, paths, shapes) in tqdm(dataloader):
            img = img.half()
            img /= 255.0  # 0 - 255 to 0.0 -1.0
            nb, _, height, width = img.shape  # batch size, channels, height, width
            img = img.numpy()
            img.astype(np.float16).tofile("{}/{}.bin".format(opt.prep_data, i))
            path_list.append(paths)
            shapes_list.append(shapes)
            i += 1

        np.save("path_list.npy", path_list)
        np.save("shapes_list.npy", shapes_list)
        print(
            "The dataset has been processed.The image path is stored in path_list.npy,the shape of the image is stored in the shapes_list.npy")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLOv5 offline model inference.')
    parser.add_argument('--data_path', type=str, default="coco", help='root dir for val images and annotations')
    parser.add_argument('--tag', type=float, default=6.1, help='yolov5 tags')
    parser.add_argument('--prep_data', type=str, default='./prep_data', help='prepared dataset')
    parser.add_argument('--nms_mode', type=str, default='nms_script', help='nms compute mode [nms_op/nms_script]')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--img_size', nargs='+', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--cfg_file', type=str, default='model.yaml', help='model parameters config file')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--img_info', type=str, default="./img_info", help='save the information of the images')
    parser.add_argument('--device-id', type=int, default=0, help='device id')
    opt = parser.parse_args()

    with open(opt.cfg_file) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    main(opt, cfg)
