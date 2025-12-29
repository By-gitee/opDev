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
import cv2
import numpy as np
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YoloV5 offline model inference.')
    parser.add_argument('--calib_img_list', type=str, default="common/quantify/calib_img_list.txt", help='original data')
    parser.add_argument('--save_path', type=str, default="./calib_data", help='data for calibration')
    opt = parser.parse_args()

    images = []
    img_info = []
    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)

    with open(opt.calib_img_list, 'r') as file:
        calib_imgs = file.read().split('\n')

    for i, calib_img in enumerate(calib_imgs):
        img_path = calib_img.split()[1]
        print(img_path)
        img0 = cv2.imread(img_path)
        imgh, imgw = img0.shape[:2]
        img = cv2.resize(img0, (640, 640), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        img = np.expand_dims(img, axis=0)
        img /= 255.0

        images.append(img)
        img_info.append([640, 640, imgh, imgw])

    images = np.array(images, dtype=np.float16)
    images_bin_file = f"{opt.save_path}/images_bs{len(calib_imgs)}.bin"
    print(f"saving images bin file to {images_bin_file}")
    images.tofile(images_bin_file)

    img_info = np.array(img_info, dtype=np.float16)
    img_info_bin_file = f"{opt.save_path}/img_info_bs{len(calib_imgs)}.bin"
    print(f"saving img_info bin file to {img_info_bin_file}")
    img_info.tofile(img_info_bin_file)
