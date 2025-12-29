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

import yaml
import json
import argparse
import cv2
import os
import numpy as np
from ais_bench.infer.interface import InferSession, MemorySummary
from ais_bench.infer.summary import summary

try:
    from utils.datasets import create_dataloader
except:
    from utils.dataloaders import create_dataloader

from common.util.dataset import BatchDataLoader, evaluate
from common.util.model import forward_nms_op, forward_nms_script


def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def main(opt, cfg):
    # load model
    model = InferSession(opt.device_id, opt.model)

    if opt.nms_mode == "nms_op":
        # load dataset
        dataloader = BatchDataLoader(opt.data_path, batch_size=opt.batch_size)

        # inference & nms
        pred_results = forward_nms_op(model, dataloader)

    elif opt.nms_mode == "nms_script":
        # load dataset
        single_cls = False if opt.tag >= 6.0 else opt
        print("current path :"+opt.data_path, flush=True)
        # coco_val
        dataloader = create_dataloader(f"{opt.data_path}/val2017.txt", opt.img_size, opt.batch_size, max(cfg["stride"]), single_cls, pad=0.5)[0]        
        # inference & nms
        pred_results = forward_nms_script(model, dataloader, cfg)
        

    s = model.sumary()
    summary.npu_compute_time_list = s.exec_time_list
    summary.h2d_latency_list = MemorySummary.get_H2D_time_list()
    summary.d2h_latency_list = MemorySummary.get_D2H_time_list()
    summary.report(opt.batch_size, output_prefix=None, display_all_summary=False)

    pred_json_file = f"{opt.model.split('.')[0]}_{opt.tag}_predictions.json"
    print(f'saving results to {pred_json_file}')
    with open(pred_json_file, 'w') as f:
        json.dump(pred_results, f)
    
    if opt.dataset == "coco":
        # evaluate coco mAP
        evaluate(opt.ground_truth_json, pred_json_file)
        print("Finished")
    elif opt.dataset == "Customised_dataset":
        if opt.visualization == "False":
            from eval import evaluate_customised_dataset
            # evaluate Customised dataset mAP
            evaluate_customised_dataset(opt.ground_truth_json, pred_json_file)
            print("Finished")
        elif opt.visualization == "True":
            # class_name 根据实际的数据集进行替换
            class_names = ['plug_defect', 'screw_defect', 'fan_defect', 'fan_screw_missing',
                'screw_tilt_defect', 'screw_type_defect_1', 'screw_type_defect_2']
            
            input_directory = os.path.join(opt.data_path, "val2017")
            
            if opt.single_data == "True":
                # 单图可视化
                file_names = os.listdir(input_directory)
                output_directory = "./runs/detect/singleimg/"
                create_directory_if_not_exists(output_directory)

                for file_name in file_names:
                    if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                        src = cv2.imread(os.path.join(input_directory, file_name))
                        dst = src.copy()                # 
                        
                        for i in pred_results:
                            bbox = list(map(int, i['bbox']))
                            pt1 = (bbox[0], bbox[1])
                            pt2 = (bbox[0] + bbox[2], bbox[1] + bbox[3])
                            dst = cv2.rectangle(dst, pt1, pt2, (0, 0, 255), 8)

                            category_id = i['category_id']
                            category_name = class_names[category_id - 1]
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = 3
                            font_thickness = 3
                            category_text = f"{category_name}"
                            text_size = cv2.getTextSize(category_text, font, font_scale, font_thickness)[0]
                            text_origin = (bbox[0], bbox[1] - text_size[1] - 5)
                            dst = cv2.putText(dst, category_text, text_origin, font, font_scale, (0, 0, 255), font_thickness)

                        combined_image = cv2.hconcat([src, dst])
                        output_path = os.path.join(output_directory, f"dst_{file_name}")

                        cv2.imwrite(output_path, combined_image)
                        print(f"saved combined inference image to: {output_path}")
                        
            elif opt.single_data == "False":
                # 多图可视化
                current_image_id = None
                current_image = None
                
                output_directory = "./runs/detect/multi_imgs/"
                create_directory_if_not_exists(output_directory)
                
                for i in pred_results:
                    image_id = i['image_id']
                    
                    if current_image_id != image_id:
                        if current_image is not None:
                            output_path = os.path.join(output_directory, f"dst_{current_image_id}.jpg")
                            cv2.imwrite(output_path, current_image)
                            print(f"saved inference image to: {output_path}")
                        
                        current_image_id = image_id
                        current_image = cv2.imread(os.path.join(input_directory, f"{image_id}.jpg"))
                    
                    bbox = list(map(int, i['bbox']))
                    pt1 = (bbox[0], bbox[1])
                    pt2 = (bbox[0] + bbox[2], bbox[1] + bbox[3])
                    current_image = cv2.rectangle(current_image, pt1, pt2, (0, 0, 255), 8)

                    category_id = i['category_id']
                    category_name = class_names[category_id - 1]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 3
                    font_thickness = 3
                    category_text = f"{category_name}"
                    text_size = cv2.getTextSize(category_text, font, font_scale, font_thickness)[0]
                    text_origin = (bbox[0], bbox[1] - text_size[1] - 5)
                    current_image = cv2.putText(current_image, category_text, text_origin, font, font_scale, (0, 0, 255), font_thickness)
                
                if current_image is not None:
                    output_path = os.path.join(output_directory, f"dst_{current_image_id}.jpg")
                    cv2.imwrite(output_path, current_image)
                    print(f"saved inference image to: {output_path}")
                    
                # eval
                from eval import evaluate_customised_dataset
                # evaluate Customised dataset mAP
                evaluate_customised_dataset(opt.ground_truth_json, pred_json_file)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLOv5 offline model inference.')    
    parser.add_argument('--data_path', type=str, default="coco", help='root dir for val images and annotations')
    parser.add_argument('--ground_truth_json', type=str, default="./coco/instances_val2017.json",
                        help='annotation file path')
    parser.add_argument('--tag', type=float, default=6.1, help='yolov5 tags')
    parser.add_argument('--model', type=str, default="yolov5s.om", help='om model path')
    parser.add_argument('--nms_mode', type=str, default="nms_op", help='nms compute mode [nms_op/nms_script]')
    parser.add_argument('--batch_size', type=int, default=0, help='batch size')
    parser.add_argument('--img_size', nargs='+', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--cfg_file', type=str, default='model.yaml', help='model4customised_dataset.yaml, model parameters config file')
    parser.add_argument('--device-id', type=int, default=0, help='device id')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    
    parser.add_argument('--dataset', type=str, default="Customised_dataset", help="Customised_dataset or coco")
    parser.add_argument('--visualization', type=str, default="True")
    parser.add_argument('--single_data', type=str, default="True")
    
    opt = parser.parse_args()
    
    with open(opt.cfg_file) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    main(opt, cfg)