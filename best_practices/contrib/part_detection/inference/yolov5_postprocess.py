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
import re
import os
import sys
import torch
import numpy as np
from auto_optimizer import OnnxGraph
from tqdm import tqdm
from pathlib import Path

try:
    from utils.general import non_max_suppression, scale_coords  # tag > 2.0
except:
    from utils.utils import non_max_suppression, scale_coords  # tag = 2.0

from common.util.dataset import coco80_to_coco91_class, correct_bbox, save_coco_json, evaluate


def postprocess(opt, cfg):
    onnx_session = OnnxGraph.parse(opt.onnx)
    outputs = onnx_session.outputs
    reference_list = os.listdir(opt.output)

    if opt.nms_mode == "nms_script":
        # the output shapes
        shapes = [[opt.batch_size, 3, 80, 80, 85], [opt.batch_size, 3, 40, 40, 85], [opt.batch_size, 3, 20, 20, 85]]
        path_list = np.load("path_list.npy", allow_pickle=True)
        shapes_list = np.load("shapes_list.npy", allow_pickle=True)
        pred_results = []
        for i in tqdm(range(len(reference_list) // len(outputs))):
            if len(outputs) == 3:
                out = []
                # single img infer 3 output
                for output_num in range(len(outputs)):
                    out_filepath = f"{opt.output}/{i}_{output_num}.bin"
                    inference_result = np.fromfile(out_filepath, dtype=np.float16)
                    inference_result = inference_result.reshape(shapes[output_num])

                    anchors = torch.tensor(cfg['anchors'])
                    stride = torch.tensor(cfg['stride'])
                    cls_num = cfg['class_num']
                    correct_bbox(inference_result, anchors[output_num], stride[output_num], cls_num, out)

                box_out = torch.cat(out, 1)

            else:
                out_filepath = f"{opt.output}/{i}_0.bin"
                data = np.fromfile(out_filepath, dtype=np.float16)
                box_out = torch.tensor(data.reshape(opt.batch_size, -1, 85))

            # non_max_suppression
            boxout = nms(box_out, conf_thres=cfg["conf_thres"], iou_thres=cfg["iou_thres"])

            for idx, pred in enumerate(boxout):
                try:
                    scale_coords((640, 640), pred[:, :4], shapes_list[i][idx][0][:])

                except:
                    pred = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
                # append to COCO-JSON dictionary
                path = Path(path_list[i][idx])
                image_id = int(path.stem) if path.stem.isnumeric() else path.stem
                save_coco_json(pred, pred_results, image_id, coco80_to_coco91_class())

    elif opt.nms_mode == "nms_op":
        img_name = np.load("img_name.npy", allow_pickle=True)
        pred_results = []
        for i in tqdm(range(len(reference_list) // 2)):  # op type has 2 output
            # single img infer 3 output
            box_out_filename = f"{opt.output}/{i}_0.bin"
            box_out_num_filename = f"{opt.output}/{i}_1.bin"
            box_out = np.fromfile(box_out_filename, dtype=np.float16).reshape(opt.batch_size, 6144)
            box_out_num = np.fromfile(box_out_num_filename, dtype=np.float16).reshape(opt.batch_size, 8)

            for idx in range(opt.batch_size):
                # coordinate change
                try:
                    num_det = int(box_out_num[idx][0])
                except:
                    continue
                boxout = box_out[idx][:num_det * 6].reshape(6, -1).transpose().astype(np.float32)  # 6xN -> Nx6
                # append to COCO-JSON dictionary
                image_id = int(img_name[i][idx].split('.')[0])
                save_coco_json(boxout, pred_results, image_id, coco80_to_coco91_class())

    pred_json_file = f"{opt.onnx.split('.')[0]}_predictions.json"

    with open(pred_json_file, 'w') as f:
        json.dump(pred_results, f)
    print(f"saving results to {pred_json_file}")

    # evaluate mAP
    evaluate(opt.ground_truth_json, pred_json_file)


def nms(box_out, conf_thres=0.4, iou_thres=0.5):
    try:
        boxout = non_max_suppression(box_out, conf_thres=conf_thres, iou_thres=iou_thres, multi_label=True)
    except:
        boxout = non_max_suppression(box_out, conf_thres=conf_thres, iou_thres=iou_thres)

    return boxout


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLOv5 offline model inference.')
    parser.add_argument('--ground_truth_json', type=str, default="coco/instances_val2017.json",
                        help='annotation file path')
    parser.add_argument('--onnx', type=str, default="yolov5s.onnx", help='om model path')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--cfg_file', type=str, default='model.yaml', help='model parameters config file')
    parser.add_argument('--output', type=str, help='ais_bench inference output path')
    parser.add_argument('--img_info', type=str, default="img_info.json", help='ais_bench inference output path')
    parser.add_argument('--nms_mode', type=str, default="nms_script", help='ais_bench inference output path')

    opt = parser.parse_args()

    with open(opt.cfg_file) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    postprocess(opt, cfg)
