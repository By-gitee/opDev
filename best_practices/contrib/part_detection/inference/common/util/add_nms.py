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

import sys
import yaml
import argparse
import onnx
from onnx import helper

from models.experimental import attempt_load

ROOT = './'
if ROOT not in sys.path:
    sys.path.append(ROOT)  # add ROOT to PATH


def ceil_x(value, align_len):
    return (value + align_len - 1) // align_len * align_len


def main(opt, cfg):
    # load pth to get the anchors
    pt_model = attempt_load(opt.pt_path, map_location='cpu')
    m = pt_model.module.model[-1] if hasattr(pt_model, 'module') else pt_model.model[-1]
    for i in range(m.nl):
        m.anchors[i] *= m.stride[i]

    # load onnx to modify
    onnx_model = onnx.load(opt.onnx_path)

    # create yolo pre-detection layer
    h, w = cfg["img_size"]
    f_h, f_w = h // 8, w // 8
    for i in range(m.nl):
        crd_align_len = ceil_x(f_h * f_w * 2 + 32, 32) // 2
        obj_align_len = ceil_x(m.na * f_h * f_w * 2 + 32, 32) // 2

        helper.make_tensor_value_info(f"yolo{i}_coord", onnx.TensorProto.FLOAT, ['batch', m.na * 4, crd_align_len])
        helper.make_tensor_value_info(f"yolo{i}_obj", onnx.TensorProto.FLOAT, ['batch', obj_align_len])
        helper.make_tensor_value_info(f"yolo{i}_classes", onnx.TensorProto.FLOAT,
                                      ['batch', cfg["class_num"], obj_align_len])

        yolo_pre_node = helper.make_node('YoloPreDetection',
                                         inputs=[onnx_model.graph.output[i].name],
                                         outputs=[f"yolo{i}_coord", f"yolo{i}_obj", f"yolo{i}_classes"],
                                         boxes=m.na,
                                         coords=4,
                                         classes=cfg["class_num"],
                                         yolo_version='V5',
                                         name=f'yolo_{i}')
        onnx_model.graph.node.append(yolo_pre_node)
        f_h, f_w = f_h // 2, f_w // 2

    # create yolo detection output layer
    img_info = helper.make_tensor_value_info("img_info", onnx.TensorProto.FLOAT, ['batch', 4])
    box_out = helper.make_tensor_value_info("box_out", onnx.TensorProto.FLOAT, ['batch', 6 * 1024])
    box_out_num = helper.make_tensor_value_info("box_out_num", onnx.TensorProto.INT32, ['batch', 8])

    yolo_detout_node = helper.make_node('YoloV5DetectionOutput',
                                        inputs=[f"yolo{i}_coord" for i in range(m.nl)] +
                                               [f"yolo{i}_obj" for i in range(m.nl)] +
                                               [f"yolo{i}_classes" for i in range(m.nl)] +
                                               ['img_info'],
                                        outputs=['box_out', 'box_out_num'],
                                        boxes=m.na,
                                        coords=4,
                                        classes=cfg["class_num"],
                                        pre_nms_topn=1024,
                                        post_nms_topn=1024,
                                        relative=1,
                                        out_box_dim=2,
                                        obj_threshold=cfg["conf_thres"],
                                        score_threshold=cfg["conf_thres"],
                                        iou_threshold=cfg["iou_thres"],
                                        biases=m.anchors.numpy().flatten().astype('float16').tolist(),
                                        name='YoloV5DetectionOutput_1')

    # add input and output
    onnx_model.graph.node.append(yolo_detout_node)
    while len(onnx_model.graph.output) > 0:
        onnx_model.graph.output.remove(onnx_model.graph.output[0])

    onnx_model.graph.input.append(img_info)
    onnx_model.graph.output.append(box_out)
    onnx_model.graph.output.append(box_out_num)

    onnx.save(onnx_model, opt.onnx_path.split('.onnx')[0] + "_nms.onnx")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Add NMS operator")
    parser.add_argument('--pt-path', type=str, default='./yolov5s.pt', help='pt_model path')
    parser.add_argument('--onnx-path', type=str, default='./yolov5s.onnx', help='onnx_model path')
    parser.add_argument('--cfg-file', type=str, default='model.yaml', help='model parameters config file')
    opt = parser.parse_args()

    with open(opt.cfg_file) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    main(opt, cfg)
