"""
# Copyright 2024 Huawei Technologies Co., Ltd
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
"""


import os
import argparse
import copy
import torch # pylint: disable=E0401
from PIL import Image # pylint: disable=E0401
from torchvision import transforms # pylint: disable=E0401

import amct_pytorch as amct # pylint: disable=E0401
from resnet import resnet101 # pylint: disable=E0401, C0415


PATH = os.path.realpath('./')
IMG_DIR = os.path.join(PATH, 'data/images')
LABEL_FILE = os.path.join(IMG_DIR, 'image_label.txt')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--test-on-npu-flag', dest='test_on_npu_flag', help='If test in the NPU environment', action="store_true")
args = parser.parse_args()


OUTPUTS = os.path.join(PATH, 'outputs/calibration')

TMP = os.path.join(OUTPUTS, 'tmp')


def get_labels_from_txt(label_file):
    """Read all images' name and label from label_file"""
    images = []
    labels = []
    with open(label_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            images.append(line.split(' ')[0])
            labels.append(int(line.split(' ')[1]))
    return images, labels


def prepare_image_input(images):
    """Read all images"""
    input_tensor = torch.zeros(len(images), 3, 224, 224) # pylint: disable=E1101
    preprocess = transforms.Compose(
        [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    for index, image in enumerate(images):
        input_image = Image.open(image).convert('RGB')
        input_tensor[index, ...] = preprocess(input_image)
    return input_tensor


def img_postprocess(probs, labels):
    """Do image post-process"""
    # calculate top1 and top5 accuracy
    top1_get = 0
    top5_get = 0
    prob_size = probs.shape[1]
    for index, label in enumerate(labels):
        top5_record = (probs[index, :].argsort())[prob_size - 5: prob_size]
        if label == top5_record[-1]:
            top1_get += 1
            top5_get += 1
        elif label in top5_record:
            top5_get += 1
    return float(top1_get) / len(labels), float(top5_get) / len(labels)


def model_forward(model, batch_size, iterations, npu_flag=False):
    """Do pytorch model forward"""
    images, labels = get_labels_from_txt(LABEL_FILE)
    images = [os.path.join(IMG_DIR, image) for image in images]
    top1_total = 0
    top5_total = 0
    for i in range(iterations):
        input_batch = prepare_image_input(images[i * batch_size: (i + 1) * batch_size])
        # move the input and model to GPU or NPU for speed if available
        if npu_flag:
            try:
                import torch_npu
            except ImportError as exception:
                raise ImportError(exception, "torch_npu pkg is necessary for deploy model")
            torch.npu.set_compile_mode(jit_compile=False)
            torch.npu.config.allow_internal_format = False
            torch.npu.conv.allow_hf32 = False
            torch_npu.npu.set_device(0)
            if torch_npu.npu.is_available():
                input_batch = input_batch.to('npu')
                model.to('npu')
            else:
                raise RuntimeError('The current environment does not support NPU')
        else:
            if torch.cuda.is_available():
                input_batch = input_batch.to('cuda')
                model.to('cuda')

        with torch.no_grad():
            output = model(input_batch)
        top1, top5 = img_postprocess(output, labels[i * batch_size: (i + 1) * batch_size])
        top1_total += top1
        top5_total += top5
        print('****************iteration:{}*****************'.format(i))
        print('top1_acc:{}'.format(top1))
        print('top5_acc:{}'.format(top5))
    print('******final top1:{}'.format(top1_total / iterations))
    print('******final top5:{}'.format(top5_total / iterations))
    return top1_total / iterations, top5_total / iterations


def main():
    """Sample main function"""
    model = resnet101(pretrained=True)
    model.eval()
    copied_model = copy.deepcopy(model)
    ori_top1, ori_top5 = model_forward(model, batch_size=32, iterations=5)

    # Quantize configurations
    if torch.cuda.is_available():
        model.to('cuda')
    config_file = os.path.join(TMP, 'config.json')

    config_defination = os.path.join(PATH, 'src/quant_conf/quant.cfg')
    amct.create_post_quant_config(config_file, model, config_defination=config_defination)

    # Phase1: do weights calibration and generate calibration model
    record_file = os.path.join(TMP, 'record.txt')
    quant_post_model = amct.create_post_quant_model(config_file=config_file, 
                                                   record_file=record_file, 
                                                   model=model)

    # Phase2: do calibration
    batch_num = 2
    model_forward(quant_post_model, batch_size=32, iterations=batch_num)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Phase3: save final model, fake_quant_model for CPU/GPU
    fake_quant_model = amct.save_post_quant_model(record_file, copy.deepcopy(copied_model), mode='fakequant')

    # Phase4: run fake_quant_model test
    quant_top1, quant_top5 = model_forward(fake_quant_model, batch_size=32, iterations=5)
    print('[INFO] ResNet101 before quantize top1:{:>10} top5:{:>10}'.format(ori_top1, ori_top5))
    print('[INFO] ResNet101 after quantize  top1:{:>10} top5:{:>10}'.format(quant_top1, quant_top5))

    if args.test_on_npu_flag:
        # Phase5: run deploy_model test for npu
        deploy_model = amct.save_post_quant_model(record_file, copy.deepcopy(copied_model), mode='deploy')
        quant_top1, quant_top5 = model_forward(deploy_model, batch_size=32, iterations=5, npu_flag=True)
        print('[INFO] NPU: ResNet101 before quantize top1:{:>10} top5:{:>10}'.format(ori_top1, ori_top5))
        print('[INFO] NPU: ResNet101 after quantize  top1:{:>10} top5:{:>10}'.format(quant_top1, quant_top5))

if __name__ == '__main__':
    main()
