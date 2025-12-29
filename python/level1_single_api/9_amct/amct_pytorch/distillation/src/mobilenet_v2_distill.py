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

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import amct_pytorch as amct
import onnxruntime as ort


PATH = os.path.realpath('./')
IMG_DIR = os.path.join(PATH, 'data/images')
LABEL_FILE = os.path.join(IMG_DIR, 'image_label.txt')
TMP = os.path.join(PATH, 'tmp')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument(
    '--config_defination', dest='config_defination', default=None, type=str, help='The simple configure define file.')
parser.add_argument(
    '--num_parallel_reads', dest='num_parallel_reads', default=4, type=int,
    help='The number of files to read in parallel.')
parser.add_argument(
    '--dist_url', dest='dist_url', default='tcp://127.0.0.1:50011', type=str,
    help='url used to set up distributed training')
parser.add_argument(
    '--distributed', dest='distributed', action='store_true', help='Use multi-processing distributed training')
parser.add_argument('--batch_size', dest='batch_size', default=32, type=int, help='batch size (default: 32)')

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
    input_tensor = torch.zeros(len(images), 3, 224, 224)
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


def model_forward(model, batch_size, iterations):
    """Do pytorch model forward"""
    images, labels = get_labels_from_txt(LABEL_FILE)
    images = [os.path.join(IMG_DIR, image) for image in images]
    top1_total = 0
    top5_total = 0
    for i in range(iterations):
        input_batch = prepare_image_input(
            images[i * batch_size: (i + 1) * batch_size])
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
        with torch.no_grad():
            output = model(input_batch)
        top1, top5 = img_postprocess(
            output, labels[i * batch_size: (i + 1) * batch_size])
        top1_total += top1
        top5_total += top5
    return top1_total / iterations, top5_total / iterations


class MiniDataset(Dataset):
    def __init__(self, gpu_index):
        self.images, _ = get_labels_from_txt(LABEL_FILE)
        self.gpu_index = gpu_index

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        preprocess = transforms.Compose(
            [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        input_image = Image.open(os.path.join(
            IMG_DIR, self.images[idx])).convert('RGB')
        input_tensor = preprocess(input_image)
        if self.gpu_index >= 0:
            input_tensor = input_tensor.to(f'cuda:{self.gpu_index}')
        return input_tensor


def onnx_forward(onnx_model, batch_size, iterations):
    """Do onnx model forward"""
    ort_session = ort.InferenceSession(onnx_model)

    images, labels = get_labels_from_txt(LABEL_FILE)
    images = [os.path.join(IMG_DIR, image) for image in images]
    top1_total = 0
    top5_total = 0
    for i in range(iterations):
        input_batch = prepare_image_input(
            images[i * batch_size: (i + 1) * batch_size])
        output = ort_session.run(None, {'in_data.1': input_batch.numpy()})
        top1, top5 = img_postprocess(
            output[0], labels[i * batch_size: (i + 1) * batch_size])
        top1_total += top1
        top5_total += top5
    return top1_total / iterations, top5_total / iterations

def main():
    args = parser.parse_args()

    if torch.cuda.is_available():
        gpu_num = torch.cuda.device_count()
    else:
        gpu_num = 0

    if args.distributed:
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function.
        if gpu_num == 0:
            raise RuntimeError('Must have at least one available GPU in distributed training mode!')
        print('Using multi GPUs: DistributedDataParallel mode.')
        mp.spawn(main_worker, nprocs=gpu_num, args=(gpu_num, args))
    else:
        # Simply call main_worker function.
        if gpu_num > 0:
            gpu_index = 0
            print('Using single GPU.')
        else:
            gpu_index = -1
            print('Using CPU, this will be slow')
        main_worker(gpu_index, gpu_num, args)

def main_worker(gpu_index, gpu_num, args):
    if args.distributed:
        dist.init_process_group(backend='nccl', init_method=args.dist_url, world_size=gpu_num, rank=gpu_index)
    model = models.mobilenet_v2(pretrained=True)
    model = model.eval()
    if gpu_num > 0:
        if args.distributed:
            torch.cuda.set_device(gpu_index)
            model.cuda(gpu_index)
            model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu_index], find_unused_parameters=True)
        else:
            torch.cuda.set_device(gpu_index)
            model = model.cuda(gpu_index)
    
    ori_top1, ori_top5 = model_forward(model, batch_size=args.batch_size, iterations=5)
    args_shape = (args.batch_size, 3, 224, 224)
    input_data = torch.randn(args_shape)
    if torch.cuda.is_available():
        input_data = input_data.to('cuda')

    # 1. step1 create distill config json file
    config_json_file = os.path.join(TMP, 'distill_config.json')
    amct.create_distill_config(
        config_json_file,
        model,
        input_data,
        args.config_defination
    )

    # 2. step2 create distill student model
    student_model = amct.create_distill_model(
        config_json_file, model, input_data)

    # 3. step3 do distill to quantize the model
    dataset = MiniDataset(gpu_index)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        train_sampler = None

    if gpu_num >= 0:
        num_workers = 0
        pin_memory = False
    else:
        num_workers = args.num_parallel_reads
        pin_memory = True
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                             num_workers=num_workers, pin_memory=pin_memory, sampler=train_sampler)
    student_model = amct.distill(
        model, student_model, config_json_file, data_loader, 1)

    # 4. step4 save distilled model
    if gpu_index == 0 or gpu_index == -1:
        result_path = os.path.join(PATH, 'result/mobilenet_v2')
        record_file = os.path.join(TMP, 'scale_offset_record.txt')
        amct.save_distill_model(student_model, result_path,
                                input_data, record_file)
        quant_top1, quant_top5 = onnx_forward(
            '%s_%s' % (result_path, 'fake_quant_model.onnx'), batch_size=args.batch_size, iterations=5)
        print('[INFO] MobilenetV2 before quantize top1:{:>10} top5:{:>10}'.format(
            ori_top1, ori_top5))
        print('[INFO] MobilenetV2 after quantize  top1:{:>10} top5:{:>10}'.format(
            quant_top1, quant_top5))


if __name__ == "__main__":
    main()
