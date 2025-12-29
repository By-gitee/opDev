"""
# Copyright 2025 Huawei Technologies Co., Ltd
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
import copy
import time
import tqdm
import torch
import argparse
import torch.nn as nn

from utils import get_llama2, get_calib_dataset, build_model_and_enc
import amct_pytorch as amct
from amct_pytorch.post_quant_calibration import LLMHelper


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--calibration_data', type=str, default='/pile_val_backup')
    parser.add_argument('--model', type=str, default='/data/Models/pytorch/Llama2/Llama2_7b_hf')
    parser.add_argument('--finetune', type=bool, default=False)

    args = parser.parse_args()
    model, model_path = get_llama2(args.model)
    model = model.eval()
    gpu_num = torch.cuda.device_count()
    model, enc = build_model_and_enc(model, model_path, gpu_num)

    proto_file = './config/lut4_quant.cfg'
    config_file = './output/config.json'
    record_file = './output/record.txt'

    test_start_time = time.time()
    # Phase1: generate quant config json
    amct.create_post_quant_config(config_file,
                             model,
                             config_defination=proto_file)
    
    # Phase2: generate calibration model
    samples = get_calib_dataset(
        data_path=args.calibration_data, tokenizer=enc, n_samples=512, block_size=256
    )
    samples = torch.cat(samples, dim=0)[:1,:]
    # do weights calibration without finetune
    # Please check README.md for LLMHelper usage
    with torch.no_grad():
        post_quant_model = amct.create_post_quant_model(config_file,
                                                        record_file,
                                                        model)
    calibration_helper = LLMHelper(post_quant_model, samples, calibration_block='LlamaDecoderLayer', layer_filter=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    post_quant_model.config.use_cache = False
    amct.quant_calibration(calibration_helper)
     # do weights calibration with finetune
    if args.finetune:
        with torch.no_grad():
            post_quant_model = amct.create_post_quant_model(config_file,
                                                            record_file,
                                                            post_quant_model)
        calibration_finetune_helper = LLMHelper(post_quant_model, samples, calibration_block='LlamaDecoderLayer', layer_filter=True)                                                   
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        post_quant_model.config.use_cache = False
        amct.quant_calibration(calibration_finetune_helper)
    test_end_time = time.time()
    total_time = test_end_time - test_start_time
    print('Calibration success, time taken: ', total_time // 60, 'min ', total_time%60, 's')
