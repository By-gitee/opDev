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

import argparse
import os
import copy
import time
import tqdm
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig
from accelerate import infer_auto_device_map, dispatch_model
from accelerate.utils.modeling import get_balanced_memory

from utils import get_loaders,  get_llama2, get_calib_dataset
import amct_pytorch as amct


def build_model_and_enc(model, model_path, gpu_num):
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if "mpt" in config.__class__.__name__.lower():
        enc = AutoTokenizer.from_pretrained(
            config.tokenizer_name, trust_remote_code=True
        )
    else:
        enc = AutoTokenizer.from_pretrained(
            model_path, use_fast=False, trust_remote_code=True
        )

    # Move the model to GPU (as much as possible) for LM evaluation
    # max_memory = ['0:16GiB', '1:16GiB','2:16GiB', 'cpu:30GiB'], '0' means the first GPU that you specify.
    # I don't recommend use 16GiB, we need to reserve some space for other tensors during calculation
    # please see the recommand memeory allocation in the Word file
    # Adjust the max_size accroding to the real situation
    # a clever way:

    max_memory = []
    for i in range(gpu_num):
        max_memory.append(f'{i}:12GiB')
    max_memory.append('cpu:80GiB')
    print('Max_memory allocation: \n', max_memory)

    max_memory = [v.split(":") for v in (max_memory or [])]
    max_memory = {(int(k) if k.isdigit() else k): v for k, v in max_memory}
    kwargs = {
        "max_memory": get_balanced_memory(
            model, max_memory if len(max_memory) > 0 else None
        )
    }
    model.tie_weights()
    device_map = infer_auto_device_map(
        model,
        no_split_module_classes=[
            "LlamaDecoderLayer",
        ],
        **kwargs,
    )
    model = dispatch_model(model, device_map=device_map, 
        offload_dir=os.path.join(model_path, 'offload_dir'))

    return model, enc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--calibration_data', type=str, default='/pile_val_backup')
    parser.add_argument('--verify_data', type=str, default='/data/Datasets/wikitext/wikitext-2-raw-v1/wikitext-2-raw/wikiscript.py')
    parser.add_argument('--model', type=str, default='/data/Models/pytorch/Llama2/Llama2_7b_hf')

    args = parser.parse_args()
    model, model_path = get_llama2(args.model)
    model = model.eval()
    copied_model = copy.deepcopy(model)
    gpu_num = torch.cuda.device_count()
    model, enc = build_model_and_enc(model, model_path, gpu_num)

    proto_path = './src/quantization.cfg'
    config_file = './output/config.json'
    record_file = './output/record.txt'

    test_start_time = time.time()
    # Phase1: generate quant config json
    amct.create_post_quant_config(config_file,
                             model,
                             config_defination=proto_path)
    
    # Phase2: do weights calibration and generate calibration model
    samples = get_calib_dataset(
        data_path=args.calibration_data, tokenizer=enc, n_samples=512, block_size=518
    )
    samples = torch.cat(samples, dim=0)[:1,:]

    post_quant_model = amct.create_post_quant_model(config_file,
                                                    record_file,
                                                    model)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    post_quant_model.config.use_cache = False
    with torch.no_grad():
        post_quant_model(samples.to(next(post_quant_model.parameters()).device))
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    test_end_time = time.time()
    total_time = test_end_time - test_start_time
    print('Calibration time taken: ', total_time // 60, 'min ', total_time%60, 's')
    # save memory, del unuse model
    del post_quant_model
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model, enc = build_model_and_enc(copied_model, model_path, gpu_num)
    
    # Phase3: save fakequant model
    testenc = get_loaders(data_path=args.verify_data,
                        enc=enc,
                        seqlen=model.seqlen)

    testenc = testenc.input_ids.to(model.device)

    quant_model = amct.save_post_quant_model(record_file, model, mode='fakequant')

    nsamples = testenc.numel() // model.seqlen
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Phase4: Test ppl result
    nlls = []
    test_start_time = time.time()
    for i in tqdm.tqdm(range(nsamples), desc="evaluating..."):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(
            quant_model.device
        )
        with torch.no_grad():
            lm_logits = quant_model(batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous().float().cpu()
        shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:].cpu()
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    test_end_time = time.time()

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    total_time = test_end_time - test_start_time
    print('Test time taken: ', total_time // 60, 'min ', total_time%60, 's'  )
    print('Score: ', ppl.item())