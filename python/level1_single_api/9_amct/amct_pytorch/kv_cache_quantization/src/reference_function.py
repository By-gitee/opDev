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
import time

import tqdm
import torch
import torch.nn as nn
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from accelerate import infer_auto_device_map, dispatch_model
from accelerate.utils.modeling import get_balanced_memory
from transformers import LlamaForCausalLM
from datasets import load_dataset
from google.protobuf import text_format

import amct_pytorch as amct
from amct_pytorch.proto import scale_offset_record_pb2


def get_llama2(model_path, seqlen=2048):
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)

    model.seqlen = seqlen
    return model

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
        # TODO: can we remove this?
        no_split_module_classes=[
            "OPTDecoderLayer",
            "LlamaDecoderLayer",
            "BloomBlock",
            "MPTBlock",
            "DecoderLayer",
        ],
        **kwargs,
    )
    model = dispatch_model(model, device_map=device_map, 
        offload_dir=os.path.join(model_path, 'offload_dir'))

    return model, enc


def get_loaders(dataset_path: str, enc, seqlen):
    print('Loading dataset c4/realnewslike')
    testenc = load_dataset(
            'json', 
            data_files={'validation':dataset_path},
            split = 'validation'
        )
    testenc = enc(' '.join(testenc[:1100]['text']), return_tensors='pt')
    testenc = testenc.input_ids[:, :(256 * seqlen)]

    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    testenc = TokenizerWrapper(testenc)
    
    return testenc
        
        
def read_kv_cache_factors(record_file, layer_name):
    records = scale_offset_record_pb2.ScaleOffsetRecord()
    with open(record_file, 'r') as read_file:
        pbtxt_string = read_file.read()
        text_format.Merge(pbtxt_string, records)

    done_flag = False
    scale = []
    offset = []
    for record in records.record:
        if record.key == layer_name:
            if not record.kv_cache_value.scale:
                raise RuntimeError("Cannot find scale of layer {} in record file".format(layer_name))
            scale.extend(record.kv_cache_value.scale)
            if not record.kv_cache_value.offset:
                raise RuntimeError("Cannot find offset of layer {} in record file".format(layer_name))
            offset.extend(record.kv_cache_value.offset)
            done_flag = True
            break
    if not done_flag:
        raise RuntimeError("Cannot find layer {} in record file".format(layer_name))
    return scale, offset


def do_quant(data, scale, offset):
    if len(scale) == 1:
        quant_data = torch.clip((data / scale[0] + offset[0]), min=-128, max=127).to(torch.int8)
    else:
        quant_channels = []
        for i, channel in enumerate(torch.chunk(data, data.shape[-1], dim=-1)):
            quant_channel = torch.clip(channel / scale[i] + offset[i], min=-128, max=127)
            quant_channels.append(quant_channel.to(torch.int8))
        quant_data = torch.cat(quant_channels, dim=-1)
    return quant_data


def do_antiquant(quant_data, scale, offset, data_type):
    if len(scale) == 1:
        antiquant_data = (quant_data.to(data_type) - offset[0]) * scale[0]
    else:
        antiquant_channels = []
        for i, channel in enumerate(torch.chunk(quant_data, quant_data.shape[-1], dim=-1)):
            antiquant_channel = (channel.to(data_type) - offset[i]) * scale[i]
            antiquant_channels.append(antiquant_channel)
        antiquant_data = torch.cat(antiquant_channels, dim=-1)
    return antiquant_data


if __name__ == '__main__':
    # Load model
    model_path = './model/Llama2_7b_hf'
    model = get_llama2(model_path)
    model = model.eval()
    gpus = os.getenv('CUDA_VISIBLE_DEVICES')
    if gpus == '' or gpus is None:
        gpu_num = 0
    else:
        gpu_num = len(gpus.split(','))
    model, enc = build_model_and_enc(model, model_path, gpu_num)
    model.seqlen = 2048

    # Load dataset
    dataset_path = './data/c4/realnewslike/c4-validation.00000-of-00001.json'
    testenc = get_loaders(dataset_path=dataset_path,
                          enc=enc, 
                          seqlen=model.seqlen)
    
    testenc = testenc.input_ids.to(model.device)
    
    # do inference
    nlls = []
    nsamples = testenc.numel() // model.seqlen
    test_start_time = time.time()
    for i in tqdm.tqdm(range(nsamples), desc="evaluating..."):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(model.device)
        with torch.no_grad():
            lm_logits = model(batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    test_end_time = time.time()
    
    ppl = torch.exp(torch.stack(nlls).sum() / (len(nlls) * model.seqlen))
    
    total_time = test_end_time - test_start_time
    print('Test time taken: ', total_time // 60, 'min ', total_time%60, 's'  )
    print('Score: ', ppl.item())