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
import torch
import torch.nn as nn
from datasets import load_dataset,load_from_disk

from transformers import AutoTokenizer, AutoConfig
from accelerate import infer_auto_device_map, dispatch_model
from accelerate.utils.modeling import get_balanced_memory

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

def get_llama2(model_path, seqlen=2048):
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    
    model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32, offload_folder="offload/")

    model.seqlen = seqlen
    return model, model_path


def get_loaders(data_path: str, enc, seqlen):
    print('Loading dataset: Wikitext2')
    testenc = load_dataset(data_path, 'wikitext-2-raw-v1', split='test', trust_remote_code=True)
    testenc = enc("\n\n".join(testenc["text"]), return_tensors="pt")
    
    return testenc


def get_calib_dataset(data_path="pileval", tokenizer=None, n_samples=512, block_size=512):
    dataset = load_from_disk(data_path)
    dataset = dataset.shuffle(seed=42)
    samples = []
    n_run = 0
    for data in dataset:
        line = data["text"]
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > 512:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    print(f" * Split into {n_split} blocks")
    return [
        cat_samples[:, i * block_size : (i + 1) * block_size] for i in range(n_split)
    ]
