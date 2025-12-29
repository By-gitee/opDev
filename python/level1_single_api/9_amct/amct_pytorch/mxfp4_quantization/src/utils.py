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

import torch
import torch.nn as nn
from datasets import load_dataset,load_from_disk

def get_llama2(model, seqlen=2048):
    '''If model is specified from ['7b', '13b', '70b'], then we load official pretrained model;
       If you want to load checkpoints other than the official ones, please specifiy the model path,
       otherwise please choose from ['7b', '13b', '70b'] for better clarity
    '''

    def skip(*args, **kwargs):
        pass

    if model in ['7b', '13b', '70b']:
        model_path = f'/data/Models/pytorch/Llama2/Llama2_{model}_hf'
        print(f'Getting official pretrained Llama2-{model}')
    else:
        model_path = model
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    
    model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, offload_folder="offload/")

    model.seqlen = seqlen
    return model, model_path


def get_loaders(dataset_name: str, enc, seqlen):
    if dataset_name == 'wikitext2':
        print('Loading dataset: Wikitext2')
        testenc = load_dataset('/data/Datasets/wikitext/wikitext-2-raw-v1/wikitext-2-raw/wikiscript.py', 'wikitext-2-raw-v1', split='test', trust_remote_code=True)
        testenc = enc("\n\n".join(testenc["text"]), return_tensors="pt")
    
    return testenc


def get_calib_dataset(data="pileval", tokenizer=None, n_samples=512, block_size=512):
    if data == "pileval":
        dataset = load_from_disk('/pile_val_backup')
    else:
        raise NotImplementedError
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
