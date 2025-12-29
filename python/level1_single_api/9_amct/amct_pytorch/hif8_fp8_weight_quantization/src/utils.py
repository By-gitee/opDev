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

def get_llama2(model_path, seqlen=2048):
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import LlamaForCausalLM
    
    model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, offload_folder="offload/")

    model.seqlen = seqlen
    return model, model_path


def get_loaders(data_path: str, enc, seqlen):

    print('Loading dataset: Wikitext2')
    testenc = load_dataset(data_path, 'wikitext-2-raw-v1', split='test', trust_remote_code=True)
    testenc = enc("\n\n".join(testenc["text"]), return_tensors="pt")
    
    return testenc


def get_calib_dataset(data_path, tokenizer=None, n_samples=512, block_size=512):
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
