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

from utils import get_loaders,  get_llama2, get_calib_dataset, build_model_and_enc
import amct_pytorch as amct


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify_data', type=str, default='/data/Datasets/wikitext/wikitext-2-raw-v1/wikitext-2-raw/wikiscript.py')
    parser.add_argument('--model', type=str, default='/data/Models/pytorch/Llama2/Llama2_7b_hf')

    args = parser.parse_args()
    model, model_path = get_llama2(args.model)
    model = model.eval()
    gpu_num = torch.cuda.device_count()

    record_file = './output/record.txt'

    test_start_time = time.time()
    model, enc = build_model_and_enc(model, model_path, gpu_num)
    
    # Phase1: save fakequant model
    testenc = get_loaders(data_path=args.verify_data,
                        enc=enc,
                        seqlen=model.seqlen)

    testenc = testenc.input_ids.to(model.device)
    nsamples = testenc.numel() // model.seqlen
    fake_quant_model = amct.save_post_quant_model(record_file, model, mode='fakequant')
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Phase2: Test ppl result
    nlls = []
    test_start_time = time.time()
    for i in tqdm.tqdm(range(nsamples), desc="evaluating..."):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(
            model.device
        )
        with torch.no_grad():
            lm_logits = fake_quant_model(batch).logits
        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:]
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