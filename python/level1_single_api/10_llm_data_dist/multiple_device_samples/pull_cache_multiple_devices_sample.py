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

import argparse
import json
import logging
import os
import subprocess
import time

from llm_datadist import *
import torch
import torch_npu
import torchair

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


def get_device_ips():
    device_ips = []
    for i in range(8):
        # 如果命令不存在，可在Ascend安装目录查找
        out = subprocess.check_output(["hccn_tool", "-i", str(i), "-ip", "-g"])
        device_ips.append(str(out).split("\\n")[0].split(":")[1])
    return device_ips


ALL_IPS = get_device_ips()


def init_llm_datadist(role: LLMRole, cluster_id) -> LLMDataDist:
    config_file = "numa_config.json" if role == LLMRole.PROMPT else "numa_config_1.json"
    os.environ["RESOURCE_CONFIG_PATH"] = os.path.join(os.getcwd(), config_file)

    datadist = LLMDataDist(role, cluster_id)
    llm_config = LLMConfig()
    cluster_config = {
        "logic_device_id": ["0:0:0:0", "0:0:1:0", "0:0:2:0", "0:0:3:0"]
    }
    if role == LLMRole.PROMPT:
        cluster_config["listen_ip_info"] = [{"ip": ip, "port": 26000} for ip in ALL_IPS[:4]]
    llm_config.cluster_info = json.dumps(cluster_config)
    # pull_cache超时时间，单位ms
    llm_config.sync_kv_timeout = 3000
    llm_options = llm_config.generate_options()
    llm_options.update({
        "ge.flowGraphMemMaxSize": "1000000000",
        "ge.session_device_id": '0' if role == LLMRole.PROMPT else "4"  # 需配置为当前进程所在device列表中的第一个
    })
    datadist.init(llm_options)
    logging.info("init suc.")
    return datadist


def link_unlink(datadist, is_link=True):
    cluster_info = LLMClusterInfo()
    cluster_info.remote_cluster_id = 1
    for i in range(4):
        cluster_info.append_local_ip_info(ALL_IPS[i + 4], 26000)
        cluster_info.append_remote_ip_info(ALL_IPS[i], 26000)
    func = datadist.link_clusters if is_link else datadist.unlink_clusters
    _, rets = func([cluster_info], 5000)
    for ret in rets:
        if ret != LLMStatusCode.LLM_SUCCESS:
            raise RuntimeError(f'link failed, ret = {ret}')


CACHE_DESC = CacheDesc(80, [2, 2 * 1024 * 1024], DataType.DT_FLOAT16)
PROMPT_CACHE_KEYS = [CacheKey(prompt_cluster_id=1, req_id=1), CacheKey(prompt_cluster_id=1, req_id=2)]


def run_decoder_sample(datadist):
    link_unlink(datadist)
    kv_cache_manager = datadist.kv_cache_manager
    kv_cache = kv_cache_manager.allocate_cache(CACHE_DESC)

    # wait prompt fill end.
    time.sleep(20)
    try:
        kv_cache_manager.pull_cache(PROMPT_CACHE_KEYS[0], kv_cache, batch_index=1)
    except LLMException as ex:
        logging.info(f"pull raise exception:{ex.status_code}")
        raise ex
    for device_index in [0, 1, 2, 3]:
        torch.npu.set_device(device_index + 4)
        kv_tensor_addrs = kv_cache.per_device_tensor_addrs[device_index]
        kv_tensors = torchair.llm_datadist.create_npu_tensors(kv_cache.cache_desc.shape, torch.float16, kv_tensor_addrs)
        for kv_tensor in kv_tensors:
            logging.info(f"after pull val={kv_tensor}")
    link_unlink(datadist, False)
    datadist.finalize()


def run_prompt_sample(datadist):
    kv_cache_manager = datadist.kv_cache_manager
    kv_cache = kv_cache_manager.allocate_cache(CACHE_DESC, PROMPT_CACHE_KEYS)
    for device_index in [0, 1, 2, 3]:
        torch.npu.set_device(device_index)
        kv_tensor_addrs = kv_cache.per_device_tensor_addrs[device_index]
        kv_tensors = torchair.llm_datadist.create_npu_tensors(kv_cache.cache_desc.shape, torch.float16, kv_tensor_addrs)
        for kv_tensor in kv_tensors:
            kv_tensor.fill_(2)
            logging.info(f"after fill val={kv_tensor}")

    time.sleep(40)
    datadist.finalize()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster_id", type=int, default=1, help='cluster id')
    args = parser.parse_args()
    if args.cluster_id not in [1, 2]:
        raise RuntimeError("Not supported cluster id")
    role = LLMRole.PROMPT if args.cluster_id == 1 else LLMRole.DECODER
    datadist = init_llm_datadist(role, args.cluster_id)
    if role == LLMRole.PROMPT:
        run_prompt_sample(datadist)
    else:
        run_decoder_sample(datadist)
    logging.info('Sample end')
