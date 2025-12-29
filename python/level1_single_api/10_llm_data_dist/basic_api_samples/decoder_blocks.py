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

import time
from typing import List
import argparse
import logging
from llm_datadist import LLMDataDist, LLMRole, LLMConfig, CacheDesc, BlocksCacheKey, DataType, LLMClusterInfo, \
    LLMStatusCode, Placement, KvCache
import torch

# 用于被拉取KV的ip与port, 需要修改为实际device ip, 可以与多个Prompt建立链接
PROMPT_CLUSTER_ID_TO_DEVICE_IP_LIST = {
    0: ['192.168.1.1', '192.168.1.2', '192.168.1.3', '192.168.1.4',
        '192.168.1.5', '192.168.1.6', '192.168.1.7', '192.168.1.8'],
}
PROMPT_DEVICE_LISTEN_PORT = 26000
# Decoder使用的device信息, 需要修改为实际device ip
DECODER_DEVICE_IP_LIST = ['192.168.2.1', '192.168.2.2', '192.168.2.3', '192.168.2.4',
                          '192.168.2.5', '192.168.2.6', '192.168.2.7', '192.168.2.8']
DEVICE_ID_LIST = [0, 1, 2, 3, 4, 5, 6, 7]

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


def init_llm_datadist(rank_id: int) -> LLMDataDist:
    datadist = LLMDataDist(LLMRole.DECODER, cluster_id=0)
    llm_config = LLMConfig()
    llm_config.device_id = DEVICE_ID_LIST[rank_id]
    llm_options = llm_config.generate_options()
    datadist.init(llm_options)
    return datadist


def gen_cluster_info(rank_id: int) -> List[LLMClusterInfo]:
    cluster_info_list = []
    for prompt_cluster_id, device_ip_list in PROMPT_CLUSTER_ID_TO_DEVICE_IP_LIST.items():
        cluster_info = LLMClusterInfo()
        cluster_info.remote_cluster_id = prompt_cluster_id
        cluster_info.append_remote_ip_info(device_ip_list[rank_id], PROMPT_DEVICE_LISTEN_PORT)
        cluster_info.append_local_ip_info(DECODER_DEVICE_IP_LIST[rank_id], 0)
        cluster_info_list.append(cluster_info)
    return cluster_info_list


def _allocate_cpu_cache(block_size, num_block, num_tensors):
    cpu_addrs = []
    cpu_tensors = []
    for _ in range(num_tensors):
        kv_tensor = torch.rand(size=(num_block, block_size), dtype=torch.float16, device="cpu")
        cpu_addrs.append(kv_tensor.data_ptr())
        cpu_tensors.append(kv_tensor)
    cpu_cache_desc = CacheDesc(num_tensors=num_tensors, shape=[num_block, block_size],
                               data_type=DataType.DT_FLOAT16, placement=Placement.HOST)
    return KvCache.create_cpu_cache(cpu_cache_desc, cpu_addrs), cpu_tensors


def run_decoder_sample(rank_id: int):
    # 1. 初始化LLMDataDist
    datadist = init_llm_datadist(rank_id=rank_id)
    logging.info('[initialize] llm_datadist success')
    # 2. 和Prompt建立连接
    cluster_info_list = gen_cluster_info(rank_id)
    ret, rets = datadist.link_clusters(cluster_info_list, timeout=5000)
    if ret != LLMStatusCode.LLM_SUCCESS:
        raise RuntimeError(f'[link_cluster] failed, ret={ret}')
    logging.info('[link_cluster] success')

    # 3. 通过kv_cache_manager执行KvCache相关操作
    kv_cache_manager = datadist.kv_cache_manager
    cache_desc = CacheDesc(num_tensors=4, shape=[10, 128], data_type=DataType.DT_FLOAT16)
    kv_cache = kv_cache_manager.allocate_blocks_cache(cache_desc)
    logging.info('[allocate_blocks_cache] success')
    cache_key = BlocksCacheKey(prompt_cluster_id=0, model_id=0)
    # 等待prompt侧计算完kv cache
    time.sleep(1)
    # 可以指定拉取KV到指定block index
    kv_cache_manager.pull_blocks(cache_key, kv_cache, [0, 1], [2, 3])
    logging.info('[pull_blocks] success')
    kv_cache_manager.copy_blocks(kv_cache, {2: [0, 1]})
    logging.info('[copy_blocks] success')
    kv_host_tensors = kv_cache_manager.get_cache_tensors(kv_cache, 0)
    logging.info('[get_cache_tensor] success')
    logging.info(f'block_index = 0, tensor = {kv_host_tensors[0].numpy()[0, :]}')
    logging.info(f'block_index = 1, tensor = {kv_host_tensors[0].numpy()[1, :]}')
    # swap blocks
    cpu_cache, cpu_tensors = _allocate_cpu_cache(128, 10, 4)
    # swap out
    kv_cache_manager.swap_blocks(kv_cache, cpu_cache, {0: 0, 1: 1})
    # swap in
    kv_cache_manager.swap_blocks(cpu_cache, kv_cache, {0: 0, 1: 1})
    kv_cache_manager.deallocate_cache(kv_cache)
    logging.info('[deallocate_cache] success')
    # 4. Finalize流程
    ret, rets = datadist.unlink_clusters(cluster_info_list, timeout=5000)
    if ret != LLMStatusCode.LLM_SUCCESS:
        raise RuntimeError(f'[unlink_cluster] failed, ret={ret}')
    logging.info('[unlink_cluster] success')
    datadist.finalize()
    logging.info('[finalize] success')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank_id", type=int, default=0, help='rank_id')
    args = parser.parse_args()
    logging.info(f'Sample start, rank_id = {args.rank_id}, device_id = {DEVICE_ID_LIST[args.rank_id]}')
    run_decoder_sample(args.rank_id)
    logging.info('Sample end')
