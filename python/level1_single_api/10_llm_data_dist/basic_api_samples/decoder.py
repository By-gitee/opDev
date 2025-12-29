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


from typing import List
import argparse
import logging
from llm_datadist import LLMDataDist, LLMRole, LLMConfig, CacheDesc, CacheKey, DataType, LLMClusterInfo, LLMStatusCode

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
    cache_desc_1bs = CacheDesc(num_tensors=4, shape=[1, 4, 8], data_type=DataType.DT_FLOAT16)
    cache_desc_4bs = CacheDesc(num_tensors=4, shape=[4, 4, 8], data_type=DataType.DT_FLOAT16)
    kv_cache_1bs = kv_cache_manager.allocate_cache(cache_desc_1bs)
    kv_cache_4bs = kv_cache_manager.allocate_cache(cache_desc_4bs)
    logging.info('[allocate_cache] success')
    cache_key_0 = CacheKey(prompt_cluster_id=0, req_id=0, model_id=0)
    cache_key_1 = CacheKey(prompt_cluster_id=0, req_id=1, model_id=0)
    # 可以指定拉取KV到指定batch_index, 也可以指定拉取的tensor的大小
    kv_cache_manager.pull_cache(cache_key_1, kv_cache_4bs, 1, 64)
    kv_cache_manager.pull_cache(cache_key_0, kv_cache_1bs)
    logging.info('[pull_cache] success')
    kv_cache_manager.copy_cache(kv_cache_4bs, kv_cache_1bs, 0, 0)
    logging.info('[copy_cache] success')
    kv_host_tensors = kv_cache_manager.get_cache_tensors(kv_cache_4bs, 0)
    logging.info('[get_cache_tensor] success')
    logging.info(f'batch_index = 0, tensor = {kv_host_tensors[0].numpy()[0, :]}')
    logging.info(f'batch_index = 1, tensor = {kv_host_tensors[0].numpy()[1, :]}')
    kv_cache_manager.deallocate_cache(kv_cache_1bs)
    kv_cache_manager.deallocate_cache(kv_cache_4bs)
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

