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
import logging
import pickle
import time
import socket
from typing import Optional, List
from llm_datadist import LLMDataDist, LLMRole, LLMConfig, CacheDesc, KvCache, DataType, LayerSynchronizer, \
    LLMClusterInfo, LLMStatusCode, TransferConfig
import torch
import torch_npu
import torchair

PROMPT_CLUSTER_ID = 0
DECODER_CLUSTER_ID = 1

# 用于被拉取KV的ip与端口
DEVICE_IP_LIST = ['192.168.2.1', '192.168.2.2', '192.168.2.3', '192.168.2.4',
                  '192.168.2.5', '192.168.2.6', '192.168.2.7', '192.168.2.8']
DEVICE_LISTEN_PORT = 26000
DEVICE_ID_LIST = [0, 1, 2, 3, 4, 5, 6, 7]
PROMPT_HOST_IP = '127.0.0.1'
HOST_LISTEN_PORT_BASE = 27000

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


class LayerSynchronizerImpl(LayerSynchronizer):
    def __init__(self, events):
        self._events = events

    def synchronize_layer(self, layer_index: int, timeout_in_millis: Optional[int]) -> bool:
        self._events[layer_index].wait()
        return True


def init_llm_datadist(rank_id: int) -> LLMDataDist:
    datadist = LLMDataDist(LLMRole.PROMPT, PROMPT_CLUSTER_ID)
    llm_config = LLMConfig()
    listen_ip_info = DEVICE_IP_LIST[rank_id] + ":" + str(DEVICE_LISTEN_PORT)  # ex. 192.168.1.1:26000
    llm_config.listen_ip_info = listen_ip_info
    llm_config.device_id = DEVICE_ID_LIST[rank_id]
    llm_options = llm_config.generate_options()
    datadist.init(llm_options)
    return datadist


def run_with_kv_cache(kv_cache: KvCache, events):
    kv_tensor_addrs = kv_cache.per_device_tensor_addrs[0]
    logging.info(f'cache_id={kv_cache.cache_id}, kv_tensor_addrs={kv_tensor_addrs}')
    # 使用kv_tensor_addr 构造对应前端框架(如torch)的Tensor
    kv_tensors = torchair.llm_datadist.create_npu_tensors(kv_cache.cache_desc.shape, torch.float16, kv_tensor_addrs)
    # 对kv cache进行赋值，传输完成后可验证结果
    y = torch.tensor([i for i in range(4 * 4 * 8)], dtype=torch.float16).reshape(kv_cache.cache_desc.shape).npu()
    for i in range(kv_cache.cache_desc.num_tensors):
        kv_tensors[i].fill_(i * 10000)
        kv_tensors[i].add_(y)
        if i % 2 == 0:
            events[i // 2].record()


def recv_kv_addrs(rank_id: int):
    sock = socket.socket(family=socket.AF_INET, type=socket.SOCK_STREAM)
    sock.bind((PROMPT_HOST_IP, HOST_LISTEN_PORT_BASE + rank_id))
    sock.listen(5)
    conn, _ = sock.accept()
    req_data = conn.recv(2048)
    kv_addresses = pickle.loads(req_data)
    conn.close()
    return kv_addresses


def run_prompt_sample(rank_id: int):
    # 1. 初始化llm_datadist
    datadist = init_llm_datadist(rank_id)
    logging.info('[initialize] llm_datadist success')
    # 2. 通过kv_cache_manager分配kv cache
    kv_cache_manager = datadist.kv_cache_manager
    # 描述一个kv cache，管理4个tensor, 每个tensor的batch_size=4, dtype为FP16
    cache_desc = CacheDesc(num_tensors=4, shape=[4, 4, 8], data_type=DataType.DT_FLOAT16)
    kv_cache = kv_cache_manager.allocate_cache(cache_desc)
    logging.info('[allocate_cache] success')

    # 获取decoder kv_cache内存地址，此处使用socket简单处理，用户需根据实际场景实现地址的传输
    kv_cache_addrs = recv_kv_addrs(rank_id)
    logging.info('[recv_kv_addr] success')

    # 操作kv cache，同时用event记录各层执行结束状态，用户可以改写该方法使用kv cache进行推理
    events = [torch.npu.Event() for _ in range(cache_desc.num_tensors // 2)]
    run_with_kv_cache(kv_cache, events)
    logging.info('[run_with_kv_cache] success')

    # 以连续内存的方式传输
    transfer_config = TransferConfig(DECODER_CLUSTER_ID, kv_cache_addrs[0], range(0, 2))
    cache_task = kv_cache_manager.transfer_cache_async(kv_cache, LayerSynchronizerImpl(events), [transfer_config])
    logging.info('[transfer_cache_async] success')
    # 传输是异步的，传输过程中可以做其他操作，此处省略
    # 同步并获取传输结果
    ret = cache_task.synchronize()
    if ret != LLMStatusCode.LLM_SUCCESS:
        ret_per_transfer_config = cache_task.get_results()
        raise RuntimeError(
            f'[transfer_cache_async] failed, ret={ret}, ret_per_transfer_config = {ret_per_transfer_config}')
    logging.info('transfer cache success')
    # 以blocks的方式传输
    transfer_config = TransferConfig(DECODER_CLUSTER_ID, kv_cache_addrs[1], range(0, 2))
    cache_task = kv_cache_manager.transfer_cache_async(kv_cache, LayerSynchronizerImpl(events),
                                                       [transfer_config],
                                                       src_block_indices=[0, 1, 2, 3],
                                                       dst_block_indices=[3, 2, 1, 0])
    logging.info('[transfer_cache_async] success')
    ret = cache_task.synchronize()
    if ret != LLMStatusCode.LLM_SUCCESS:
        ret_per_transfer_config = cache_task.get_results()
        raise RuntimeError(
            f'[transfer_cache_async] failed, ret={ret}, ret_per_transfer_config = {ret_per_transfer_config}')
    logging.info('transfer blocks cache success')

    kv_cache_manager.deallocate_cache(kv_cache)
    logging.info('[deallocate_cache] success')
    datadist.finalize()
    logging.info('[finalize] success')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank_id", type=int, default=0, help='rank_id')
    args = parser.parse_args()
    logging.info(f'Sample start, rank_id = {args.rank_id}, device_id = {DEVICE_ID_LIST[args.rank_id]}')
    torch.npu.set_device(DEVICE_ID_LIST[args.rank_id])
    run_prompt_sample(args.rank_id)
    logging.info('Sample end')
