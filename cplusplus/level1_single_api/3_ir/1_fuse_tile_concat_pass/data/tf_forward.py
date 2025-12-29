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

import tensorflow as tf
import numpy as np
import npu_device
from npu_device.compat.v1.npu_init import *
npu_device.compat.enable_v1()
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def create_graph():
    tf.compat.v1.disable_eager_execution()
    input1 = tf.compat.v1.placeholder(tf.int32, shape=(2, 3,), name='input1')
    input2 = tf.compat.v1.placeholder(tf.int32, shape=(2, 3,), name='input2')
    a = np.array([1, 2])
    const1 = tf.constant(a, dtype=tf.int32, shape=(2, ), name="const1")
    tile1 = tf.tile(input1, const1)
    tile2 = tf.tile(input2, const1)
    const2 = tf.constant(0, dtype=tf.int32, name="const2")
    output = tf.concat([tile1, tile2], const2, name="concatV2")
    return tf.compat.v1.get_default_graph()

def NetworkRun():
    graph = create_graph()
    input_nodes1 = graph.get_tensor_by_name('input1:0')
    input_nodes2 = graph.get_tensor_by_name('input2:0')
    output_nodes = graph.get_tensor_by_name('concatV2:0')
    input_tensor1 = np.array([[1, 2, 3], [4, 5, 6]])
    input_tensor2 = np.array([[7, 8, 9], [10, 11, 12]])
    # 适配npu
    config_proto = tf.compat.v1.ConfigProto()
    custom_op = config_proto.graph_options.rewrite_options.custom_optimizers.add()
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("allow_mix_precision")
    config_proto.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    tf_config = npu_config_proto(config_proto=config_proto)
    with tf.compat.v1.Session(config=tf_config, graph=graph) as sess:
        out = sess.run(output_nodes, feed_dict={input_nodes1:input_tensor1, input_nodes2:input_tensor2})
        print('---out---\n', out)

if __name__=='__main__':
    NetworkRun()
