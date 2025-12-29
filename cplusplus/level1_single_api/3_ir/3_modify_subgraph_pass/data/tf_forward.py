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
import npu_bridge
tf.compat.v1.enable_control_flow_v2()
tf.compat.v1.enable_resource_variables()
from tensorflow.core.protobuf.rewriter_config_pb2 import RewriterConfig
import numpy as np

def generate_tf_graph():
    x = tf.compat.v1.placeholder(tf.float32, shape=[], name="X")
    y = tf.compat.v1.placeholder(tf.float32, shape=[], name="Y")

    const1 = tf.constant([-1, 1], dtype=tf.float32, name='Const1')
    const2 = tf.constant([-2, 2], dtype=tf.float32, name='Const2')
    const3 = tf.constant([3, 3], dtype=tf.float32, name='Const3')
    case_cond = tf.equal(x=x, y=y)
    def true_fn_branch(input_data1):
        output = input_data1
        print("-------------true_fn_branch-----------%s--" % output)
        return output

    def false_fn_branch(input_data1):
        output = input_data1
        print("-------------false_fn_branch-----------%s--" % output)
        return output
    cond = tf.cond(case_cond, true_fn=lambda: true_fn_branch(const1), false_fn=lambda: false_fn_branch(const1))
    output = tf.add(const3, cond, name='Add')

    return tf.compat.v1.get_default_graph()

def NetworkRun():
    graph = generate_tf_graph()
    input_x = graph.get_tensor_by_name('X:0')
    input_y = graph.get_tensor_by_name('Y:0')
    output_nodes = graph.get_tensor_by_name('Add:0')
    x = np.array(1)
    y = np.array(2)
    
    # 适配npu
    config = tf.compat.v1.ConfigProto()
    custom_op = config.graph_options.rewrite_options.custom_optimizers.add()
    # 配置1：选择在昇腾AI处理器上执行推理
    custom_op.name = "NpuOptimizer"
    custom_op.parameter_map["use_off_line"].b = True

    # 配置2：在线推理场景下建议保持默认值force_fp16，使用float16精度推理，以获得较优的性能
    custom_op.parameter_map["precision_mode"].s = tf.compat.as_bytes("force_fp16")

    # 配置3：图执行模式，推理场景下请配置为0，训练场景下为默认1
    custom_op.parameter_map["graph_run_mode"].i = 0

    # 配置4：关闭remapping和MemoryOptimizer
    config.graph_options.rewrite_options.remapping = RewriterConfig.OFF
    config.graph_options.rewrite_options.memory_optimization = RewriterConfig.OFF

    with tf.compat.v1.Session(config=config, graph=graph) as sess:
        out = sess.run(output_nodes, feed_dict={input_x:x, input_y:y})
        print('---out---\n', out)

if __name__=='__main__':
    NetworkRun()
