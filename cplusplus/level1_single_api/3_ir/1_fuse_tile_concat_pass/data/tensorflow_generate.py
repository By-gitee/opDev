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
from tensorflow.compat.v1.graph_util import convert_variables_to_constants
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys

model_root_path = './'

def create_model(argv):
    test_name = "tile_concat"  # 用例名称
    tf.compat.v1.disable_eager_execution()
    input1 = tf.compat.v1.placeholder(tf.int32, shape=(2, 3,), name='input1')
    input2 = tf.compat.v1.placeholder(tf.int32, shape=(2, 3,), name='input2')
    a = np.array([1, 2])
    const1 = tf.constant(a, dtype=tf.int32, shape=(2, ), name="const1")
    tile1 = tf.tile(input1, const1)
    tile2 = tf.tile(input2, const1)
    const2 = tf.constant(0, dtype=tf.int32, name="const2")
    output = tf.concat([tile1, tile2], const2, name="concatV2")

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        graph = convert_variables_to_constants(sess, sess.graph_def, ["concatV2"])
        
        tf.io.write_graph(graph, '.', model_root_path + test_name + '.pb', as_text=False)
        print('Create Model Successful.')
        print('Path: ', model_root_path + test_name + '.pb')
    tf.compat.v1.reset_default_graph()

if __name__=='__main__':
    create_model(sys.argv)
