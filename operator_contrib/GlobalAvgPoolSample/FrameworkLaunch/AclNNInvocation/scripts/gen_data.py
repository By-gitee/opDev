import os
import onnx
from onnx import helper
from onnx import TensorProto
import onnxruntime as ort
import numpy as np


def gen_onnx_model(shape_x, shape_y):
    # x = helper.make_tensor_value_info("x", TensorProto.FLOAT16, shape_x)
    # y = helper.make_tensor_value_info("y", TensorProto.FLOAT16, shape_y)

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, shape_x)
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, shape_y)
    node_def = helper.make_node('GlobalAveragePool',
                                inputs=['x'],
                                outputs=['y']
                                )
    graph = helper.make_graph(
        [node_def],
        "test_GlobalAveragePool_case_1",
        [x],
        [y]
    )

    model = helper.make_model(graph, producer_name="onnx-GlobalAveragePool_test")
    model.opset_import[0].version = 11
    onnx.save(model, "./test_GlobalAveragePool_v11.onnx")


def run_mode(x):
    # 加载ONNX模型
    model_path = 'test_GlobalAveragePool_v11.onnx'  # 替换为你的ONNX模型路径
    sess = ort.InferenceSession(model_path)

    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    input_data = x

    outputs = sess.run([output_name], {input_name: input_data})
    return outputs[0]


def gen_golden_data_simple():
    shape_x = [200, 20, 1]
    shape_y = [200, 20]
    # input_x = np.random.uniform(-10, 10, shape_x).astype(np.float16)
    input_x = np.random.uniform(-10, 10, shape_x).astype(np.float32)

    print(input_x)

    gen_onnx_model(shape_x, shape_y)
    golden = run_mode(input_x)
    os.system("mkdir -p input")
    os.system("mkdir -p output")
    input_x.tofile("./input/input_x.bin")
    golden.tofile("./output/golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple()
