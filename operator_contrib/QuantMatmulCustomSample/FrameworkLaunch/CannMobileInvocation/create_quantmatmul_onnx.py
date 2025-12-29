import torch
import torch.backends.cuda
import torch.nn as nn
import numpy as np

class QuantMatmul_(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input1, input2, scale_fp16, zero_fp16, weight_Size_list): # ctx require
        input = input1
        return input

    @staticmethod
    def symbolic(g:torch.Graph,
                 input0: torch.Tensor,
                 weight_U2: torch.Tensor,
                 scale_fp16: torch.Tensor,
                 zero_fp16: torch.Tensor,
                 weight_Size_list):

        return g.op("QuantMatmulCustom", input0, weight_U2, scale_fp16, zero_fp16, input1_shape_i=weight_Size_list)

QuantMatmul_ = QuantMatmul_.apply

class TinyNet(nn.Module):
    def __init__(self):
        super(TinyNet, self).__init__()

    def forward(self, x):
        weight_size_origin = [1280, 3840]
        x = QuantMatmul_(
            x,
            weight_U2,
            scale_fp16,
            zero_fp16,
            weight_size_origin
        )
        return x

net = TinyNet()
input0 = torch.randn((32, 1280), dtype=torch.float32)
weight_U2 = np.random.randint(0, 256, (1228800), dtype=np.uint8)
weight_U2 = torch.tensor(weight_U2, dtype=torch.uint8)
scale_fp16 = np.random.rand(153600).astype(np.float16)
scale_fp16 = torch.tensor(scale_fp16, dtype=torch.float16)
zero_fp16 = np.random.rand(153600).astype(np.float16)
zero_fp16 = torch.tensor(zero_fp16, dtype=torch.float16)
torch.onnx.export(net, (input0), './QuantMatMulCustom.onnx', opset_version = 11)