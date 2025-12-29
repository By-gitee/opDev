import torch
import torch_npu
import torchair
from torchair.configs.compiler_config import CompilerConfig
from torchair.core.utils import logger
import logging

logger.setLevel(logging.DEBUG)
config = CompilerConfig()
config.debug.graph_dump.type = "pbtxt"
config.experimental_config.tiling_schedule_optimize = True
npu_backend = torchair.get_npu_backend(compiler_config=config)

import torchair.ops.add_custom_tiling_sink

class MyModule(torch.nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()

    def forward(self, x, y):
        z = torch.ops.air.add_custom_tiling_sink.default(x, y)
        return z


# 创建并编译模块
module = MyModule().npu()
module = torch.compile(module, fullgraph=True, backend=npu_backend, dynamic=False)

# 示例输入
x = torch.randn(6, 64, dtype=torch.float32).npu()
y = torch.randn(6, 64, dtype=torch.float32).npu()

output = module(x, y)
print(output.shape)