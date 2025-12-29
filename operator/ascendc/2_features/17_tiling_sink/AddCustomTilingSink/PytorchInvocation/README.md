## 目录结构介绍

```
├── PytorchInvocation       // torch注册的自定义算子
│   ├── src
│   │   ├── add_custom_tiling_sink.py   // 自定义算子py文件
│   └── test_add_custom_tiling_sink.py  // 测试脚本
```

## 代码实现

src/add_custom_tiling_sink.py是调用自定义算子的torch脚本，如何开发该脚本代码，具体步骤如下。
> 注意：如需详细了解入图操作，请参考Ascend torchair仓中[converter补齐](https://gitee.com/ascend/torchair/blob/master/CONTRIBUTING.md#converter%E8%A1%A5%E9%BD%90)章节。 

1.下载[torchair工程源码](https://gitee.com/ascend/torchair)，并在torchair/python/torchair/ops目录下新建add_custom_tiling_sink.py空文件。  
> 注意，请根据实际情况下载配套版本分支的torchair工程源码，版本配套关系请查看[PyTorch框架适配官网](https://www.hiascend.com/software/ai-frameworks/pytorch)。

2.将自定义算子注册到PyTorch框架。
```python
# add_custom_tiling_sink.py
import torch

lib = torch.library.Library("air", "FRAGMENT")
lib.define(
    """
    add_custom_tiling_sink(Tensor x, Tensor y) -> Tensor
    """
)
```
3.实现自定义算子的单算子模式。  
该部分目前仅为示例，当前预留为为实现，请用户根据实际需要自行定义。
```python
def kernel_impl(x, y):
    raise NotImplementedError("torch.ops.air.add_custom_tiling_sink kernel_impl is not implemented!")

torch.library.impl(lib, "add_custom_tiling_sink", "CPU")(kernel_impl)
torch.library.impl(lib, "add_custom_tiling_sink", "PrivateUse1")(kernel_impl)
```

4.为自定义算子注册Meta函数，通过PyTorch Meta后端完成入图时所需要的shape和data type推导。
```python
@torch.library.impl(lib, "add_custom_tiling_sink", "Meta")
def kernel_meta(x, y):
    return torch.empty_like(x)
```

5.codegen生成ge构图api  
（1）将REG_OP算子原型放置到codegen/custom_op/custom_reg_op.h文件中，替换原来示例的REG_OP

```cpp
#ifndef ASCENDADAPTER2_CUSTOM_REG_OP_H
#define ASCENDADAPTER2_CUSTOM_REG_OP_H
#include "graph/operator_reg.h"

namespace ge {
REG_OP(AddCustomTilingSink)
   .INPUT(x, TensorType::ALL())
   .INPUT(y, TensorType::ALL())
   .OUTPUT(z, TensorType::ALL())
   .OP_END_FACTORY_REG(AddCustomTilingSink)
}

#endif  // ASCENDADAPTER2_CUSTOM_REG_OP_H
```

（2）进入torchair工程源码根目录执行编译命令，产物在codegen/custom_op/auto_generated_ge_raw_custom_ops.py目录。

```
cd build
cmake ..
make generate_ge_raw_custom_ops
```

生成的ge.api函数内容如下所示：

```python
# This file is auto-generated
# Summary: total 1, generated 1, skipped 0
from typing import Any, Dict, List, Tuple, Union, Callable, Optional
from torchair.ge._ge_graph import auto_convert_to_tensor, TensorType
from torchair.ge import Tensor, DataType, attr
from torchair._ge_concrete_graph.ge_converter import ge_op, IrDef


# This api is auto-generated from IR AddCustomTilingSink
@auto_convert_to_tensor([False, False], [False, False], inputs_tensor_type=[TensorType.TT_ALL, TensorType.TT_ALL])
def AddCustomTilingSink(x: Tensor, y: Tensor, *, dependencies=[], node_name=None):
    """REG_OP(AddCustomTilingSink)\n
.INPUT(x, TensorType::ALL())\n
.INPUT(y, TensorType::ALL())\n
.OUTPUT(z, TensorType::ALL())\n
"""

    # process inputs
    inputs = {
        "x": x,
        "y": y,
    }

    # process attrs
    attrs = {
    }

    # process outputs
    outputs = [
    "z",
    ]

    return ge_op(
        op_type="AddCustomTilingSink",
        inputs=inputs,
        attrs=attrs,
        outputs=outputs,
        dependencies=dependencies,
        ir=IrDef("AddCustomTilingSink") \
        .input("x", "") \
        .input("y", "") \
        .output("z" , "")
    )
```

需要修改`from torchair._ge_concrete_graph.ge_converter import ge_op, IrDef`
为`from torchair._ge_concrete_graph.compat_ir import ge_op, IrDef`

将上述生成内容拷贝至前面我们新建的add_custom_tiling_sink.py文件中。

6.实现自定算子converetr并注册：

```python
from typing import (
    Optional,
    Union,
    List,
)
from torchair._ge_concrete_graph.fx2ge_converter import register_fx_node_ge_converter
from torchair.ge._ge_graph import Tensor, TensorSpec

@register_fx_node_ge_converter(torch.ops.air.add_custom_tiling_sink.default)
def convert_add_custom_tiling_sink(x: torch.Tensor, y: torch.Tensor, meta_outputs: Union[TensorSpec, List[TensorSpec]] = None):
    return AddCustomTilingSink(x, y) # 此为前面生成的构图api
```

## 运行样例算子

### 编译安装torchair包

1.编译，进入torchair根目录，执行：

```
bash build.sh -c
```

2.安装，进入torchair根目录，执行注意pip3.x为对应Python版本：

```
pip3.x uninstall torchair
pip3.x install output/torchair_xxxx.whl
```

3.删除环境上torch_npu模块下的torchair子模块，使得我们安装的torchair模块生效：

```
rm -rf /usr/local/python3.8.1/lib/python3.8/site-packages/torch_npu/dynamo/torchair
```

查看环境上安装的torch_npu的路径：

```
pip3.x show torch_npu
```

### 编译部署自定义算子包
请参考[AddCustomTilingSink自定义算子实现](../AddCustomTilingSink/README.md)。

### 执行脚本
test_add_custom_tiling_sink.py是图模式调用算子tiling下沉测试脚本，请根据实际情况替换里面的模型定义、参数等内容。  
该脚本有2个关键操作必须确保完成，具体如下：  
1.测试脚本必须import自定义的add_custom_tiling_sink.py模块。
```python
import torchair.ops.add_custom_tiling_sink

def forward(self, x, y):
    z = torch.ops.air.add_custom_tiling_sink.default(x, y)
    return z
```

2.测试脚本显式开启tiling_schedule_optimize配置项。
```python
from torchair.configs.compiler_config import CompilerConfig

config = CompilerConfig()
config.experimental_config.tiling_schedule_optimize = True
```

## 更新说明

| 时间      | 更新事项     |
| --------- | ------------ |
| 2025/5/22 | 新增本readme |