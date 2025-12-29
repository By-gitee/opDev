## 概述
本样例介绍Add算子的核函数直调方法。
## 目录结构介绍
```
├── 3_add_kernellaunch                // 使用核函数直调的方式调用Add自定义算子
│   ├── AddKernelInvocationAcl        // 使用aclrtLaunchKernelWithConfig接口调用核函数样例
│   ├── AddKernelInvocationNeo        // Kernel Launch方式调用核函数样例
│   ├── AddKernelInvocationTilingNeo  // Kernel Launch方式调用核函数样例，带有Tiling
│   └── CppExtensions                 // pybind方式调用核函数样例，带有Tiling
```

## 算子描述
Add算子实现了两个数据相加，返回相加结果的功能。对应的数学表达式为：  
```
z = x + y
```
## 算子规格描述
<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Add</td></tr>
</tr>
<tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
<tr><td align="center">y</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
</tr>
</tr>
<tr><td rowspan="1" align="center">算子输出</td><td align="center">z</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
</tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">add_custom</td></tr>
</table>

## 支持的产品型号
本样例支持如下产品型号：
- Atlas 训练系列产品
- Atlas 推理系列产品AI Core
- Atlas A2训练系列产品/Atlas 800I A2推理产品
- Atlas 200/500 A2推理产品

## 编译运行样例算子
针对自定义算子工程，编译运行包含如下步骤：
- 编译自定义算子工程；
- 调用执行自定义算子；

详细操作如下所示。
### 1. 获取源码包
编译运行此样例前，请参考[准备：获取样例代码](../README.md#codeready)获取源码包。
### 2. 编译运行样例工程
- [AddKernelInvocationAcl样例运行](./AddKernelInvocationAcl/README.md)
- [AddKernelInvocationNeo样例运行](./AddKernelInvocationNeo/README.md)
- [AddKernelInvocationTilingNeo样例运行](./AddKernelInvocationTilingNeo/README.md)
- [CppExtensions样例运行](./CppExtensions/README.md)
## 更新说明
| 时间       | 更新事项                                            | 注意事项                                       |
| ---------- | --------------------------------------------------- | ---------------------------------------------- |
| 2023/10/09 | 新增AddCustomSample/KernelLaunch样例                |                                                |
| 2024/01/04 | 新增AddKernelInvocationNeo样例                      | 需要基于社区CANN包7.0.0.alpha003及之后版本运行 |
| 2024/01/04 | 新增AddKernelInvocationTilingNeo样例                | 需要基于社区CANN包7.0.0.alpha003及之后版本运行 |
| 2024/02/02 | 新增AddCustomSample/KernelLaunch/CppExtensions样例  | 需要基于社区CANN包7.0.0.alpha003及之后版本运行 |
| 2024/05/22 | 更新readme结构                                      | 需要基于社区CANN包7.0.0.alpha003及之后版本运行 |
| 2024/06/06 | AddKernelInvocation样例转维护，不再更新，不推荐使用 |
| 2024/08/11 | 删除AddKernelInvocation样例 |
| 2024/11/11 | 样例目录调整 |   |
| 2025/06/05 | 新增AddKernelInvocationAcl样例 |   |
