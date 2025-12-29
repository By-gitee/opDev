## 概述
本样例介绍无DataCopyPad的非对齐ReduceMin算子核函数直调方法。
## 目录结构介绍
```
├── 17——unaligned_reducemin_kernellaunch // 使用核函数直调的方式调用非对齐ReduceMin自定义算子
|   └── ReduceMinKernelInvocation        // Kernel Launch方式调用非对齐轴规约ReduceMin核函数样例
```

## 算子描述
ReduceMin算子的numpy表达式为：
```
z = np.min(x, axis=1)
```

非对齐ReduceMin算子的功能是计算非对齐输入数据的绝对值。

## 算子规格描述

<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">ReduceMin</td></tr>
</tr>
<tr><td rowspan="2" align="center">算子输入</td>
<td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x</td><td align="center">16*4</td><td align="center">float16</td><td align="center">ND</td></tr>
</tr>
</tr>
<tr><td rowspan="1" align="center">算子输出</td><td align="center">z</td><td align="center">16*4</td><td align="center">float16</td><td align="center">ND</td></tr>
</tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">reduce_min_custom</td></tr>
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
- [ReduceMinKernelInvocation样例运行](./ReduceMinKernelInvocation/README.md)
## 更新说明
| 时间| 更新事项|
| - | - |
| 2024/09/20 | 新增本readme |
| 2024/11/16 | 样例目录调整 |
