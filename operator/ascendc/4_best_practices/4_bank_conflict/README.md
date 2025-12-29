## 概述

本样例介绍基于Add算子优化bank冲突的实现，并提供核函数直调方法。

## 目录结构介绍

```
├── 4_bank_conflict      // 使用核函数直调的方式调用Add自定义算子
│   └── KernelLaunch     // Kernel Launch方式调用核函数样例
```

## 算子描述

算子实现的是固定shape为1×4096的Add算子。

Add的计算公式为：

```python
z = x + y
```

- x：输入，形状为\[1, 4096]，数据类型为float；
- y：输入，形状为\[1, 4096]，数据类型为float；
- z：输出，形状为\[1, 4096]，数据类型为float；

## 算子规格描述

<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Add</td></tr>
</tr>
<tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x</td><td align="center">1 * 4096</td><td align="center">float</td><td align="center">ND</td></tr>
<tr><td align="center">y</td><td align="center">1 * 4096</td><td align="center">float</td><td align="center">ND</td></tr>
</tr>
</tr>
<tr><td rowspan="1" align="center">算子输出</td><td align="center">y</td><td align="center">1 * 4096</td><td align="center">float</td><td align="center">ND</td></tr>
</tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">add_custom_v1 / add_custom_v2</td></tr>
</table>

## 支持的产品型号

本样例支持如下产品型号：

- Atlas A2训练系列产品/Atlas 800I A2推理产品

## 编译运行样例算子

针对自定义算子工程，编译运行包含如下步骤：

- 编译自定义算子工程；
- 调用执行自定义算子；

详细操作如下所示。

### 1. 获取源码包

编译运行此样例前，请参考[准备：获取样例代码](../README.md#codeready)获取源码包。

### 2. 编译运行样例工程

- [KernelLaunch样例运行](./KernelLaunch/README.md)

## 更新说明


| 时间       | 更新事项         |
| ---------- | ---------------- |
| 2025/07/01 | 新增直调方式样例 |
