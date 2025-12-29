## 概述

本样例介绍QuantGroupMatmul算子的高性能实现，并提供核函数直调方法。

## 目录结构介绍

```
├── 6_group_matmul      // 使用核函数直调的方式调用QuantGroupMatmul自定义算子
│   └── KernelLaunch    // Kernel Launch方式调用核函数样例
```

## 算子描述

算子实现了分组的pertoken量化matmul计算，分组轴为m轴，并对结果进行激活函数Gelu计算。

QuantGroupMatmul的计算公式为：

```python
offset = 0
for i in range(g):
    mmOut = x[offset:offset + group[i]] * weight[i] + bias[i]  # Cube计算
    y[offset:offset + group[i]] = Gelu(mmOut * scale[i] * pertokenScale[offset:offset + group[i]])  # vector计算
    offset += group[i]
```

- x：左矩阵，形状为\[m, k]，数据类型为int8；
- weight：右矩阵，形状为\[g, k, n]，数据类型为int8；
- bias：矩阵乘偏置，形状为\[g, n]，数据类型为int32，对第i次矩阵乘结果的每一行都采用bias[i]进行偏置；
- group：记录每组m的大小，数据类型为int64；
- scale：右矩阵的量化参数，形状为\[g, n]，数据类型为float，用于矩阵乘结果的反量化，对第i次矩阵乘结果采用scale[i]进行反量化；
- pertokenScale：左矩阵的量化参数，形状为\[m]，数据类型为float，用于矩阵乘结果的反量化，采用与x行相同的索引范围进行反量化；
- y：输出，存放矩阵乘结果的矩阵，形状为\[m, n]，数据类型为float16；

## 算子规格描述

<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">QuantGroupMatmul</td></tr>
</tr>
<tr><td rowspan="7" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x</td><td align="center">1024 * 1024</td><td align="center">int8</td><td align="center">ND</td></tr>
<tr><td align="center">weight</td><td align="center">8 * 1024 * 8192</td><td align="center">int8</td><td align="center">NZ</td></tr>
<tr><td align="center">bias</td><td align="center">8 * 8192</td><td align="center">int32</td><td align="center">ND</td></tr>
<tr><td align="center">groupList</td><td align="center">8</td><td align="center">uint64</td><td align="center">ND</td></tr>
<tr><td align="center">scale</td><td align="center">8 * 8192</td><td align="center">float</td><td align="center">ND</td></tr>
<tr><td align="center">pretokenScale</td><td align="center">1024</td><td align="center">float</td><td align="center">ND</td></tr>
</tr>
</tr>
<tr><td rowspan="1" align="center">算子输出</td><td align="center">y</td><td align="center">1024 * 8192</td><td align="center">float16</td><td align="center">ND</td></tr>
</tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">quant_group_matmul_custom</td></tr>
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
| 时间       | 更新事项     |
| ---------- | ------------ |
| 2024/11/20 | 新增直调方式样例 |
