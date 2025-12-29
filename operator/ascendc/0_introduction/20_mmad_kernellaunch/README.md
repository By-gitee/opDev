## 概述
本样例介绍基于基础API的matmul算子实现及核函数直调方法。

## 目录结构介绍
```
└── 20_mmad_kernellaunch      // 使用核函数直调的方式调用Matmul自定义算子。
    └── MmadInvocation        // Kernel Launch方式调用核函数样例。
    └── MmadBiasInvocation    // Kernel Launch方式调用核函数样例，新增bias输入。
```

## 算子描述
算子使用基础API包括DataCopy、LoadData、Mmad等，实现Matmul矩阵乘功能。

Matmul的计算公式为：

```
C = A * B + Bias
```

- A、B为源操作数，A为左矩阵，形状为\[M, K]；B为右矩阵，形状为\[K, N]。
- C为目的操作数，存放矩阵乘结果的矩阵，形状为\[M, N]。
- Bias为矩阵乘偏置，形状为\[N]。对A*B结果矩阵的每一行都采用该Bias进行偏置。

## 支持的产品型号
本样例支持如下产品型号：
- Atlas 推理系列产品
- Atlas A2训练系列产品/Atlas 800I A2推理产品

## 编译运行样例算子

### 1. 获取源码包
编译运行此样例前，请参考[准备：获取样例代码](../README.md#codeready)获取源码包。

### 2. 编译运行样例工程
- [MmadBiasInvocation样例运行](./MmadBiasInvocation/README.md)
- [MmadInvocation样例运行](./MmadInvocation/README.md)

## 更新说明
| 时间       | 更新事项                 |
| ---------- | ------------------------ |
| 2024/11/20 | 新增readme |

## 已知issue

  暂无
