## 概述
本样例介绍MatmulLeakyRelu算子实现及核函数直调方法。

## 目录结构介绍
```
└── 22_baremix_kernellaunch      // 使用核函数直调的方式调用MatmulLeakyRelu自定义算子。
    └── BareMixInvocation        // Kernel Launch方式调用核函数样例。
```

## 算子描述
算子使用了Matmul高阶API，实现了快速的MatmulLeakyRelu矩阵乘法的运算操作。

MatmulLeakyRelu的计算公式为：

```
C = A * B + Bias
C = C > 0 ? C : C * 2.0
```

- A、B为源操作数，A为左矩阵，形状为\[M, K]；B为右矩阵，形状为\[K, N]。
- C为目的操作数，存放矩阵乘结果的矩阵，形状为\[M, N]。
- Bias为矩阵乘偏置，形状为\[N]。对A*B结果矩阵的每一行都采用该Bias进行偏置。

## 支持的产品型号
本样例支持如下产品型号：
- Atlas A2训练系列产品/Atlas 800I A2推理产品

## 编译运行样例算子

### 1. 获取源码包
编译运行此样例前，请参考[准备：获取样例代码](../README.md#codeready)获取源码包。

### 2. 编译运行样例工程
- [BareMixInvocation样例运行](./BareMixInvocation/README.md)

## 更新说明
| 时间       | 更新事项                 |
| ---------- | ------------------------ |
| 2025/7/28 | 新增22_baremix_kernellaunch |

## 已知issue

  暂无
