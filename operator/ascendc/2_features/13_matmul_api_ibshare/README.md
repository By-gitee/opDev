## 概述
本样例介绍Matmul IBShare特性样例，实现MatmulABshare算子及其核函数直调方法。

## 目录结构介绍
```
├── 13_matmul_api_ibshare
│   └── MatmulABshareInvocation                          // 使用核函数直调的方式调用MatmulABshare自定义算子。
```

## 算子描述
  算子名中ABshare的含义为，Matmul计算的A矩阵与B矩阵同时使能类型信息MatmulType的参数IBShare；当A矩阵和B矩阵同时使能IBShare时，表示L1 Buffer上的A矩阵和B矩阵同时复用；IBShare的详细介绍请参考[Ascend C算子开发接口](https://www.hiascend.com/document/redirect/CANNCommunityAscendCApi)>高阶API>Matmul>Matmul>使用说明 章节。  
  本样例中包含两个算子的核函数实现，分别为使能ABshare的[MatmulABshare](./MatmulABshareInvocation/matmul_ABshare_custom.cpp)算子和未使能ABshare的[MatmulNoABshare](./MatmulABshareInvocation/matmul_noABshare_custom.cpp)算子。MatmulNoABshare算子的A矩阵与B矩阵均未使能IBSHARE，数据按照K列进行切分计算。MatmulABshare算子的A矩阵与B矩阵均使能IBShare，不对k列进行切分计算，实现了算子性能提升。通过对比两个算子的运行时间，计算MatmulABshare算子的性能提升百分比。


对应的数学表达式为：
```
C = A * B
```
- A、B为源操作数，A为左矩阵，形状为\[M, K]；B为右矩阵，形状为\[K, N]。
- C为目的操作数，存放矩阵乘结果的矩阵，形状为\[M, N]。

## 算子规格描述
在核函数直调样例中，算子实现支持的shape为：M = 128, N = 256, K = 384。
<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Matmul</td></tr>
</tr>
<tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">a</td><td align="center">M * K</td><td align="center">float16</td><td align="center">ND</td></tr>
<tr><td align="center">b</td><td align="center">K * N</td><td align="center">float16</td><td align="center">ND</td></tr>
</tr>
</tr>
<tr><td rowspan="1" align="center">算子输出</td><td align="center">c</td><td align="center">M * N</td><td align="center">float</td><td align="center">ND</td></tr>
</tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">matmul_ABshare_custom</td></tr>
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
- [MatmulABshareInvocationNeo样例运行](./MatmulABshareInvocation/README.md)

## 更新说明
| 时间       | 更新事项                 |
| ---------- | ------------------------ |
| 2024/11/28 | 挪动目录               |
| 2024/11/12 | 新增readme               |
