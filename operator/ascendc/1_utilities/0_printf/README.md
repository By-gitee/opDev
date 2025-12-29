## printf样例说明
本样例通过Ascend C编程语言实现了Matmul算子，同时在算子中添加printf调测，并按照不同的算子调用方式分别给出了对应的端到端实现。
- [FrameworkLaunch](./FrameworkLaunch)：使用框架调用Matmul算子及printf调测。  
  按照工程创建->算子实现->编译部署>算子调用的流程完成算子开发。整个过程都依赖于算子工程：基于工程代码框架完成算子核函数的开发和Tiling实现，通过工程编译脚本完成算子的编译部署，继而实现单算子调用或第三方框架中的算子调用。
- [KernelLaunch](./KernelLaunch)：使用核函数直调Matmul算子及printf调测。  
  核函数的基础调用（Kernel Launch）方式，开发者完成算子核函数的开发和Tiling实现后，即可通过AscendCL运行时接口，完成算子的调用。

本样例中包含如下调用方式：
<table>
<th> 调用方式 </th>
<th> 目录 </th>
<th> 描述 </th>
<tr>
<th rowspan="1"><a href="./FrameworkLaunch"> FrameworkLaunch </a></th>
<td><a href="./FrameworkLaunch/AclNNInvocation"> AclNNInvocation </a></td>
<td> 通过aclnn调用的方式调用Matmul算子及printf调测。 </td>
</tr>
<tr>
<th rowspan="1"><a href="./KernelLaunch"> KernelLaunch </a></th>
<td><a href="./KernelLaunch/MatmulInvocationNeo"> MatmulInvocationNeo </a></td>
<td> Kernel Launch方式调用Matmul算子及printf调测。 </td>
</tr>
</table>

## 算子描述
Matmul实现了快速的Matmul矩阵乘法的运算操作。
printf可以实现调试场景下的格式化输出功能。
本样例将打印的内容保存到输出文件中，并对比文件中是否有打印的内容，以及样例正常执行未中断报错，从而判断样例是否执行成功。

Matmul的计算公式为：

```
C = A * B + Bias
```

- A、B为源操作数，A为左矩阵，形状为\[M, K]；B为右矩阵，形状为\[K, N]。
- C为目的操作数，存放矩阵乘结果的矩阵，形状为\[M, N]。
- Bias为矩阵乘偏置，形状为\[N]。对A*B结果矩阵的每一行都采用该Bias进行偏置。

在上述算子中添加printf，可以打印执行过程中的部分变量值，变量类型包括无符号整型、整型、字符型、bool类型、half和float浮点型等等，可以方便调试。

## 算子规格描述
在核函数直调样例中，算子实现支持的shape为：M = 512, N = 1024, K = 512。
<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Matmul</td></tr>
</tr>
<tr><td rowspan="4" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">a</td><td align="center">M * K</td><td align="center">float16</td><td align="center">ND</td></tr>
<tr><td align="center">b</td><td align="center">K * N</td><td align="center">float16</td><td align="center">ND</td></tr>
<tr><td align="center">bias</td><td align="center">N</td><td align="center">float</td><td align="center">ND</td></tr>
</tr>
</tr>
<tr><td rowspan="1" align="center">算子输出</td><td align="center">c</td><td align="center">M * N</td><td align="center">float</td><td align="center">ND</td></tr>
</tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">matmul_custom</td></tr>
</table>

## 支持的产品型号
本样例支持如下产品型号：
- Atlas 推理系列产品AI Core
- Atlas A2训练系列产品/Atlas 800I A2推理产品

## 目录结构介绍
```
├── FrameworkLaunch         //使用框架调用的方式调用Matmul算子及printf调测。
└── KernelLaunch            //使用核函数直调的方式调用Matmul算子及printf调测。
```
## 环境要求
编译运行此样例前，请参考[《CANN软件安装指南》](https://hiascend.com/document/redirect/CannCommunityInstSoftware)完成开发运行环境的部署。

## 编译运行样例算子

### 1. 准备：获取样例代码<a name="codeready"></a>

 编译运行此样例前，请参考[准备：获取样例代码](../README.md#codeready)获取源码包。

### 2. 编译运行样例工程
- 若使用框架调用的方式，编译运行操作请参见[FrameworkLaunch](./FrameworkLaunch)。
- 若使用核函数直调的方式，编译运行操作请参见[KernelLaunch](./KernelLaunch)。
## 更新说明
| 时间       | 更新事项                                            |
| ---------- | --------------------------------------------------- |
| 2025/01/06 | 新增本readme                                      |
