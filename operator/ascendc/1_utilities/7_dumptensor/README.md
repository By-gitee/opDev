## DumpTensor样例说明
本样例通过Ascend C编程语言实现了Add算子和Mmad算子，同时在算子中添加DumpTensor调测，并按照不同的算子调用方式分别给出了对应的端到端实现。
- [FrameworkLaunch](./FrameworkLaunch)：使用框架调用Add算子和Mmad算子及DumpTensor调测。  
  按照工程创建->算子实现->编译部署>算子调用的流程完成算子开发。整个过程都依赖于算子工程：基于工程代码框架完成算子核函数的开发和Tiling实现，通过工程编译脚本完成算子的编译部署，继而实现单算子调用或第三方框架中的算子调用。
- [KernelLaunch](./KernelLaunch)：使用核函数直调Add算子和Mmad算子及DumpTensor调测。  
  核函数的基础调用（Kernel Launch）方式，开发者完成算子核函数的开发和Tiling实现后，即可通过AscendCL运行时接口，完成算子的调用。


本样例中包含如下调用方式：
<table>
    <th>调用方式</th><th>目录</th><th>描述</th>
    <tr>
        <td rowspan='2'><a href="./FrameworkLaunch"> FrameworkLaunch</td>
        <td><a href="./FrameworkLaunch/DumpTensorCube"> DumpTensorCube</td><td>使用框架调用的方式调用Cube场景MmadCustom算子工程，并添加DumpTensor调测功能。</td>
    </tr>
    <tr>
        <td><a href="./FrameworkLaunch/DumpTensorVector"> DumpTensorVector</td><td>使用框架调用的方式调用Vector场景AddCustom算子工程，并添加DumpTensor调测功能。</td>
    </tr>
    <tr>
        <td rowspan='3'><a href="./KernelLaunch"> KernelLaunch</td>
    </tr>
    <tr>
        <td><a href="./KernelLaunch/DumpTensorKernelInvocationCube"> DumpTensorKernelInvocationCube</td><td>Kernel Launch方式调用Cube场景核函数的样例，Cube场景为Mmad算子实现样例</td>
    </tr>
    <tr>
        <td><a href="./KernelLaunch/DumpTensorKernelInvocationVector"> DumpTensorKernelInvocationVector</td><td>Kernel Launch方式调用Vector场景核函数的样例，Vector场景为Add算子实现样例。</td>
    </tr>
</table>

## 算子描述
###  DumpTensor介绍
使用DumpTensor可以Dump指定Tensor的内容，同时支持打印自定义的附加信息。此外，DumpAccChkPoint可以支持指定偏移位置的Tensor打印。
本样例将Dump的内容保存到输出文件中，并对比文件中是否有Dump的内容，从而判断样例是否执行成功。

### Cube场景Mmad算子介绍
算子使用基础API包括DataCopy、LoadData、Mmad等，实现矩阵乘功能。

计算公式为：

```
C = A * B + Bias
```

- A、B为源操作数，A为左矩阵，形状为\[M, K]；B为右矩阵，形状为\[K, N]。
- C为目的操作数，存放矩阵乘结果的矩阵，形状为\[M, N]。
- Bias为矩阵乘偏置，形状为\[N]。对A*B结果矩阵的每一行都采用该Bias进行偏置。

### Vector场景Add算子介绍
Add算子实现了两个数据相加，返回相加结果的功能。对应的数学表达式为：  
```
z = x + y
```

## 支持的产品型号
本样例支持如下产品型号：
- Atlas 推理系列产品AI Core
- Atlas A2训练系列产品/Atlas 800I A2推理产品

## 目录结构介绍
```
├── FrameworkLaunch         //使用框架调用的方式调用Add算子和Mmad算子及DumpTensor调测。
└── KernelLaunch            //使用核函数直调的方式调用Add算子和Mmad算子及DumpTensor调测。
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

