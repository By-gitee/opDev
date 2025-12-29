## 概述
本样例展示了DumpTensor接口的使用方法，分为Cube场景和Vector场景。其中Cube场景是在[20_mmad_kernellaunch](../../../0_introduction//20_mmad_kernellaunch)样例的基础上，Dump指定Tensor内容；Vector场景是在[3_add_kernellaunch](../../../0_introduction/3_add_kernellaunch)样例的基础上，Dump指定Tensor内容。样例中算子的调用方式采用核函数直调方式。

## 目录结构介绍
```
├── KernelLaunch
│   ├── DumpTensorKernelInvocationCube           // Kernel Launch方式调用Cube场景核函数的样例，Cube场景为Mmad算子实现样例。
│   └── DumpTensorKernelInvocationVector         // Kernel Launch方式调用Vector场景核函数的样例，Vector场景为Add算子实现样例。
```

## 编译运行样例算子
针对自定义算子工程，编译运行包含如下步骤：
- 编译自定义算子工程；
- 调用执行自定义算子；

详细操作如下所示。
### 1. 获取源码包
编译运行此样例前，请参考[准备：获取样例代码](../../README.md#codeready)完成源码包获取。

### 2. 编译运行样例工程
- [DumpTensorKernelInvocationCube样例运行](./DumpTensorKernelInvocationCube/README.md)
- [DumpTensorKernelInvocationVector样例运行](./DumpTensorKernelInvocationVector/README.md)

## 更新说明
| 时间       | 更新事项                     |
| ---------- | ---------------------------- |
| 2025/01/06 | 新增本readme |
