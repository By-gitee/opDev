## 概述
本样例展示了DumpTensor接口的使用方法，分Cube场景和Vector场景，并介绍了单算子工程、单算子调用。其中Cube场景是在Mmad样例的基础上，Dump指定Tensor内容；Vector场景是在Add样例的基础上，Dump指定Tensor内容。

## 目录结构介绍
```
├── FrameworkLaunch                // 使用框架调用的方式调用DumpTensor接口。
│   ├── DumpTensorCube             // 使用框架调用的方式调用Cube场景MmadCustom算子工程，并添加DumpTensor调测功能。
│   └── DumpTensorVector           // 使用框架调用的方式调用Vector场景AddCustom算子工程，并添加DumpTensor调测功能。
```

## 算子描述
详见[Cube场景算子描述](./DumpTensorCube/README.md#descReady)和[Vector场景算子描述](./DumpTensorVector/README.md#descReady)

## 支持的产品型号
本样例支持如下产品型号：
- Atlas 推理系列产品AI Core
- Atlas A2训练系列产品/Atlas 800I A2推理产品

## 编译运行样例算子
详见[Cube场景编译运行样例算子](./DumpTensorCube/README.md#runReady)和[Vector场景编译运行样例算子](./DumpTensorVector/README.md#runReady)

## 更新说明
| 时间       | 更新事项                     |
| ---------- | ---------------------------- |
| 2025/01/06 | 新增本readme |
