## `AddCustom`自定义算子样例说明

本样例通过`Ascend C`编程语言实现了`AddCustom`算子带Tiling场景。

### 算子描述

`AddCustom`算子返回两个数据相加的结果。

### 算子规格描述

| 算子类型(OpType) | AddCustom  |          |           |        |
| ---------------- | ---------- | -------- | --------- | ------ |
| 算子输入         | name       | shape    | data type | format |
| x                | 8 * 2048   | float16  | ND        |        |
| y                | 8 * 2048   | float16  | ND        |        |
| 算子输出         | z          | 8 * 2048 | float16   | ND     |
| 核函数名         | add_custom |          |           |        |

### 支持的产品型号

本样例支持如下产品型号：

- Atlas 训练系列产品
- Atlas 推理系列产品
- Atlas A2训练系列产品
- Atlas 800I A2推理产品
- Atlas 200I/500 A2推理产品

### 目录结构介绍

```
├── examples                     // 调用示例目录
├── add_custom_tiling.h          // 算子tiling结构体定义
├── add_custom.cpp               // 算子kernel代码
├── CMakeLists.txt               // cmake编译文件
├── run.sh                       // 运行脚本
└── README.md                   // 样例指导手册 
```

### 环境要求

编译运行此样例前，请参考[《CANN软件安装指南》](https://gitee.com/link?target=https%3A%2F%2Fhiascend.com%2Fdocument%2Fredirect%2FCannCommunityInstSoftware)完成开发运行环境的部署。

### 算子包编译部署

1.进入到样例目录

```
cd ${git_clone_path}/samples/operator_contrib/AddCustomSample/KernelLaunch/AddCustomKernelTiling
```

2.算子编译部署

- 打包动态库部署

  ```
  bash run.sh -l SHARED -v Ascend***(由npu-smi info查询得到)
  ```

- 打包静态库部署

  ```
  bash run.sh -l STATIC -v Ascend***(由npu-smi info查询得到)
  ```

  

### 算子调用

| 目录                                                         | 描述                                     |
| ------------------------------------------------------------ | ---------------------------------------- |
| [PythonInvocation](./examples/PythonInvocation) | 通过Python调用的方式调用AddCustom算子。 |

### 更新说明

| 时间       | 更新事项     |
| ---------- | ------------ |
| 2025/01/06 | 新增本readme |