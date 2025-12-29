## 概述
本样例以AddCustomTilingSink自定义算子为例，介绍了在开发自定义算子时如何启用Tiling下沉，以及如何通过PyTorch在图模式下调用该自定义算子的完整流程。

## 目录结构介绍

```
├── AddCustomTilingSink      
│   ├── AddCustomTilingSink  // AscendC算子实现
│   └── PytorchInvocation    // Pytorch调用样例
```

## 算子描述
Add算子实现了两个数据相加，返回相加结果的功能。对应的数学表达式为：
```
z = x + y
```
## 算子规格描述
<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Add</td></tr>
</tr>
<tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
<tr><td align="center">y</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
</tr>
</tr>
<tr><td rowspan="1" align="center">算子输出</td><td align="center">z</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
</tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">add_custom_tiling_sink</td></tr>
</table>

## 支持的产品型号
本样例支持如下产品型号：
- Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件
- Atlas A3 训练系列产品/Atlas A3 推理系列产品

## 编译运行样例算子

### 1. 实现Pytorch自定义算子并注册
请参考本目录中[PytorchInvocation/readme.md](./PytorchInvocation/README.md)实现Pytorch侧注册。

### 2. 实现CANN自定义算子，并完成编译部署
请参考本目录中[AddCustomTilingSink/README.md](./AddCustomTilingSink/README.md)部署自定义算子包。

### 3. 执行测试脚本
执行本目录中[PytorchInvocation/test_add_custom_tiling_sink.py](./PytorchInvocation/test_add_custom_tiling_sink.py)测试脚本验证功能。 

## 更新说明

| 时间      | 更新事项     |
| --------- | ------------ |
| 2025/5/28 | 新增本readme |
