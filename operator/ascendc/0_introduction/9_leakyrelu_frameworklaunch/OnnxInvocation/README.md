AscendC 自定义算子入Onnx网络示例教程:
以Atlas 推理系列产品AI Core上leakyrelu单算子离线推理为例。

## 目录结构介绍
```
└── OnnxInvocation             // 通过onnx网络调用的方式调用LeakyReluCustom算子
    └── leaky_relu.py          // 生成单算子onnx模型的文件
```

## 运行样例算子
### 一、自定义算子准备
1.在LeakyReluCustom目录下执行编译操作，编译出算子run包。
2.安装在LeakyReluCustom/build_out/目录下生成的自定义算子run包。

### 二、离线推理验证流程
1.获取单算子onnx模型, 该模型参考leaky_relu.py生成
```bash
wget https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/AscendC/leaky_relu.onnx
```

2.onnx模型转换为om模型

示例如下，Ascendxxxyy请替换为实际的AI处理器型号。
```bash
atc --model=./leaky_relu.onnx --framework=5 --soc_version=Ascendxxxyy --output=./leaky_relu --input_shape="X:8,16,1024" --input_format=ND
```
- 使用export ASCEND_GLOBAL_LOG_LEVEL=1改变日志级别为INFO，若出现:
start compile Ascend C operator LeakyReluCustom. kernel name is leaky_relu_custom
compile Ascend C operator: LeakyReluCustom success!
打印，表明进入了AscendC算子编译

- 若出现：
ATC run success, welcome to the next use 表明离线om模型转换成功

3.执行离线推理
可使用https://gitee.com/ascend/tools/tree/master/msame 工具进行推理验证

## 更新说明
| 时间      | 更新事项     |
| --------- | ------------ |
| 2023/5/15 | 更新目录结构 |
| 2024/11/11 | 样例目录调整 |