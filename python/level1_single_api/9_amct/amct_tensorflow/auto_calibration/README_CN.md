# MobileNet V2

## 1. 基于精度的自动量化

### 1.1 量化前提

+ **模型准备**  
请点击下载 [MobileNet V2](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com:443/resource/mobilenet_v2_1.0_224.tgz) 模型文件。解压并将其中的 mobilenet_v2_1.0_224_frozen.pb 文件放到 [model](./model/) 目录下。

+ **数据集准备**  
自动量化回退过程中，需要不断的对模型进行校准和测试，因此需要用户准备数据集，本示例所采用的数据集为标准 TFRecord 格式的 ImageNet 的 子集 ILSVRC-2012-CLS 的验证集，共有 50000 张图片，如果采用其他数据集，则需要用户自行修改 sample 文件中的数据预处理部分以匹配模型输入。

### 1.2 量化示例

执行量化示例前，请先检查当前目录下是否包含以下文件及目录：

+ [model](./model/)
  + mobilenet_v2_1.0_224_frozen.pb
+ [src](./src/)
  + [mobilenet_v2_accuracy_based_auto_calibration.py](./src/mobilenet_v2_accuracy_based_auto_calibration.py)

在当前目录执行如下命令运行示例程序：

```bash
python ./src/mobilenet_v2_accuracy_based_auto_calibration.py --dataset DATASET
```

上述命令只给出了常用的参数，不常用参数以及各个参数解释请参见如下表格：

| 参数 | 必填项 | 数据类型 | 默认值 | 参数解释 |
| :-- | :-: | :-: | :-: | :-- |
| -h | 否 | / | / | 显示帮助信息。 |
| --dataset | 是 | string | None | 标准 TFRecord 格式的 ImageNet 的子集 ILSVRC-2012-CLS 的验证部分。 |
| --num_parallel_reads | 否 | int | 4 | 用于读取数据集的线程数，根据硬件运算能力酌情调整。 |
| --batch_size | 否 | int | 32 | TensorFlow 运行一次所使用的样本数量，根据内存或显存大小酌情调整。 |
| --model | 否 | string | ./model/mobilenetv2_tf.pb | 自动量化回退时使用的原始模型。 |

若出现如下信息则说明模型量化成功：

```none
INFO - [AMCT]:[AMCT]: Accuracy of original model is         71.84
INFO - [AMCT]:[AMCT]: Accuracy of global quantized model is 70.85
INFO - [AMCT]:[AMCT]: Accuracy of saved model is            71.476
INFO - [AMCT]:[AMCT]: The generated model is stored in dir: ./outputs/accuracy_based_auto_calibration
INFO - [AMCT]:[AMCT]: The records is stored in dir: ./outputs/accuracy_based_auto_calibration
```

### 1.3 量化结果

量化成功后，在当前目录会生成如下文件：

+ amct_log: 量化日志文件夹
  + amct_tensorflow.log: 量化日志文件。
  + accuracy_based_auto_calibration_record.json: 基于精度的自动量化回退历史记录文件。
+ ./outputs/accuracy_based_auto_calibration: 输出文件夹
  + accuracy_based_auto_calibration_final_config.json: 回退后的量化配置文件，描述了如何对模型中的每一层进行量化。
  + accuracy_based_auto_calibration_ranking_information.json: 量化层量化敏感信息。
  + config.json: 回退前的量化配置文件，描述了如何对模型中的每一层进行量化。
  + mobilenet_v2_quantized.pb: 量化模型，可在 TensorFlow 环境进行精度仿真并可在昇腾 AI 处理器部署。
  + mobilenet_v2_quant.json: 量化信息文件，记录了量化模型同原始模型节点的映射关系，用于量化后模型同原始模型精度比对使用。
  + record.txt: 量化因子记录文件，记录量化因子。

> 对该模型重新进行量化时，在量化后模型的同级目录下生成的上述结果文件将会被覆盖。
