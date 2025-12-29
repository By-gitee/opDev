# MobileNet V2

## 1. `convert_model` 接口

### 1.1 量化前提

+ **模型准备**  
请点击下载 [MobileNet V2](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com:443/resource/mobilenet_v2_1.0_224.tgz) 模型文件。解压并将其中的 mobilenet_v2_1.0_224_frozen.pb 文件放到 [model](./model/) 目录下。

+ **数据集准备**  
使用昇腾模型压缩工具对模型完成量化后，需要对模型进行推理，以测试量化数据的精度。推理过程中需要使用和模型相匹配的数据集。请下载测试图片 [classification.jpg](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/models/mobilenet_v2_calibration/classification.jpg)，并将该图片放到 [data](./data/) 目录下。

### 1.2 量化示例

执行量化示例前，请先检查当前目录下是否包含以下文件及目录：

+ [data](./data/)
  + classification.jpg
  + [record_quantized.txt](./data/record_quantized.txt)
+ [model](./model/)
  + mobilenet_v2_1.0_224_frozen.pb
+ [src](./src/)
  + [mobilenet_v2_convert_model.py](./src/mobilenet_v2_convert_model.py)

在当前目录执行如下命令运行示例程序：

```bash
python ./src/mobilenet_v2_convert_model.py
```

若出现如下信息则说明模型量化成功：

```none
INFO - [AMCT]:[save_model]: The model is saved in ./outputs/convert_model/mobilenet_v2_quantized.pb
Origin Model Prediction:
        category index: 443
        category prob: 0.375
Quantized Model Prediction:
        category index: 443
        category prob: 0.517
```

### 1.3 量化结果

量化成功后，在当前目录会生成量化日志文件 ./amct_log/amct_tensorflow.log 和 ./outputs/convert_model 文件夹，文件夹内包含以下内容：

+ mobilenet_v2_quant.json: 量化信息文件，记录了量化模型同原始模型节点的映射关系，用于量化后模型同原始模型精度比对使用。
+ mobilenet_v2_quantized.pb: 量化模型，可在 TensorFlow 环境进行精度仿真并可在昇腾 AI 处理器部署。

> 对该模型重新进行量化时，在量化后模型的同级目录下生成的上述结果文件将会被覆盖。
