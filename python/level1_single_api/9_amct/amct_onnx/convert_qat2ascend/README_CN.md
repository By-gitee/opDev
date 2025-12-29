## QAT 模型转 Ascend 模型

如果 Onnx 框架的原始模型已经做了量化功能（以下简称 QAT 模型），但是该模型无法使用 ATC 工具转成适配昇腾 AI 处理器的离线模型，则需要借助 昇腾模型压缩工具提供的 `convert_qat_model` 接口，将该 QAT 模型转成 Ascend 量化模型格式，然后才能使用 ATC 工具，将 Ascend 量化模型转成适配昇腾 AI 处理器的离线模型。

### 1 量化前提

+ **模型准备**  
请下载 [MobileNetV2 QAT](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/mobilenetv2_convert_qat/mobilenetv2_qat.onnx) 模型文件到 [model](./model/) 目录。

+ **数据集准备**  
使用昇腾模型压缩工具对模型完成量化后，需要对模型进行推理，以测试量化数据的精度。推理过程中需要使用和模型相匹配的数据集。请下载[测试图片](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/models/mobilenet_v2_calibration/classification.jpg)并将该图片放到 [data](./data/) 目录下。

### 2 量化示例

执行量化示例前，请先检查当前目录下是否包含以下文件及目录：

+ [data](./data/)
  + classification.jpg
+ [model](./model/)
  + mobilenetv2_qat.onnx
+ [src](./src/)
  + [mobilenet_v2_convert_qat.py](./src/mobilenet_v2_convert_qat.py)

在当前目录执行如下命令运行示例程序：

```none
python ./src/mobilenet_v2_convert_qat.py
```

若出现如下信息则说明模型量化成功：

```none
INFO - [AMCT]:[Utils]: The model file is saved in ./outputs/calibration/mobilenet_v2_deploy_model.onnx
INFO - [AMCT]:[Utils]: The model file is saved in ./outputs/calibration/mobilenet_v2_fake_quant_model.onnx
Origin Model Prediction:
        category index: 699
        category prob: 0.543
Quantized Model Prediction:
        category index: 699
        category prob: 0.398
```

### 3 量化结果

量化成功后，在当前目录会生成量化日志文件 ./amct_log/amct_onnx.log 和 outputs 文件夹，该文件夹内包含以下内容：

+ convert_qat
  + record.txt: 量化因子记录文件记录量化因子。
  + mobilenet_v2_fake_quant_model.onnx: 量化仿真模型，即可在 ONNX 环境进行精度仿真的模型。
  + mobilenet_v2_deploy_model.onnx: 量化部署模型，即可在昇腾 AI 处理器部署的模型。

> 对该模型重新进行量化时，在量化后模型的同级目录下生成的上述结果文件将会被覆盖。
