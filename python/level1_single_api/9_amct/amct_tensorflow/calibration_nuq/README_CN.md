# ResNet-50 V1

## 1. 非均匀量化

### 1.1 量化前提

+ **模型准备**  
请下载[ResNet-50](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/amct_resnet-50_v1/resnet50_tf.pb) 模型文件。将模型文件重命名为 resnet_v1_50.pb 放到 [model](./model/) 目录下。

+ **数据集准备**  
使用昇腾模型压缩工具对模型完成量化后，需要对模型进行推理，以测试量化数据的精度。推理过程中需要使用和模型相匹配的数据集。请下载测试图片 [classification.jpg](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/models/mobilenet_v2_calibration/classification.jpg)，并将该图片放到 [data](./data/) 目录下。

+ **校准集准备**  
校准集用来产生量化因子，保证精度。
计算量化参数的过程被称为“校准 (calibration)”。校准过程需要使用一部分测试图片来针对性计算量化参数，使用一个或多个 batch 对量化后的网络模型进行推理即可完成校准。请下载[校准集](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/models/mobilenet_v2_calibration/calibration.rar)，解压后将 calibration 文件夹放到 [data](./data/) 目录下。

### 1.2 量化示例

执行量化示例前，请先检查当前目录下是否包含以下文件及目录：

+ [data](./data/)
  + calibration
  + classification.jpg
+ [model/](./model/)
  + resnet_v1_50.pb
+ [src](./src/)
  + [nuq_conf](./src/nuq_conf/)
    + [nuq_quant.cfg](src/nuq_conf/nuq_quant.cfg)
    + [resnet-50_v1_quantized.json](./src/nuq_conf/resnet-50_v1_quantized.json)
  + [resnet-50_v1_nuq.py](./src/resnet-50_v1_nuq.py)

在当前目录执行如下命令

```bash
python ./src/resnet-50_v1_nuq.py
```

若出现如下信息则说明模型量化成功：

```none
INFO - [AMCT]:[save_model]: The model is saved in ./outputs/nuq/resnet-50_v1_quantized.pb
Origin Model Prediction:
        category index: 699
        category weight: 6.534
Quantized Model Prediction:
        category index: 699
        category weight: 6.661
```

### 1.3 量化结果

量化成功后，在当前目录会生成量化日志文件 ./amct_log/amct_tensorflow.log 和 ./outputs/nuq 文件夹，该文件夹内包含以下内容：

+ amctorflow_nuq_record.txt: NUQ 量化层。
+ config.json: 量化配置文件，描述了如何对模型中的每一层进行量化。
+ record.txt: 量化因子记录文件，记录量化因子。
+ resnet-50_v1_quant.json: 量化信息文件。
+ resnet-50_v1_quantized.pb: 量化模型，可在 TensorFlow 环境进行精度仿真并可在昇腾 AI 处理器部署。

> 对该模型重新进行量化时，在量化后模型的同级目录下生成的上述结果文件将会被覆盖。
