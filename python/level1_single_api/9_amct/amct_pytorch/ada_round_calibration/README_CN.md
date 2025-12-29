# ResNet-18

## 1. ada round量化校准

### 1.1 量化前提

+ **模型准备**  
请下载 [ResNet-18](https://download.pytorch.org/models/resnet18-5c106cde.pth) 模型文件并保存到`model`目录。


+ **数据集准备**  
使用昇腾模型压缩工具对模型完成量化后，需要对模型进行推理，以测试量化数据的精度。推理过程中需要使用与模型相匹配的数据集。请下载[测试图片](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/resnet-101_nuq/images.zip)，解压后将`images`文件夹放到`data`目录。

+ **校准集准备**  
校准集用来产生量化因子，保证精度。本 sample 校准集与数据集相同。

### 1.2 简易量化配置
./config/ada_round.cfg文件为用户自定义的简易量化配置，具体表示信息如下：


| 字段 |类型| 说明 | 默认值 | 取值范围 |
|:--| :-: | :-- | :-: | :-: |
|common_config.ada_quantize.num_iteration|uint|adaround训练迭代次数|10000|>0|
|common_config.ada_quantize.warm_start|float|预热因子|0.2|(0,1)|
|common_config.ada_quantize.reg_param|float|正则化参数|0.01|(0,1)|
|common_config.ada_quantize.channel_wise|bool|是否开启pre channel|true|false/true|
|common_config.ada_quantize.beta_range_start|uint|退火起始参数|20|>beta_range_end|
|common_config.ada_quantize.beta_range_end|uint|退火终止参数|2|>0|

更多参数配置请参考[官方文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/devaids/devtools/amct/atlasamct_16_0131.html)


### 1.3 量化示例

执行量化示例前，请先检查当前目录下是否包含以下文件及目录，其中 images 文件夹内部包含有 160 张用于校准和测试的图片：

+ [data](./data/)
  + images
+ [model](./model/)
  + resnet18-5c106cde.pth
+ [src](./src/)
  + [config/ada_round.cfg](./src/config/ada_round.cfg)
  + [\_\_init__.py](./src/__init__.py)
  + [resnet-18_calibration.py](./src/resnet-18_calibration.py)
  + [resnet.py](./src/resnet.py)

请在当前目录执行如下命令运行示例程序：

```bash
CUDA_VISIBLE_DEVICES=0 python ./src/resnet-18_calibration.py
```

> 其中 `CUDA_VISIBLE_DEVICES` 是必填参数，表示使用 CPU 还是 GPU 进行量化，参数取值为：
>
> + -1：使用 CPU 进行量化。
> + 其他 Device ID使用 GPU 进行量化，具体 ID 请以用户实际环境为准。
> 

若出现如下信息，则说明量化成功：

```none
INFO - [AMCT]:[Utils]: The model file is saved in ./outputs/calibration/resnet-18_deploy_model.onnx
INFO - [AMCT]:[Utils]: The model file is saved in ./outputs/calibration/resnet-18_fake_quant_model.onnx
[INFO] ResNet18 before quantize top1:    0.84375 top5:    0.9625
[INFO] ResNet18 after quantize  top1:    0.84375 top5:   0.95625
```

### 1.4 量化结果

量化成功后，在当前目录会生成量化日志文件 ./amct_log/amct_pytorch.log 和 ./outputs/calibration 文件夹，该文件夹内包含以下内容：


+ tmp: 临时文件夹
  + config.json: 量化配置文件，描述了如何对模型中的每一层进行量化。
  + record.txt: 量化因子记录文件记录量化因子。
  + modified_model.onnx: 临时模型文件，即原始的 PyTorch 模型 BN 融合后导出的 ONNX 模型文件。
+ resnet-18_deploy_model.onnx: 量化部署模型，即量化后的可在昇腾 AI 处理器部署的模型文件。
+ resnet-18_fake_quant_model.onnx: 量化仿真模型，即量化后的可在 ONNX 执行框架 ONNX Runtime 进行精度仿真的模型

> 如果量化脚本所在目录下已经存在量化配置文件，则再次调用 `create_quant_config` 接口时，如果新生成的量化配置文件与已有的文件同名，则会覆盖已有的量化配置文件，否则生成新的量化配置文件。
