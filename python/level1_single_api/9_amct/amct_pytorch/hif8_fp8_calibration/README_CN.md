# ResNet-101

## 1. HIF8/FP8校准

### 1.1 量化前提

+ **模型准备**  
请下载 [ResNet-101](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/resnet-101_nuq/resnet101-5d3b4d8f.pth) 模型文件并保存到`model`目录。


+ **数据集准备**  
使用昇腾模型压缩工具对模型完成量化后，需要对模型进行推理，以测试量化数据的精度。推理过程中需要使用与模型相匹配的数据集。请下载[测试图片](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/resnet-101_nuq/images.zip)，解压后将`images`文件夹放到`data`目录。

+ **校准集准备**  
校准集用来产生量化因子，保证精度。本 sample 校准集与数据集相同。

### 1.2 简易量化配置
./quant_conf/quant.cfg文件为用户自定义的简易量化配置，具体表示信息如下：


| 字段 |类型| 说明 | 默认值 | 取值范围 | 注意事项 |
|:--| :-: | :-- | :-: | :-: | :-: |
|batch_num|uint32|量化使用的batch数量 |1|/|/|
|common_config.act_type|enum|量化后激活值类型|HIFLOAT8|HIFLOAT8/FLOAT8_E4M3FN|act_type与wts_type需要保持一致|
|common_config.wts_type|enum|量化后权重类型|HIFLOAT8|HIFLOAT8/FLOAT8_E4M3FN|/|
|common_config.weight_granularity|enum|权重量化粒度|/|PER_TENSOR/PER_CHANNEL|/|
|common_config.round_mode|enum|舍入模式|/|HYBRID/ROUND/RINT|HIFLOAT8支持HYBRID/ROUND模式，FLOAT8_E4M3FN仅支持RINT模式|

### 1.3 量化示例

执行量化示例前，请先检查当前目录下是否包含以下文件及目录，其中 images 文件夹内部包含有 160 张用于校准和测试的图片：

+ [data](./data/)
  + images
+ [model](./model/)
  + resnet101-5d3b4d8f.pth
+ [src](./src/)
  + [quant_conf/quant.cfg](./src/quant_conf/quant.cfg)
  + [\_\_init__.py](./src/__init__.py)
  + [resnet-101_calibration.py](./src/resnet-101_calibration.py)
  + [resnet.py](./src/resnet.py)

请在当前目录执行如下命令运行示例程序：

```bash
CUDA_VISIBLE_DEVICES=0 python ./src/resnet-101_calibration.py
```

> 其中 `CUDA_VISIBLE_DEVICES` 是必填参数，表示使用 CPU 还是 GPU 进行量化，参数取值为：
>
> + -1：使用 CPU 进行量化。
> + 其他 Device ID使用 GPU 进行量化，具体 ID 请以用户实际环境为准。
> 
> test-on-npu-flag参数表明是否要生成部署模型在NPU上进行测试：
> + 设置该参数：生成部署模型并在NPU上进行测试，当前不支持设置该参数（暂未适配算子）
> + 不设置该参数: 不生成部署模型


### 1.4 量化结果

量化成功后，在当前目录会生成量化日志文件 ./amct_log/amct_pytorch.log 和 ./outputs/calibration 文件夹，该文件夹内包含以下内容：

+ tmp: 临时文件夹
  + config.json: 量化配置文件，描述了如何对模型中的每一层进行量化。
  + record.txt: 量化因子记录文件记录量化因子。

