## 基于精度的自动量化

该接口会根据模型在数据集测试的精度和用户设定的精度损失要求来自动决定每一层是否进行量化，最终生成量化的混合模型；精度目标如果设置的比较苛刻，所有层均不量化才能达到目标则不会生成量化模型，可以适当降低精度损失的目标，继续调用该接口进行模型的量化。

### 1 量化前提

+ **模型准备**  
请下载 [MobileNet-V2](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/acc_based_auto_calibration/mobilenetv2_v11.onnx) 模型文件到 [model](./model/) 目录。

+ **数据集准备**  
使用昇腾模型压缩工具对模型完成量化后，需要对模型进行推理，以测试量化数据的精度。推理过程中需要使用和模型相匹配的数据集。请下载[测试图片](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/resnet-101_nuq/images.zip)，解压后将 “images” 文件夹放到 [data](./data/) 目录下。

+ **校准集准备**  
校准集用来产生量化因子，保证精度。本 sample 校准集与数据集相同。

### 2 量化示例

在当前目录执行如下命令运行示例程序：

```bash
python ./src/mobilenet_v2_accuracy_based_auto_calibration.py
```

### 3 量化结果

量化成功后会生成量化 fake quant 和 deploy 的 onnx 模型文件和量化层敏感度的信息文件, 量化因子记录文件等；

| 目录     | 文件或目录                                               | 说明                                                         |
| -------- | -------------------------------------------------------- | ------------------------------------------------------------ |
| amct_log | amct_onnx.log                                            | amct_onnx 工具在执行过程中相关日志文件                       |
|          | accuracy_based_auto_calibration_record.json              | 基于精度的自动量化回退过程中的量化配置记录文件               |
|          | tempxxxxx/ 目录                                          | 里面保存了隐藏层的 feature map 和其他临时性模型文件，当执行完毕后就可以删除 |
| results  | mobilenet_v2_fake_quant_model.onnx                       | 生成的 fake quant onnx 模型文件                              |
|          | mobilenet_v2_deploy_model.onnx                           | 生成的 deploy onnx 模型文件                                  |
|          | mobilenet_v2_quant.json                                  | 生成的融合信息文件                                           |
|          | accuracy_based_auto_calibration_final_config.json        | 基于精度的自动量化回退最终搜索得到的量化配置文件             |
|          | accuracy_based_auto_calibration_ranking_information.json | 基于精度的自动量化回退过程中记录的每层量化敏感度信息文件     |
| tmp      | config.json                                              | 量化过程中的量化配置文件                                     |
|          | scale_offset_record.txt                                  | 量化因子记录文件                                             |
