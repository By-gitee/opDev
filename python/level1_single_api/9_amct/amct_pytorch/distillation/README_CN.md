## 1.逐层蒸馏

本功能根据图结构与用户配置将模型进行Block切分，并根据划分结果进行逐个Block的蒸馏量化获得量化因子，最终根据量化因子生成量化后模型。

### 1.1 蒸馏前提

- **安装依赖**

本 sample 依赖 torchvison 包：

安装命令：

```
pip3 install torchvision==0.9.0
```

torchvison 0.9.0 是 torch 1.8.0 的配套版本，如果使用其他版本的torch，则安装配套的 torchvison； 如果需要使用 GPU 版本的 torch, 则安装和 CUDA 环境对应的 torch, torchvision 版本即可，详细安装指导可以参考 torch 官网

+ **数据集准备**  

使用昇腾模型压缩工具逐层蒸馏功能时，需要使用与模型匹配的数据集进行蒸馏流程。请下载[测试图片](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/resnet-101_nuq/images.zip)，解压后将 “images” 文件夹放到 [data](./data/) 目录下。

### 1.2 蒸馏示例

执行量化示例前，请先检查当前目录下是否包含以下文件及目录，其中 images 文件夹内部包含有 160 张用于蒸馏和测试的图片：

+ [data](./data/)
  + images
+ [src](./src/)
  + [mobilenet_v2_distill.py](./src/mobilenet_v2_distill.py)

请在当前目录执行如下命令运行示例程序：

+ 单卡蒸馏

  ```bash
  CUDA_VISIBLE_DEVICES=0 python ./src/mobilenet_v2_distill.py
  ```

+ 蒸馏

  ```bash
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ./src/mobilenet_v2_distill.py --distributed
  ```

  > 仅支持 distribution 模式的多卡蒸馏，不支持 DataParallel 模式的多卡蒸馏。如果使用 DataParallel 模式的多卡蒸馏，会出现如下错误信息：
  >
  > ```none
  > RuntimeError: The model cannot export to onnx, exception is: torch.nn.DataParallel is not supported by ONNX exporter, please use 'attribute' module to unwrap model from torch.nn.DataParallel. Try torch.onnx.export(model.module, ...)
  > ```

上述命令只给出了常用的参数，不常用参数以及各个参数解释请参见如下表格：

| 参数 | 必填项 | 数据类型 | 默认值 | 参数解释 |
| :-- | :-: | :-: | :-: | :-- |
| -h | 否 | / | / | 显示帮助信息。 |
| --config_defination CONFIG_DEFINATION | 否 | string | None | 量化的简易配置文件路径。 |
|  --num_parallel_reads NUM_PARALLEL_READS | 否 | int | 4 | 用于读取数据集的线程数，根据硬件运算能力酌情调整。 |
| --batch_size BATCH_SIZE | 否 | int | 32 | PyTorch 执行一次前向推理所使用的样本数量，根据内存或显存大小酌情调整。 |
| --dist_url DIST_URL | 否 | string | tcp://127.0.0.1:50011 | 初始化多卡蒸馏通信进程的方法。 |
| --distributed | 否 | / | / | 使用该参数表示进行多卡蒸馏，否则不进行多卡蒸馏。 |

若出现如下信息，则说明量化成功：

```none
2024-01-18 10:50:57,144 - INFO - [AMCT]:[Graph]: Doing whole model dump...
2024-01-18 10:50:57,413 - INFO - [AMCT]:[Utils]: The model file is saved in xxx/result/mobilenet_v2_fake_quant_model.onnx
[INFO] MobilenetV2 before quantize top1:   0.84375 top5:      0.95
[INFO] MobilenetV2 after quantize  top1:   0.83125 top5:    0.9375
```

说明：本示例结果与示例命令相匹配仅作为参考，不同的迭代次数精度不同，数据随机性可能导致结果微小差异。

### 1.3 蒸馏结果

蒸馏成功后，在当前目录会生成量化日志文件 ./amct_log/amct_pytorch.log 和 ./tmp 与 ./result 文件夹，该文件夹内包含以下内容：

+ tmp: 临时文件夹
  + distill_config.json: 蒸馏配置文件，描述了如何对模型中的每一层进行量化以及Block划分信息。
  + scale_offset_record.txt: 量化因子记录文件，记录量化因子。
+ result
  + mobilenet_v2_deploy_model.onnx: 蒸馏部署模型，即蒸馏后的可在昇腾 AI 处理器部署的模型文件。
  + mobilenet_v2_fake_quant_model.onnx: 蒸馏仿真模型，即蒸馏后的可在 ONNX 执行框架 ONNX Runtime 进行精度仿真的模型。