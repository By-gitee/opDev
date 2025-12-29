# ResNet-50 量化 sample 指导

amct_mindspore 工具 sample 运行指导

## 环境要求

1. Ascend 910 host 环境；

2. 正确安装了 mindspore ascend 版本和与之配套的 amct_mindspore 工具；

## 使用方法

### 1. 准备 resnet50 预训练 checkpoint 

你可以在mindspore的官网下载 ResNet50 的预训练 checkpoint 文件，训练基于 CIFAR-10 dataset；

[resnet50 mindspore pretrain ckpt download page](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/amct_mindspore/resnet50_224_new.ckpt)

将 resnet50_224_new.ckpt 文件放到 model 目录；

### 2. 准备 CIFAR-10 数据集

Dataset used: [CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html)

- 数据集大小：60,000 32*32 10种类的彩色图片
  - Train：50,000 images
  - Test： 10,000 images
  
- 数据集格式：二进制文件

- 下载了 cifar-10 数据集后，进行一下的处理：

  ```bash
  tar -zxvf cifar-10-binary.tar.gz
  mkdir cifar-10-verify-bin
  cp cifar-10-batches-bin/batches.meta.txt ./cifar-10-verify-bin/
  cp cifar-10-batches-bin/readme.html ./cifar-10-verify-bin/
  cp cifar-10-batches-bin/test_batch.bin ./cifar-10-verify-bin/
  ```

  处理后的文件目录结构如下：

```none
├─cifar-10-batches-bin
│      batches.meta.txt
│      data_batch_1.bin
│      data_batch_2.bin
│      data_batch_3.bin
│      data_batch_4.bin
│      data_batch_5.bin
│      readme.html
│
└─cifar-10-verify-bin
        batches.meta.txt
        readme.html
        test_batch.bin
```

**文件说明：**

- scripts/download_files.py：用于下载模型定义文件及工具类脚本的处理脚本。
- src/resnet50_sample.py ：训练后量化脚本。
- README.md：量化示例使用说明文件。

### 3. 下载模型定义文件

  切换到 scripts 目录下，执行如下的命令下载模型定义文件：

  `python3.7.5 download_files.py --close_certificate_verify`

其中 --close_certificate_verify 参数可选，用于关闭证书验证，推荐在不关闭情况下下载，如果提示证书认证失败再添加该参数重新下载。

若出现如下的信息则说明下载成功：

```none
# resnet50 模型定义文件
[INFO]Download file to "../src/resnet.py" success.
# 数据与处理脚本，执行量化脚本时，会调用该脚本进行数据集的预处理
[INFO]Download file to "../src/dataset.py" success.
# 控制学习率脚本，量化感知训练过程会调用该文件的方法，控制学习率
[INFO]Download file to "../src/lr_generator.py" success.
# 分布式训练场景使用的脚本，用于生成 hccl 配置文件
[INFO]Download file to "../src/hccl_tools.py" success.
```

### 4. 训练后量化示例

1. 执行训练后量化：

```bash
cd src
python3 resnet50_sample.py --dataset_path your_dataset_path/cifar-10-verify-bin  --checkpoint_path your_resnet50_checkpoint_file_path

注：上述命令中的 your_dataset_path your_resnet50_checkpoint_file_path 分别对应实际环境的 cifar10 数据集路径和预训练 ckpt 文件路径；
```

参数解释：

表1 量化脚本所用参数说明

| 参数              | 说明                                           | 是否必填 |
| ----------------- | --------------------------------------------- | -------- |
| -h                | 显示帮助信息                                   | 否       |
| --dataset_path    | 数据集路径： 如 ../cifar-10-verify-bin         | 是       |
| --checkpoint_path | MindSpore 权重文件 resnet50_224_new.ckpt 路径  | 是       |

2. 提示如下信息则说明量化成功

```bash
INFO - [AMCT]:[QuantizeTool]：Generate AIR file : ../src/results/resnet50_quant.air success!
[INFO] the quantized AIR file has been stored at:
results/resnet50_quant.air
```

3. 量化结果说明

   量化成功后，在 Resnet50/src 目录下生成如下文件：

   - config.json: 量化配置文件，描述了如何对模型中的每一层进行量化。如果量化脚本所在目录下已经存在量化配置文件，则再次调用 create_quant_config 接口时，如果新生成的量化配置文件和已有的文件重名会覆盖已有的量化配置文件，否则重新生成量化配置文件。
   - amct_log/acmt_mindspore.log ：量化日志文件，记录了量化过程的日志信息。
   - results/resnet50_quant.air: 量化结果文件，可以通过 ATC 生成昇腾AI处理器上的可部署的模型文件。
   - kernel_meta: 算子编译生成的文件目录
   - （可选）amct_dump/calibration_record.txt : 如果执行量化时，设置了 export DUMP_AMCT_RECORD=1的环境变量，则在量化脚本的同级目录还会生成量化因子目录，该目录下的量化因子记录文件 calibration_record.txt，记录了每个量化层的量化因子。
