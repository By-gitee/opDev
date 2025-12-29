# AMCT_Caffe ResNet-50 示例

该示例包含 ResNet-50 的自动非均匀量化。

## 1， 准备工作

### 1.1 AMCT_Caffe环境

执行本用例需要配置 AMCT_Caffe python 环境，详细环境准备流程可以参照[昇腾社区开发者文档](https://ascend.huawei.com/zh/#/document?tag=developer)页面下“全流程开发工具链” 下的 “模型压缩 (Caffe)” 文档进行配置。

### 1.2 原始模型 (Pre-trained model)

本样例依赖指定的原始模型定义文件 (.prototxt) 与预训练权重文件 (.caffemodel)。通过执行以下命令下载 .prototxt 模型文件与 .caffemodel 权重文件：

```bash
python3.7.5 ./src/download_models.py  \
--caffe_dir CAFFE_DIR \ # 必填。Caffe源代码路径，支持相对路径和绝对路径。
--close_certificate_verify #可选。关闭证书验证参数，确保模型正常下载。
```

若执行成功，将会下载 `ResNet-50-deploy.prototxt`与 `ResNet-50-model.caffemodel` 到该路径下。

若用户环境无法连接网络。则请先手动下载相应文件上传到创建的 model 路径下：

```bash
wget https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/amct_caffe/ResNet-50-deploy.prototxt
wget https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/resnet_50/ResNet-50-model.caffemodel
```

### 1.3 数据集

#### 1.3.1 准备 ImageNet LMDB 数据集

自动量化回退过程中，需要不断对模型进行校准和测试，因此需要准备数据集，本示例所使用的数据集为 LMDB 格式的 ImageNet 数据集，关于数据集的下载以及制作请参见 Caffe 工程caffe-master/examples/imagenet/readme.md文件或者参见 [Caffe 官方 Github 链接](https://github.com/BVLC/caffe/tree/master/examples/imagenet)。

## 2， 执行样例

### 2.1 原始网络验证

在执行量化前，可对原始网络模型进行预测试，检测原始模型是否可以在 Caffe 环境中正常运行。以避免数据集和模型不匹配、模型无法在 Caffe 环境中执行等问题:

```bash
python3.7.5 ./src/ResNet50_sample.py \
--model_file model/ResNet-50-deploy.prototxt \
--weights_file model/ResNet-50-model.caffemodel \
--caffe_dir caffe-master \
--gpu 0 \
--pre_test 
```

入参说明:

* `model_file`: 必填。Caffe 模型文件 (.prototxt) 路径。
* `weights_file`: 必填。Caffe 权重文件 (.caffemodel) 路径。
* `caffe_dir`: 必填。Caffe 源代码路径，支持相对路径和绝对路径。
* `gpu`: 可选。指定推理时使用 GPU 设备 ID; 若没有可以使用的 GPU, 可忽略此项。  
* `pre_test`: 可选。对量化之前的模型进行预测试，并给出推理结果，用于测试原始模型是否可以在 Caffe 环境中正常运行。

若出现如下信息，则说明原始模型在Caffe环境中运行正常

```none
[AMCT][INFO]Run ResNet-50 without quantize success!
```

### 2.2 执行自动非均匀量化

自动非均匀量化是 AMCT_CAFFE 在满足给出的精度损失要求的前提下自动搜索出一个量化配置，使量化后的模型压缩率更高的功能特性。用户需要基于给出的自动非均匀量化基类 `AutoNuqEvaluatorBase` 实现一个子类，需要实现 `eval_model(self, model_file, weight_file, batch_num)` 和 `is_satisfied(self, original_metric, new_metric)` 两个方法：

1. eval_model: 根据输入的模型和 batch 数量 (batch_num) 进行数据前处理、模型推理、数据后处理，得到模型的评估结果，要求返回的评估结果唯一，如分类网络的 top1，检测网络的 mAP 等，同时也可以是指标加权的结果。

2. is_satisfied: 用于判断量化后的模型是否达到了精度损失的要求，如果达到了则需返回 True，否则返回 False。

详细请参照`./src/auto_nuq_resnet50_sample.py`中的`AutoNuqEvaluator`类实现方式。

#### 2.2.1 配置文件

配置文件与静态非均匀量化一致，请参考 2.4.1 章节进行相关配置。

#### 2.2.2 执行

同样，自动非均匀量化有使用 `.src/auto_nuq_resnet50_sample.py` 量化脚本执行，与使用 bash 脚本 `.scripts/run_resnet50_auto_nuq.sh` 执行两种方式。后者基于前者封装而成，配置参数较少，用户请根据实际情况选择一种方式执行。

通过 `.src/auto_nuq_resnet50_sample.py` 执行：

```bash
python3.7.5 ./src/auto_nuq_resnet50_sample.py \
--model_file model/ResNet-50-deploy.prototxt \
--weights_file model/ResNet-50-model.caffemodel \
--caffe_dir /path/to/caffe-master \
--gpu 0 \
--cfg_define src/nuq_files/quant.cfg \
--dataset caffe-master/examples/imagenet/ilsvrc12_val_lmdb
```

入参说明:

* `model_file`: 必填。Caffe 模型文件 (.prototxt) 路径。
* `weights_file`: 必填。Caffe 权重文件 (.caffemodel) 路径。
* `caffe_dir`: 必填。Caffe 源代码 caffe-master 路径，支持相对路径和绝对路径。
* `gpu`/`cpu`: 可选。指定推理时使用 CPU 或则使用的 GPU 设备 ID。
* `cfg_define`: 必填。非均匀量化简易配置文件路径。
* `dataset`: 必填。用于测试目标精度的验证集路径。

或则通过 `.scripts/run_resnet50_auto_nuq.sh` 执行：

```bash
bash scripts/run_resnet50_auto_nuq.sh \
-c /path/to/caffe-master
-g 0 \
-d /path/to/validation/data/set
```

入参说明:

* `-c`: 必填，caffe-master 路径
* `-g`: 选填，GPU 的设备 ID。如果不指定该参数，则默认在 CPU 上运行
* `-d`: 必填，验证数据集路径

