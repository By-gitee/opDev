# AMCT_Caffe ResNet-50 示例

该示例包含 ResNet-50 的执行 convert_model 接口量化示例。

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

### 1.3 示例数据集下载

校准集用来产生量化因子，保证精度。计算量化参数的过程被称为“校准 (calibration)”。校准过程需要使用一部分图片来针对性计算量化参数，使用一个或多个 batch 对量化后的网络模型进行推理即可完成校准。为了保证量化精度，校准集与测试精度的数据集来源应一致。

该示例提供了一组样例校准集用于量化校准，可切换到该样例根目录下执行以下命令获取校准数据与标签：

```bash
cd data
wget https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/models/amct_caffe/imagenet_calibration.tar.gz
tar -xvf imagenet_calibration.tar.gz
```

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

### 2.2 执行 convert_model 接口量化示例

通过 convert_model 接口用户可以基于自己计算得到的量化因子以及 Caffe 模型，生成可以在昇腾 AI 处理器上做在线推理的部署模型和可以在 Caffe 环境下运行精度仿真的 Fakequant 模型。需提供自己得到的量化参数记录文件, 请参见 `model/record.txt` 进行配置。

在该示例根目录执行如下命令转换 ResNet-50 网络模型:

```bash
python3.7.5 ./src/convert_model.py \
--model_file model/ResNet-50-deploy.prototxt \
--weights_file model/ResNet-50-model.caffemodel \
--record_file model/record.txt \
--gpu 0 \
--caffe_dir caffe-master
```

入参说明:

* `model_file`: 必填。Caffe 模型文件 (.prototxt) 路径。
* `weights_file`: 必填。Caffe 权重文件 (.caffemodel) 路径。
* `record_file`: 必填，量化因子记录文件 (.txt) 路径。
* `caffe_dir`: 必填。Caffe 源代码路径，支持相对路径和绝对路径。
* `gpu`: 可选。指定推理时使用 GPU 设备 ID; 若没有可以使用的 GPU, 可忽略此项。

若出现如下信息则说明模型量化成功（如下 top1，top5 的推理精度只是样例，请以实际环境量化结果为准）：

```none
******final top1:0.86875
******final top5:0.95625
[AMCT][INFO]Run ResNet-50 with quantize success!
```

