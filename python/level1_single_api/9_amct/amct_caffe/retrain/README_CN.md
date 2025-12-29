# AMCT_Caffe ResNet-50 示例

该示例包含 ResNet-50 的量化感知训练示例。

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

若执行成功，将会下载 `ResNet-50-deploy.prototxt`, `ResNet-50_retrain.prototxt` 与 `ResNet-50-model.caffemodel` 到该路径下。

若用户环境无法连接网络。则请先手动下载相应文件上传到创建的 model 路径下：

```bash
wget https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/amct_caffe/ResNet-50-deploy.prototxt
wget https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/amct_caffe/ResNet-50_retrain.prototxt
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

### 2.2 执行量化感知训练示例

在该示例根目录下执行：

```bash
python3.7.5 ./src/ResNet50_retrain.py \
--model_file model/ResNet-50_retrain.prototxt \
--weights_file model/ResNet-50-model.caffemodel \
--gpu 0 \
--caffe_dir caffe-master \
--train_data  caffe-master/examples/imagenet/ilsvrc12_train_lmdb \
--test_data caffe-master/examples/imagenet/ilsvrc12_val_lmdb
```

入参说明:

* `model_file`: 必填。Caffe 模型文件 (.prototxt) 路径。
* `weights_file`: 必填。Caffe 权重文件 (.caffemodel) 路径。
* `caffe_dir`: 必填。Caffe 源代码路径，支持相对路径和绝对路径。
* `gpu`: 可选。指定使用的 GPU 设备 ID。
* `train_data`: 必填。训练数据集路径。
* `dataset`: 必填。验证数据集路径。
* `train_batch`: 选填。一次训练迭代中的样本数。默认值为32。
* `train_iter`: 选填。重训练迭代次数。默认值为1000。
* `test_batch`: 选填。一次测试迭代中的样本数。默认值为1。
* `test_iter`: 选填。测试迭代次数。默认值为500。

```
注意：模型在CPU上进行重训练耗时长，训练中无日志打印。默认train_iter数值为1000的场景，训练时长约50小时。
建议在GPU上运行重训练流程，如果需要在CPU上调用全流程，请尝试将--train_iter调小。
```

若出现如下信息则说明重训练成功（如下 top1，top5 的推理精度只是样例，请以实际环境量化训练结果为准）：

```none
Network initialization done.
...
Top 1 accuracy = 0.688
Top 5 accuracy = 0.934
```

重训练成功后，在该示例根目录下生成重训练日志文件夹 amct_log，重训练中间结果文件夹 tmp，重训练结果文件所在文件夹 results（对该模型重新进行重训练时，如下结果文件将会被覆盖）：

* `amct_log`：记录了工具的日志信息，包括重训练过程的日志信息 amct_caffe.log。
* `results/retrain_results`：重训练结果文件，包括重训练后的模型文件、权重文件，如下所示：
  * `retrain_atc_model.prototxt`：重训练后的可在昇腾 AI 处理器部署的模型文件。
  * `retrain_deploy_model.prototxt`：重训练后的部署模型文件。
  * `retrain_deploy_weights.caffemodel`：重训练后的可在昇腾 AI 处理器部署的权重文件。
  * `retrain_fake_quant_model.prototxt`：重训练后的可在 Caffe 环境进行精度仿真模型文件。
  * `retrain_fake_quant_weights.caffemodel`：重训练后的可在 Caffe 环境进行精度仿真权重文件。
