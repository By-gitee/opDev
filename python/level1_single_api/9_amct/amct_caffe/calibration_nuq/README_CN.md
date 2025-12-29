# AMCT_Caffe ResNet-50 示例

该示例包含 ResNet-50 的静态非均匀量化。

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

### 2.2 执行静态非均匀量化

#### 2.2.1 配置文件

执行静态非均匀量化需要配置简易配置文件的 `nuq_config ` 选项，本示例提供了已经配置好的简易配置文件 [quant.cfg](./src/nuq_files/quant.cfg)， 该配置文件供本示例提供的 ResNet-50 模型使用，若只执行本示例则无需进行额外修改。

若要使用其他模型，请参见《ATC工具指南》，《CANN 开发辅助工具指南》中的“ATC 工具使用指南”。将 3.2 章节中介绍的均匀量化后的部署模型 (.om) 转换成 json 文件，获取该文件，并修改 [quant.cfg](./src/nuq_files/quant.cfg) 文件引用该文件路径：

```json
nuq_config {
    mapping_file : "../src/nuq_files/resnet50_om_model.json"
    ...
    }
}
```

#### 2.2.2 执行

首先建议确保 2.1 章节中的原始网络推理验证成功。

如果使用支持 GPU 的设备，执行：

```bash
python3.7.5 ./src/ResNet50_sample.py \
--model_file model/ResNet-50-deploy.prototxt \
--weights_file model/ResNet-50-model.caffemodel \
--caffe_dir caffe-master \
--gpu 0 \
--cfg_define src/nuq_files/quant.cfg
```

或者仅使用CPU，执行：

```bash
python3.7.5 ./src/ResNet50_sample.py \
--model_file model/ResNet-50-deploy.prototxt \
--weights_file model/ResNet-50-model.caffemodel \
--caffe_dir caffe-master \
--cpu \
--cfg_define src/nuq_files/quant.cfg
```

入参说明:

* `model_file`: 必填。Caffe 模型文件 (.prototxt) 路径。
* `weights_file`: 必填。Caffe 权重文件 (.caffemodel) 路径。
* `caffe_dir`: 必填。Caffe源 代码路径，支持相对路径和绝对路径。
* `gpu`/`cpu`: 可选。指定推理时使用 CPU 或则使用的 GPU 设备 ID。
* `cfg_define`: 必填。非均匀量化简易配置文件路径。

