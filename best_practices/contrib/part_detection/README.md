# 基于人工智能的零件表面质量检测算法

- [基于人工智能的零件表面质量检测算法](#基于人工智能的零件表面质量检测算法)
- [概述](#概述)
  - [简述](#简述)
- [NPU910-训练指导](#npu910-训练指导)
  - [准备训练环境](#准备训练环境)
    - [准备环境](#准备环境)
  - [参考样例的数据集下载](#参考样例的数据集下载)
  - [训练模型](#训练模型)
- [910训练结果展示](#910训练结果展示)
- [NPU310-模型离线推理指导](#npu310-模型离线推理指导)
  - [推理环境准备](#推理环境准备)
  - [快速上手](#快速上手)
  - [参考样例的验证集下载](#参考样例的验证集下载)
  - [模型推理](#模型推理)
    - [1 模型转换](#1-模型转换)
    - [2 开始推理验证](#2-开始推理验证)
        - [\* 如果有多卡推理的需求，请跳过该步骤，om\_val.py 该脚本不支持多卡推理](#-如果有多卡推理的需求请跳过该步骤om_valpy-该脚本不支持多卡推理)
- [310-可视化](#310-可视化)
  - [单张图片推理验证可视化](#单张图片推理验证可视化)
- [版本说明](#版本说明)
- [公网地址说明](#公网地址说明)
  - [变更](#变更)
- [FAQ](#faq)
  - [使用自制的数据集在NPU910A进行训练](#使用自制的数据集在npu910a进行训练)
  - [使用自制的数据集在NPU310进行离线推理可视化](#使用自制的数据集在npu310进行离线推理可视化)
    - [单张图像可视化](#单张图像可视化)
    - [多张图片推理验证可视化](#多张图片推理验证可视化)

# 概述

## 简述

本项目基于 YOLO 目标检测算法，YOLO（You Only Look Once）是一种经典的物体检测网络，将物体检测问题建模为回归问题。YOLO 可以在单个网络中进行训练和推理，它可以在一次推理中检测图像中的所有物体、确定其位置、类别以及相应的置信度。YOLOv5 于 2020 年 5 月 27 日首次发布。

- 参考实现：

  ```
  url=https://github.com/ultralytics/yolov5.git
  Tag=v6.0
  ```

# NPU910-训练指导

## 准备训练环境

### 准备环境

- 当前模型支持的 PyTorch 版本和已知三方库依赖如下表所示。

  **表 1** 版本支持表

  | Torch_Version | 三方库依赖版本 |
  | :-----------: | :------------: |
  |  PyTorch 1.8  | pillow==9.1.0  |

- 环境准备指导。

  请参考《[Pytorch 框架训练环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/ptes)》。

- 安装依赖。

  在模型源码包根目录下执行命令，安装模型对应 PyTorch 版本需要的依赖。

  ```
  pip3 install -r 1.8_requirements.txt       # PyTorch1.8版本
  ```

## 参考样例的数据集下载

（1）进入到和代码同级目录下

（2）下载我们提供的数据集

```bash
cd datasets
# 下载数据集，将解压后的文件放置在datasets目录中
wget https://obs-9be7.obs.myhuaweicloud.com/models/Detection_Model_for_PyTorch/mycoco-train.zip
unzip mycoco-train.zip
# 生成`train2017.txt`和`val2017.txt`
python3 create_txt.py --dataset_name mycoco1 
```

> **说明**
> 1. 目前，开源数据集的工业零件缺陷数据集，他们的图像分辨率普遍较低。本项目提供一种高分辨率工业零件数据集供用户参考。
> 2. 本项目的后续项目知道说明中，我们的数据集名称为`mycoco1`，相对路径为`./datasets/mycoco1`，数据集的信息说明存放于`./datasets/mycoco1.yaml`。

## 训练模型

1. 进入解压后的源码包根目录。

    ```
    cd ..
    ```

2. 运行训练脚本。

   - 获得与训练权重
     请根据您需要训练的模型，获得相应的预训练权重。

   在[链接](https://github.com/ultralytics/yolov5/releases)中找到`v6.0`版本下载的预训练权重，也可以使用下述命令下载。

   ```bash
   wget https://github.com/ultralytics/yolov5/releases/download/v6.0/${model}.pt       # 获得预训练权重
   ```

   - 命令参数说明：
     - `${model}`：模型大小，可选`yolov5s`，`yolov5m`，`yolov5l`。

   > **说明：**
   > 如果存在命令下载失败的情况，请前往链接下载对应的权重，并上传至项目的根目录。

   - 单机单卡训练

     - 启动单卡训练。

       ```bash
       python3 train.py --data ./datasets/mycoco1.yaml \
                         --workers 4 \
                         --batch-size 32 \
                         --epochs 300 \
                         --weights ./yolov5l.pt \
                         --cfg ./models/myyolov5l.yaml \
                         --device 0 \
                         --checkpoint yolov5l_mycoco1.pt
       ```

       > **训练说明：**
       > 1. 在上述指导说明中，我们使用`yolov5l.pt`预训练模型。
       > 2. 模型最终输出的训练权重为`yolov5l_mycoco1.pt`。
       > **训练修改说明：**
       > 1. 根据数据集的信息，修改`./models/yolov5l.yaml`中`nc: _`，并保存为`./models/myyolov5l.yaml`

     - 单机单卡验证

       ```bash
       python3 val.py --data ./datasets/mycoco1.yaml \
                      --batch-size 32 \
                      --weights ./yolov5l_mycoco1.pt \
                      --device 0 \
                      --conf-thres 0.01 \
                      --iou-thres 0.5
       ```

       > **验证说明：**
       > 1. `conf-thres`和`iou-thres`需要根据数据集特征进行相应的调整
       > 2. 由于 NMS 后处理等操作，会存在测试过程较为缓慢的情况
       
       模型训练脚本参数说明如下。
        ```
        公共参数：
        --data                              //数据集路径
        --workers                           //加载数据进程数
        --batch-size                        //训练批次大小
        --epochs                            //重复训练次数
        --weights                           //初始权重路径
        --rect                              //矩形训练
        --nosave                            //保存最后一个权重
        --noval                             //验证最后一个epoch
        --artifact_alias                    //数据集版本
        --save-period                       //权重保存
        --native_amp                        //使用torch amp进行混合精度训练，如不配置默认使用apex
        --half                              //eval执行脚本中参数，如配置默认使用混合精度计算
        --cfg                               //需要进行适当的修改（数据集内类别数量），以适合模型训练
        --checkpoint                        //保存的输出权重的名称
        ```

      训练完成后，权重文件保存在项目根目录下，通过验证输出模型训练精度和性能信息。

   - `pth`转化成`onnx`
      1. 转化环境准备，安装依赖
          ```
          git clone https://gitee.com/ascend/msadvisor.git
          cd msadvisor/auto-optimizer
          python3 -m pip install --upgrade pip
          python3 -m pip install wheel
          python3 -m pip install .
          cd ../..
          pip3 install -r onnx_requirements.txt
          ```

      2. 在 Ascend910A 环境下，启动模型转化。

         ```bash
         bash pth2onnx.sh --tag 6.0 \
                           --model yolov5l_mycoco1 \
                           --nms_mode nms_script
         ```
           - 命令参数说明：
             - `--model`： 为 pt 模型的名称

# 910训练结果展示

**表 2** 训练结果展示表

|  NAME   | device |   P   | mAP0.5 |  FPS  | AMP_Type | Torch_Version |
| :-----: | :----: | :---: | :----: | :---: | :------: | :-----------: |
| yolov5s | 1p-NPU |   -   |   -    |   -   |    O1    |      1.8      |
| yolov5m | 1p-NPU |  97   |  73.9  |   -   |    O1    |      1.8      |
| yolov5l | 1p-NPU |  97   |   99   |   -   |    O1    |      1.8      |

# NPU310-模型离线推理指导

## 推理环境准备

- 该模型需要以下插件与驱动  
  **表 1** 版本配套表

| 配套                                                              | 版本   | 环境准备指导                                                                                                                                          |
| ----------------------------------------------------------------- | ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| 固件与驱动                                                        | 22.0.4 | [Pytorch 框架推理环境准备](https://www.hiascend.com/document/detail/zh/ModelZoo/pytorchframework/pies)                                                |
| CANN                                                              | 6.0.0  | [推理应用开发学习文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/600alpha003/infacldevg/aclpythondevg/aclpythondevg_0000.html) |
| Python                                                            | 3.7.5  | -                                                                                                                                                     |
| PyTorch                                                           | 1.10.1 | -                                                                                                                                                     |
| 说明：Atlas 300I Duo 推理卡请以 CANN 版本选择实际固件与驱动版本。 | \      | \                                                                                                                                                     |

## 快速上手

1. 进入NPU310-模型离线推理目录
   
    ```bash
    cd inference
    ```

2. 安装推理环境依赖

   ```bash
   git clone https://gitee.com/ascend/msadvisor.git
   cd msadvisor/auto-optimizer
   python3 -m pip install --upgrade pip
   python3 -m pip install wheel
   python3 -m pip install .
   ```

   ```bash
   cd ../..
   pip3 install -r requirements.txt
   ```

## 参考样例的验证集下载

1. 目前已经进入到`inference`文件夹下
   
2. 下载我们提供的验证集

    ```bash
    cd datasets
    # 下载数据集，并解压
    wget https://obs-9be7.obs.myhuaweicloud.com/models/Detection_Model_for_PyTorch/mycoco-inference.zip  
    unzip mycoco-inference.zip
    ```

    参考样例的验证集目录结构如下所示：
    ```
    └── Detection_Model_for_PyTorch
        ├── inference
            ├── datasets
                ├── mycoco1                 # 下载的验证集文件夹          
                    ├── Annotations         # 测试集的VOC格式的标签（不能缺少）
                    └── val2017             # 测试集的图像
                ├── create_val2017txt.py    # 已存放在datasets文件夹下
                └── voc_annotations2json.py # 已存放在datasets文件夹下
    ```

    > **说明**
    > 在参考样例中，本项目样例存放了Annotations（VOC格式的标注文件），其用于生成instances_val2017.json，便于后续推理验证。

3. 生成特定的标注文件
    
    ```bash
    python3 create_val2017txt.py --dir_path ./mycoco1       # 创建txt文件
    python3 voc_annotations2json.py --xml_dir ./mycoco1/Annotations --output_path ./mycoco1/instances_val2017.json                                  # 创建json文件
    ```

    - 命令参数说明：
      - `--dir_path`：数据集文件夹路径
      - `--xml_dir`：Annotations文件夹路径
      - `--output_path`：生成本项目数据集`instances_val2017.json`的路径

## 模型推理

- **说明：`nms`后处理脚本（`nms_script`）**
 直接用官网`export.py`导出`onnx`模型，模型结构和官网一致，推理流程也和官方一致，NMS 后处理采用脚本实现。 \* 注意：如果使用的是 nms_script 方式，需要修改`model.yaml`文件，将其中的配置`conf_thres:0.4`和`iou_thres:0.5`修改为`conf_thres:0.001`和`iou_thres:0.6`，后续该方式下精度测试也是采用修改后的配置。

### 1 模型转换

我们已经在`NPU910A`上将模型权重文件`.pt`转换为`.onnx`文件。现在，在`NPU310`上使用`ATC`工具将`.onnx`文件转为离线推理模型`.om`文件。

1. 移动`ONNX`模型文件到`inference`目录
    1. 当通过训练获得`yolov5l_mycoco1.onnx`，请参考执行以下步骤：
      ```bash
      cd ../..      # 返回 `Detection_Model_for_PyTorch` 目录
      mv yolov5l_mycoco1.onnx ./inference
      cd inference
      ```
    2. 当未通过训练得到`yolov5l_mycoco1.onnx`，请参考执行以下步骤：
      ```bash
      cd ..         # 进入 inference
      wget https://obs-9be7.obs.myhuaweicloud.com/models/Detection_Model_for_PyTorch/yolov5l_mycoco1.onnx
      ```

    > **说明**
    > 请在`NPU910A`等设备上，将`pt`模型转化为`onnx`模型
    > 使用训练保存的模型文件，本项目后续演示采用为`yolov5l_mycoco1.onnx`

2. 使用`ATC`工具将`ONNX`模型转`OM`模型  
   2.1 配置环境变量

   ```
   source /usr/local/Ascend/ascend-toolkit/set_env.sh
   ```

   > **说明：**  
   >  该脚本中环境变量仅供参考，请以实际安装环境配置环境变量。详细介绍请参见《[CANN 开发辅助工具指南 \(推理\)](https://support.huawei.com/enterprise/zh/ascend-computing/cann-pid-251168373?category=developer-documents&subcategory=auxiliary-development-tools)》。

   3.2 执行命令查看芯片名称（`${soc_version}`）

   ```
   npu-smi info
   # 该设备芯片名为Ascend310 （自行替换）
   回显如下：
   +-------------------+-----------------+------------------------------------------------------+
   | NPU     Name      | Health          | Power(W)     Temp(C)           Hugepages-Usage(page) |
   | Chip    Device    | Bus-Id          | AICore(%)    Memory-Usage(MB)                        |
   +===================+=================+======================================================+
   | 112     310       | OK              | 15.8         42                0    / 0              |
   | 0       0         | 0000:07:00.0    | 0            2703 / 8192                             |
   +===================+=================+======================================================+
   | 128     310       | OK              | 15.4         43                0    / 0              |
   | 0       1         | 0000:08:00.0    | 0            2621 / 8192                             |
   +===================+=================+======================================================+
   ```

   3.3 导出`OM`模型  
   运行`onnx2om.sh`导出`OM`模型。

   - 数据集（mycoco1）

     ```bash
     bash onnx2om.sh --tag 6.0 --model yolov5l_mycoco1 --nms_mode nms_script --bs 4 --soc Ascend310
     ```

   - `atc`命令参数说明（参数见`onnx2om.sh`）：
     - `--model`：ONNX 模型文件
     - `--output`：输出的 OM 模型
     - `--framework`：5 代表 ONNX 模型
     - `--input_format`：输入数据的格式
     - `--input_shape`：输入数据的 shape
     - `--soc_version`：处理器型号
     - `--log`：日志级别
     - `--compression_optimize_conf`：模型量化配置，使用说明参考[该链接](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/600alpha003/infacldevg/atctool/atlasatc_16_0084.html)
     - `--enable_small_channel`：输入端 aipp 算子配置，使用说明参考[该链接](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/600alpha003/infacldevg/atctool/atlasatc_16_0081.html)
     - `--insert_op_conf`：输入端 aipp 算子配置，使用说明参考[该链接](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/600alpha003/infacldevg/atctool/atlasatc_16_0072.html)

### 2 开始推理验证

1. 安装`ais-infer`推理工具  
   `ais-infer`工具获取及使用方式请点击查看 [[ais_infer 推理工具使用文档](https://gitee.com/ascend/tools/tree/master/ais-bench_workload/tool/ais_infer)]

##### \* 如果有多卡推理的需求，请跳过该步骤，om_val.py 该脚本不支持多卡推理

2. 执行推理 & 精度验证  
   运行`om_val.py`推理 OM 模型，模型参数在[model.yaml](model.yaml)中设置，结果默认保存在`predictions.json`。

   - 数据集（mycoco1）
        ```bash
        python3 om_val.py --tag 6.0 \
                            --model=yolov5l_mycoco1_bs4.om \
                            --nms_mode nms_script \
                            --batch_size=4 \
                            --dataset Customised_dataset \
                            --visualization False \
                            --data_path=./datasets/mycoco1 \
                            --ground_truth_json ./datasets/mycoco1/instances_val2017.json \
                            --cfg_file ./model4customised_dataset.yaml
        ```
        > **说明**
        > 1. `yolov5l_mycoco1_bs4.om`为`yolov5l_mycoco1_bs4.onnx`转化的om模型
        > 2. 我们在`--cfg_file `设置`./model4customised_dataset.yaml`，对应本项目提供的测试集的推理参数。

   - 命令参数说明：
     - `--tag`：模型版本，`[6.0]`。
     - `--model`：om模型。
     - `--nms_mode`：模型推理方式，`[nms_script]`。
     - `--batch_size`: 模型推理 batch 大小，默认`4`。
     - `--cfg_file`：模型推理参数设置，默认读取文件[model.yaml](model.yaml)。
     - `--dataset`：模型的数据集，默认设置为`[Customised dataset]`，用户定义的数据集
     - `--visualization`：推理测试的图像是否可视化输出，默认`False`
    
# 310-可视化

## 单张图片推理验证可视化

此步骤不输出测试精度，仅仅进行相应检测结果的可视化。可视化输出的图片左图为原始图像，右图为标注出异常的检测图像。

1. 准备数据集，获取单张图片的数据集

   本项目提供的参考数据集已在上文解压到`./inference/datasets`，在`singleimg`内

2. 执行单张图像可视化操作

    ```bash
    python3 om_val.py --tag 6.0 \
                        --model=yolov5l_mycoco1_bs4.om \
                        --nms_mode nms_script \
                        --batch_size=4 \
                        --dataset Customised_dataset \
                        --visualization True \
                        --single_data True \
                        --data_path ./datasets/singleimg/ \
                        --cfg_file ./model4customised_dataset.yaml
   ```

   - 命令参数说明：
     - `--single_data`：数据集内是否为单张图像，默认`[True]`。

   最终，输出的单张可视化图像保存在`./inference/runs/detect/singleimg/dst_000xxx.jpg`。

## 多图推理验证可视化

1. 准备数据集
   
   采用上文已经下载的验证集：`./Detection_Model_for_PyTorch/inference/datasets/mycoco1`。

2. 执行多张图像可视化操作
   
   通过命令调用`om_val.py`脚本。
   ```bash
   python3 om_val.py --tag 6.0 \
                     --model=yolov5l_mycoco1_bs4.om \
                     --nms_mode nms_script \
                     --batch_size=4 \
                     --dataset Customised_dataset \
                     --visualization True \
                     --single_data False \
                     --data_path ./datasets/mycoco1/ \
                     --ground_truth_json ./datasets/mycoco1/instances_val2017.json \
                     --cfg_file ./model4customised_dataset.yaml
   ```

   - 命令参数说明：
     - `--single_data`：数据集内是否为单张图像，默认`[False]`

   最终，输出的可视化图像保存在`./inference/runs/detect/multi_imgs/`。

# 版本说明

# 公网地址说明

代码涉及公网地址参考 [public_address_statement.md](./public_address_statement.md)

## 变更


# FAQ

## 使用自制的数据集在NPU910A进行训练

1. 数据集修改。
   
   1. 整理VOC格式数据集
      如果您标注的数据集格式是VOC格式，请根据以下步骤进行操作，将VOC格式数据集转化成yolo格式。

      VOC数据集目录结构参考如下所示：

      ```
      └── Detection_Model_for_PyTorch
          ├──datasets
              ├── VOCdevkit
                  ├── VOC2012
                      ├── Annotations   # 数据集VOC格式标注文件
                          ├── ***.xml
                      ├── JPEGImages    # 数据集图片
                      └── classes.txt   # 标注文件内的所有类别（要一致，注意大小写）
              ├── voc_annotation_to_yolo.py
              └── create_txt.py
      ```

      本项目，所提供的参考`classes.txt`内的内容为：
      ```
      Plug_defect
      Screw_defect
      fan_defect
      fan_screw_missing
      Screw_tilt_defect
      Screw_type_defect_1
      Screw_type_defect_2
      ```
      若不存在`classes.txt`，请按照格式创建并复制该内容。

    2. VOC格式数据集转化成yolo标注文件

        ```bash
        python3 voc_annotation_to_yolo.py
        ```
        生成yolo格式标签，存储于`./Detection_Model_for_PyTorch/datasets/VOCdevkit/VOC2012/yolo`

    3. 整理yolo格式数据集
      用户自行将数据集整理，数据集目录结构参考如下所示：

        ```
        └── Detection_Model_for_PyTorch
              ├──datasets
                  ├── mycoco1
                      ├── images
                            ├── train2017   # 训练集图片
                            └── val2017     # 测试集图片
                      └── labels
                            ├── train2017   # 训练集标签
                            └── val2017     # 测试集标签
                      ├── train2017.txt     # 可以后续根据数据集结构生成
                      └── val2017.txt       # 可以后续根据数据集结构生成
                  └── mycoco1.yaml          # 请根据您数据集的信息，进行相应的修改
        ```

2. 修改`mycoco1.yaml`。
   
    - 修改 `train、val` 路径
    - 修改`nc`
    - 修改`names`，填入标签类别

3. 生成 yolo中`train2017.txt` 和 `val2017.txt`文件。

   （1）运行 create_txt.py。

      ```bash
      cd ./datasets
      python3 create_txt.py --dataset_name mycoco1
      ```
      - 参数说明
        - `--dataset_name mycoco1`: 为数据集名称。（本项目的验证集参考名称为 `mycoco1`）

   （2）运行上述脚本后，将在 `mycoco1` 验证集**根目录**生成 `train2017.txt` 和 `val2017.txt`。

## 使用自制的数据集在NPU310进行离线推理可视化

### 单张图像可视化

1. 修改数据集
   
   `om_val.py`脚本是具备单张图片推理功能的，但需要根据以下格式，创建你的单张图片可视化数据集，修改为：

   ```
    └── Detection_Model_for_PyTorch
        ├──inference
            ├──datasets
                ├── singleimg                        (此处数据集名称随便取)
                    ├── val2017                      (此处文件夹名称不能修改)
                        └── 00000000100.jpg          (这里放你需要进行推理的单张图片)
                    └── val2017.txt                  (txt文件写入所要处理图片的绝对路径)
   ```
2. 创建 `val2017.txt`
    
    ```bash
    cd datasets
    python3 create_val2017txt.py --dir_path ./singleimg
    ```

   `val2017.txt`中存储验证图像的相对路径，可能为：
   ```
   ./val2017/00000000100.jpg
   ```

   - 根据自制数据集的信息，修改`model4customised_dataset.yaml`文件中的 `class_num`。

3. 修改 `om_val.py` 脚本

   1. 在`om_val.py`中，需要根据自己的数据集内类别名称`class_names`进行修改。
        例如本项目中的`class_names`为：
        ```python
        # line 82
        class_names = ['plug_defect', 'screw_defect', 'fan_defect', 'fan_screw_missing',
                'screw_tilt_defect', 'screw_type_defect_1', 'screw_type_defect_2']
        ```

   2. 执行单张图像可视化操作

        ```bash
        python3 om_val.py --tag 6.0 \
                            --model                                       # 写入你的om模型
                            --nms_mode nms_script \
                            --batch_size=4 \
                            --dataset Customised_dataset \
                            --visualization True \
                            --single_data True \
                            --data_path ./datasets/singleimg/ \           # 如果数据集名称不是singleimg，请修改
                            --cfg_file  ./model4customised_dataset.yaml   # 根据您的数据集修改model4customised_dataset.yaml
        ```

   最终，输出的可视化图像保存在`./inference/runs/detect/singleimg/dst_0000xxx.jpg`。

### 多张图片推理验证可视化

1. 修改数据集
   `om_val.py`脚本是具备多张图片推理功能的，但需要根据以下格式，创建你的多张图片可视化数据集，修改为：

   ```
    └── Detection_Model_for_PyTorch
        ├──inference
            ├──datasets
                ├── mycoco1                         (此处数据集名称随便取)
                    ├── Annotations                 (此处文件夹名称不能修改)
                    ├── val2017                     (此处文件夹名称不能修改)
                        ├── 00000000001.jpg
                        ├── ....
                        ├── 00000001002.jpg
                        └── 00000001003.jpg         (这里放你需要进行推理的图片)
                    ├── instances_val2017.json      (标注文件)
                    └── val2017.txt                 (txt文件写入所要处理图片的绝对路径)
   ```

2. 创建 `val2017.txt`
    
    ```bash
    cd datasets
    python3 create_val2017txt.py --dir_path ./mycoco1
    ```

    `val2017.txt`中存储验证图像的相对路径，可能为：
    ```
    ./val2017/00000000001.jpg
    .....
    ./val2017/00000001002.jpg
    ./val2017/00000001003.jpg
    ```

3. 本项目`instance_val2017.json`由`Annotations`生成
    
    进入`./inference/datasets`。

    ```bash
    python3 create_val2017txt.py --dir_path ./mycoco1                           
    python3 voc_annotations2json.py --xml_dir ./mycoco1/Annotations --output_path ./mycoco1/instances_val2017.json
    ```
    
    - 根据您数据集的信息，需要修改`./inference/model4customised_dataset.yaml`文件中的 `class_num`。

4. 修改 om_val.py 脚本

   1. 在`om_val.py`中，需要根据自己的数据集内类别名称`class_names`进行修改。
        例如本项目中的`class_names`为：

        ```python
        # line 82
        class_names = ['plug_defect', 'screw_defect', 'fan_defect', 'fan_screw_missing',
                'screw_tilt_defect', 'screw_type_defect_1', 'screw_type_defect_2']
        ```

   2. 执行多张图像可视化操作
      
      通过命令调用`om_val.py`脚本。
      ```bash
      python3 om_val.py --tag 6.0 \
                        --model=yolov5l_mycoco1_bs4.om \
                        --nms_mode nms_script \
                        --batch_size=4 \
                        --dataset Customised_dataset \
                        --visualization True \
                        --single_data False \
                        --data_path ./datasets/mycoco1/ \
                        --ground_truth_json ./datasets/mycoco1/instances_val2017.json \
                        --cfg_file ./model4customised_dataset.yaml
      ```

   最终，输出的可视化图像保存在`./inference/runs/detect/multi_imgs/`。