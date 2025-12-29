# 快速入门
在本节中，您可以通过一个简单的图片分类应用了解使用AscendCL接口开发应用的基本过程以及开发过程中涉及的关键概念。

## 什么是图片分类应用？

“图片分类应用”，从名称上，我们也能直观地看出它的作用：按图片所属的类别来区分图片。

![输入图片说明](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/resource/pyacl_resnet50_firstapp.png)

但“图片分类应用”是怎么做到这一点的呢？当然得先有一个能做到图片分类的模型，我们可以直接使用一些训练好的开源模型，也可以基于开源模型的源码进行修改、重新训练，还可以自己基于算法、框架构建适合自己的模型。

鉴于当前我们是入门内容，此处我们直接获取已训练好的开源模型，毕竟这种最简单、最快。此处我们选择的是ONNX框架的ResNet-50模型。

ResNet-50模型的基本介绍如下：

-   输入数据：RGB格式、224\*224分辨率的输入图片
-   输出数据：图片的类别标签及其对应置信度

> **说明：** 
> -   置信度是指图片所属某个类别可能性。
> -   类别标签和类别的对应关系与训练模型时使用的数据集有关，需要查阅对应数据集的标签及类别的对应关系。

## 环境要求

-   操作系统及架构：CentOS 7.6 x86\_64、CentOS aarch64、Ubuntu 18.04 x86\_64、EulerOS x86、EulerOS aarch64
-   芯片：Atlas 200/300/500 推理产品、Atlas 推理系列产品、Atlas 训练系列产品
-   python及依赖的库：python3.7.5以上，Pillow、Numpy库
-   已在环境上部署昇腾AI软件栈，并配置对应的的环境变量，请参见[Link](https://www.hiascend.com/document/redirect/CannCommunityInstSoftware)中对应版本的CANN安装指南。  
    
    以下步骤中，开发环境指开发代码的环境，运行环境指运行算子、推理或训练等程序的环境，运行环境上必须带昇腾AI处理器。开发环境和运行环境可以合设在同一台服务器上，也可以分设。

## 下载样例

请选择其中一种样例下载方式：

-   压缩包方式下载（下载时间较短，但步骤稍微复杂）

    ```
    # 1. samples仓右上角选择 【克隆/下载】 下拉框并选择 【下载ZIP】。     
    # 2. 将ZIP包上传到开发环境中的普通用户家目录中，【例如：${HOME}/ascend-samples-master.zip】。      
    # 3. 开发环境中，执行以下命令，解压zip包。      
    cd ${HOME}     
    unzip ascend-samples-master.zip
    ```

    注：如果需要下载其它版本代码，请先请根据前置条件说明进行samples仓分支切换。

-   命令行方式下载（下载时间较长，但步骤简单）

    ```
    # 开发环境，非root用户命令行中执行以下命令下载源码仓。    
    cd ${HOME}     
    git clone https://gitee.com/ascend/samples.git
    ```

    注：如果需要切换到其它tag版本，以v0.5.0为例，可执行以下命令。

    ```
    git checkout v0.5.0
    ```

下载成功后，切换到“ <SAMPLE_DIR>/python/level2_simple_inference/1_classification/resnet50_firstapp”目录下，查看该样例的目录结构，**下文所有的操作步骤均需先切换到resnet50_firstapp目录**：

```
resnet50_firstapp
├── data                                // 用于存放测试图片的目录
├── model                               // 用于存放模型文件的目录                 
├── src
│   ├── constant.py                     // 常量定义文件
│   └── firstapp.py                     // 图片分类样例的运行文件
```

## 准备模型

1.  以运行用户登录开发环境。

2.  下载模型数据。

    执行以下命令，将ONNX模型下载至“model”目录下，命令中的“***<SAMPLE_DIR>***”请根据实际样例包的存放目录替换
    ```
    cd <SAMPLE_DIR>/python/level2_simple_inference/1_classification/resnet50_firstapp/model
    wget https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/resnet50/resnet50.onnx
    ```

3.  执行模型转换。

    执行以下命令（以 Atlas 推理系列产品为例），将原始模型转换为昇腾AI处理器能识别的\*.om模型文件。请注意，执行命令的用户需具有命令中相关路径的可读、可写权限。以下命令中的“***<soc_version>***”请根据实际昇腾AI处理器版本替换。

    ```
    atc --model=resnet50.onnx --framework=5 --output=resnet50 --input_shape="actual_input_1:1,3,224,224"  --soc_version=<soc_version>
    ```
    
    -   --model：ResNet-50网络的模型文件路径。
    -   --framework：原始框架类型。5表示ONNX。
    -   --output：resnet50.om模型文件的路径。若此处修改模型文件名及存储路径，则需要同步修改src/firstapp.py中模型加载处的模型文件名及存储路径，即model_path变量值。
    -   --soc\_version：昇腾AI处理器的版本。
    
    关于各参数的详细解释，请参见[《ATC离线模型编译工具》](https://www.hiascend.com/document/redirect/AscendTensorCompiler)。

## 准备测试图片

本次样例需要使用两张动物图片，请执行以下命令将图片下载至“data”目录，或通过以下链接获取后放至“data”目录。若此处修改测试图片文件名，则需要同步修改src/firstapp.py中读取图片处的文件名，即image_paths变量值。

-   [测试图片1](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/models/aclsample/dog1_1024_683.jpg)

    ```
    cd $HOME/first_app/data
    wget https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/models/aclsample/dog1_1024_683.jpg
    ```

-   [测试图片2](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/models/aclsample/dog2_1024_683.jpg)

    ```
    cd $HOME/first_app/data
    wget https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/models/aclsample/dog2_1024_683.jpg
    ```

## 运行应用
以运行用户将resnet50_firstapp目录放至运行环境，以运行用户登录运行环境，切换到resnet50_firstapp目录下，检查环境变量配置是否正确，执行以下命令。

```
python3 src/firstapp.py
```
可以得到如下输出，分别为两张测试图片的top5分类信息。

其中[161]: 0.810220表示的是类别标识索引“161”的置信度为“0.810220”。

```
======== top5 inference results: =============
[161]: 0.810220
[162]: 0.103008
[178]: 0.017485
[166]: 0.013941
[212]: 0.009581
======== top5 inference results: =============
[267]: 0.728255
[266]: 0.101687
[265]: 0.100111
[151]: 0.004214
[160]: 0.002731
```

>**说明：** 
>类别标签和类别的对应关系与训练模型时使用的数据集有关，本样例使用的模型是基于imagenet数据集进行训练的，您可以在互联网上查阅对应数据集的标签及类别的对应关系。
>
>当前屏显信息中的类别标识与类别的对应关系如下：
>
>"161": ["basset", "basset hound"]
>
>"162": ["beagle"]
>
>"163": ["bloodhound", "sleuthhound"]
>
>"166": ["Walker hound", "Walker foxhound"]
>
>"167": ["English foxhound"]