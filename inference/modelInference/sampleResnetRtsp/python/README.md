## 目录

  - [样例介绍](#样例介绍)
  - [获取源码包](#获取源码包) 
  - [第三方依赖安装](#第三方依赖安装)
  - [样例运行](#样例运行)
  - [其他资源](#其他资源)
  - [更新说明](#更新说明)
  - [已知issue](#已知issue)
    
## 样例介绍

使用live555创建rtsp流，作为样例的输入数据，使能Acllite解码rtsp，缩放图片，使用ResNet50网络执行推理，最终对输入的rtsp流的每帧图片进行分类并且给出TOP1类别置信度和相应的类别信息。
 
样例输入：rtsp流。    
样例输出：打屏显示置信度TOP1的类别标识、置信度信息和相应的类别信息。

## 获取源码包
    
 可以使用以下两种方式下载，请选择其中一种进行源码准备。

 - 命令行方式下载（下载时间较长，但步骤简单）。

   ```    
   # 开发环境，非root用户命令行中执行以下命令下载源码仓。    
   cd ${HOME}     
   git clone https://gitee.com/ascend/samples.git
   ```
   **注：如果需要切换到其它tag版本，以v0.5.0为例，可执行以下命令。**
   ```
   git checkout v0.5.0
   ```   
 - 压缩包方式下载（下载时间较短，但步骤稍微复杂）。   
   **注：如果需要下载其它版本代码，请先请根据前置条件说明进行samples仓分支切换。**   
   ``` 
   # 1. samples仓右上角选择 【克隆/下载】 下拉框并选择 【下载ZIP】。    
   # 2. 将ZIP包上传到开发环境中的普通用户家目录中，【例如：${HOME}/ascend-samples-master.zip】。     
   # 3. 开发环境中，执行以下命令，解压zip包。     
   cd ${HOME}    
   unzip ascend-samples-master.zip
   ```

## 第三方依赖安装
  设置环境变量，配置程序编译依赖的头文件与库文件路径。“$HOME/Ascend”请替换“Ascend-cann-toolkit”包的实际安装路径。
   ```   
   echo 'export THIRDPART_PATH=$HOME/Ascend/ascend-toolkit/latest/thirdpart'>> ~/.bashrc    
   echo 'export PYTHONPATH=${THIRDPART_PATH}/python:$PYTHONPATH'>> ~/.bashrc
   ```

   执行以下命令使环境变量生效并创建文件夹
   
   ```     
    source ~/.bashrc
    mkdir -p ${THIRDPART_PATH}
   ```

   python-acllite库以源码方式提供，安装时将acllite目录拷贝到运行环境的第三方库目录

   ```
    cp -r ${HOME}/samples/inference/acllite/python ${THIRDPART_PATH}

   ```   

- numpy

  执行以下命令安装numpy库。
  ```
  pip3 install numpy
  ``` 
  
- live555

    ```
    # 下载live555
    wget https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/models/sampleResnetRtsp/live.2023.05.10.tar.gz --no-check-certificate
    tar -zxvf live.2023.05.10.tar.gz
    cd live/
    ./genMakefiles linux
    make -j8
    ``` 
    
    
## 样例运行

  - 数据准备

    重新打开一个窗口进入/live/testProgs目录
    

    ```
    wget https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/models/sampleResnetRtsp/test.264 --no-check-certificate
    ```

    获取rtsp流地址
    
     ```
    ./testOnDemandRTSPServer
     ```

    即可创建rtsp流如下图所示

    ![输入图片说明](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/models/sampleResnetRtsp/rtsp.jpg "image-20211028101534905.png")
    

    获取rtsp流地址如上图黄色方框所示，例如
    rtsp://192.168.0.163:8554/h264ESVideoTest
     

    

  - ATC模型转换

    将ResNet-50原始模型转换为适配昇腾310处理器的离线模型（\*.om文件），放在model路径下。

    ```
    # 为了方便下载，在这里直接给出原始模型下载及模型转换命令,atc指令中的参数--soc_version可替换为使用设备的型号。
    cd $HOME/samples/inference/modelInference/sampleResnetRtsp/python/model
    wget https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/resnet50/resnet50.onnx --no-check-certificate
    wget https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/resnet50/resnet50_DVPP/aipp.cfg --no-check-certificate
    atc --model=resnet50.onnx --framework=5 --output=resnet50 --input_shape="actual_input_1:1,3,224,224"  --soc_version=Ascend310  --insert_op_conf=aipp.cfg
    ```

  - 样例运行

    执行运行脚本，开始样例运行。
    **将rtsp流地址换成上实际的rtsp流地址** 
    ```
    cd $HOME/samples/inference/modelInference/sampleResnetRtsp/python/scripts
    bash sample_run.sh
    ```
  - 样例结果展示
    
    执行成功后，在屏幕上的关键提示信息示例如下，提示信息中的top1表示图片置信度的前1种类别、index表示类别标识、value表示该分类的最大置信度，class表示所属类别。这些值可能会根据版本、环境有所不同，请以实际情况为准：

    ![输入图片说明](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/models/sampleResnetRtsp/out.png "image-20211028101534906.png")

## 其他资源

以下资源提供了对ONNX项目和Renet50模型的更深入理解：

**ONNX**
- [GitHub: ONNX](https://github.com/onnx/onnx)

**Models**
- [Resnet50 - image classification](https://gitee.com/ascend/ModelZoo-PyTorch/tree/master/ACL_PyTorch/built-in/cv/Resnet50_Pytorch_Infer)

**Documentation**
- [AscendCL Samples介绍](../README_CN.md)
- [使用AscendCLC API库开发深度神经网络应用](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/600alpha006/infacldevg/aclcppdevg/aclcppdevg_000000.html)
- [昇腾文档](https://www.hiascend.com/document?tag=community-developer)

## 更新说明
  | 时间 | 更新事项 |
|----|------|
| 2023/08/10 | 新增sampleResnetRtsp/python/README.md |
  

## 已知issue

  暂无
