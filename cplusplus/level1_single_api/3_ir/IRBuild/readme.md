# 样例使用指导<a name="ZH-CN_TOPIC_0302914394"></a>

## 功能描述<a name="section5991635141815"></a>

本样例为IR模型构建样例，支持通过以下几种方式构建Graph，并生成适配昇腾AI处理器的离线模型，用户可任选其一。
               
-   通过算子原型构建Graph
-   将TensorFlow原始模型解析为Graph
-   将Caffe原始模型解析为Graph

## 目录结构<a name="section766832317011"></a>

```
├── src
│   ├──main.cpp           //实现文件 
├── Makefile              //编译脚本 
├── CMakeLists.txt        //编译脚本
├── data         
│   ├──data_generate.py   // 通过算子原型构建Graph时，用于生成Graph所需要的数据信息，例如权重、偏置等数据
│   ├──tensorflow_generate.py  // 将TensorFlow原始模型解析为Graph时，用于生成.pb格式的TensorFlow模型
│   ├──caffe_generate.py  // 将Caffe原始模型解析为Graph时，用于生成.pbtxt格式的Caffe模型与.caffemodel格式的权重文件     
├── scripts
│   ├──host_version.conf  // version配置
│   ├──testcase_300.sh    // Atlas300测试脚本 
```

## 环境要求<a name="section3833348101215"></a>

-   操作系统及架构：CentOS系统、Euleros系统、Ubuntu系统
-   编译器：g++
-   芯片：Atlas 训练系列产品、Atlas 推理系列产品（配置Ascend 310P AI处理器）
-   python及依赖的库：python3.7.5
-   已完成昇腾AI软件栈在开发环境上的部署
-   如果采用模型解析方式构图，请在当前环境安装tensorflow1.15.0和caffe

## 准备构图数据<a name="section48724517295"></a>

-   如果用户需要通过算子原型构建Graph，请执行以下操作准备构图数据：
    1.  在**data**目录执行数据生成脚本，**python3.7.5  data_generate.py**
    2.  执行结束后，在**data**目录下生成.bin格式的数据。后续模型构建时会从该文件读取权重、偏置等数据。

-   如果用户需要将TensorFlow原始模型解析为Graph，请执行以下操作准备构图数据：
    1.  在**data**目录执行tensorflow原始模型生成脚本，**python3.7.5  tensorflow_generate.py**
    2.  执行结束后，在**data**目录下生成.pb格式的模型文件。后续将原始模型解析时会使用该tensorflow模型，名称为tf_test.pb。

-   如果用户需要将Caffe原始模型解析为Graph，请执行以下操作准备构图数据：
    1.  在**data**目录执行caffe原始模型生成脚本，**python3.7.5  caffe_generate.py**
    2.  执行结束后，在**data**目录下生成.pbtxt格式的模型文件与.caffemodel格式的权重文件。后续将caffe原始模型与权重文件解析时会使用两者，名称分别为caffe_test.pbtxt与caffe_test.caffemodel。


## 程序编译<a name="section6697627144813"></a>

1. 根据实际情况修改**Makefile**文件中的如下信息。

   - ASCEND_PATH：指定到ATC或FwkACLlib的安装目录，例如/home/HwHiAiUser/Ascend/ascend-toolkit/latest

   - INCLUDES：需要包含的头文件，对于本示例，无需修改。如果是用户自行开发的代码，当需要添加头文件时，在示例下方直接增加行即可，注意不要删除原有项目。如果网络中有自定义算子，请增加自定义算子的原型定义头文件。

   - LIBS：需要链接的库，对于本示例，无需修改。如果是用户自行开发的代码，当需要添加链接库时，在示例下方直接增加行即可，注意不要删除原有项目。

     >禁止链接软件包中的其他so，否则后续升级可能会导致兼容性问题。

2. 执行如下命令进行编译。

   依次执行**make clean**和**make ir_build**。

3. 编译结束后，在**out**目录下生成可执行文件**ir_build**。

## 程序运行<a name="section1843713353512"></a>

1. 配置环境变量。

      - 若运行环境上安装的“Ascend-cann-toolkit”包，环境变量设置如下：

        ```
        . ${HOME}/Ascend/cann/bin/setenv.bash
        ```

        “$HOME/Ascend”请替换相关软件包的实际安装路径。


2. 在**out**目录下执行可执行文件。

   - 如果用户采用算子原型构图方式，请执行如下命令：

     **./ir_build ${soc_version} gen**
     
     ${soc_version}：昇腾AI处理器的版本，可以从[ATC离线模型编译工具](https://hiascend.com/document/redirect/CannCommercialAtc)中的“参数说明”部分的“--soc_version”查询。
     
     编译成功提示：

     ```
     ========== Generate Graph1 Success!========== 
     Build Model1 SUCCESS!
     Save Offline Model1 SUCCESS!
     ```

   - 如果用户采用将TensorFlow原始模型解析为Graph的构图方式，请执行如下命令：

     **./ir_build ${soc_version} tf**

     编译成功提示：

     ```
     ========== Generate graph from tensorflow origin model success.==========
     Modify Graph Start.
     Find src node: const. 
     Find dst node: add.
     Modify Graph Success.
     ========== Modify tensorflow origin graph success.==========
     Build Model1 SUCCESS!
     Save Offline Model1 SUCCESS!
     ```

   - 如果用户采用将Caffe原始模型解析为Graph的构图方式，请执行如下命令：

     **./ir_build ${soc_version} caffe**

     编译成功提示：

     ```
     ========== Generate graph from caffe origin model success.========== 
     Build Model1 SUCCESS!
     Save Offline Model1 SUCCESS!
     ```

3. 检查执行结果。

   在**out**目录下生成离线模型文件。
