## 目录

  - [样例介绍](#样例介绍)
  - [目录结构](#目录结构)
  - [获取源码包](#获取源码包) 
  - [第三方依赖安装](#第三方依赖安装)
  - [样例运行](#样例运行)
  - [更新说明](#更新说明)
  - [已知issue](#已知issue)
    
## 样例介绍

功能：使用llama模型对输入问题进行aclnn推理，输出问题的答案。    
样例输入：问题语句。   
样例输出：打屏显示问题以及对应的答案。 

## 目录结构

```
├── data                        //数据文件夹
│   └── xxxx.bin                //存放样例运行所加载的除llama模型权重外的固定数据bin文件 
├── input                       //输入数据文件夹
│   └── xxxx.bin                //存放当前输入问题语句预处理后生成的bin文件
├── llama_weight                //llama模型权重数据文件夹
│   └── xxxx                    //存放llama模型所有权重文件，以及tokenizer模型
├── output                      //编译输出文件夹，存放编译生成的可执行文件，以及输出bin文件
│   ├── xxxx                    //可执行文件 
│   └── xxxx                    //样例运行的输出结果bin文件 
├── scripts                     //运行脚本文件夹
│   ├── input_trans.py          //输入问题转为模型输入脚本
│   └── run_llama.sh            //快速运行脚本
├── src                         //源文件
│   ├── CMakeLists.txt          //Cmake编译文件
│   ├── common.cpp              //重复调用的公共函数cpp文件
│   ├── common.h                //重复调用的公共函数h文件
│   ├── llama.cpp               //主函数，llama功能的实现文件  
│   └── struct.h                //结构体实现文件
└────── 
```

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

 设置环境变量，配置程序编译依赖的头文件，库文件路径。“$HOME/Ascend”请替换“Ascend-cann-toolkit”包的实际安装路径。

   ```
    source $HOME/Ascend/ascend-toolkit/set_env.sh
   ```
 
- transformer库
  
    使用压缩包方式下载安装transformer-4.29.2：
 
    1. 点击[transformers](https://github.com/huggingface/transformers/tree/v4.29.2)跳转到transformers仓，右上角选择 【Code】 下拉框并选择 【Download ZIP】
    2. 将ZIP包上传到开发环境中的普通用户家目录中，【例如：${HOME}/transformers-4.29.2.zip】
    3. 开发环境中，执行以下命令，解压安装zip包。     
   ```
  cd ${HOME}
  unzip transformers-4.29.2.zip
  pip3 install SentencePiece
  pip3 install -e transformers-4.29.2
  ```

## 样例运行
   
  - 数据准备

    请从以下链接获取该样例的权重文件，解压放在样例目录下。 
    ```
    cd $HOME/samples/inference/modelInference/sampleLlama
    wget https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/llama_weight/llama_weight.zip --no-check-certificate
    unzip llama_weight.zip
    wget https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/llama_weight/data.zip --no-check-certificate
    unzip data.zip
    ```

  - 样例运行

    执行运行脚本，开始样例运行。
    ```
    export ASCEND_CUSTOM_PATH=$HOME/Ascend/ascend-toolkit/latest
    bash run_llama.sh
    ```

  - 样例结果展示
    
    执行成功后，样例将根据输入问题，输出对应问题答案，打屏显示。


## 更新说明
  | 时间 | 更新事项 |
  |----|------|
  | 2023/12/07| 新增sampleLlama/README.md |
  

## 已知issue

  暂无