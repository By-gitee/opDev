## 目录

- [目录](#目录)
- [样例介绍](#样例介绍)
- [目录结构](#目录结构)
- [环境要求](#环境要求)
- [程序编译](#程序编译)
- [样例运行](#样例运行)


## 样例介绍

功能：通过LLM-DataDist接口实现分离部署场景下KvCache管理功能。


## 目录结构

相对当前目录，结构如下：
```
├── prompt_sample.cpp                // sample1的prompt样例main函数
├── decoder_sample.cpp               // sample1的decoder样例main函数
├── prompt_sample2.cpp               // sample2的prompt样例main函数
├── decoder_sample2.cpp              // sample2的decoder样例main函数
├── prompt_sample3.cpp               // sample3的prompt样例main函数
├── decoder_sample3.cpp              // sample3的decoder样例main函数
├── prompt_sample4.cpp               // sample4的prompt样例main函数
├── decoder_sample4.cpp              // sample4的decoder样例main函数
├── CMakeLists.txt                   // 编译脚本
```


## 环境要求

-   操作系统及架构：Euleros x86系统、Euleros aarch64系统
-   编译器：g++
-   芯片：Atlas 训练系列产品、Atlas 推理系列产品（配置Ascend 310P AI处理器）
-   python及依赖的库：python3.7.5
-   已完成昇腾AI软件栈在运行环境上的部署

## 程序编译

1. 修改CMakeLists.txt文件中的安装包路径

2. 在当前目录下执行如下命令进行编译。

   依次执行:

   ```
   mkdir build && cd build
   cmake .. && make
   ```

3. 编译结束后，在**build**目录下生成可执行文件**prompt_sample**与**decoder_sample**。

## 样例运行
1. 执行前准备：

    - 在Prompt与Decoder的主机分别执行以下命令，查询该主机的device ip信息
        ```
        for i in {0..7}; do hccn_tool -i $i -ip -g; done
        ```
        **注: 如果出现hccn_tool命令找不到的情况，可在CANN包安装目录下搜索hccn_tool，找到可执行文件执行**
2. 配置环境变量
    - 若运行环境上安装的“Ascend-cann-toolkit”包，环境变量设置如下：

        ```
        . ${HOME}/Ascend/ascend-toolkit/set_env.sh
        ```

        “$HOME/Ascend”请替换相关软件包的实际安装路径。

    - 若运行环境上安装的“CANN-XXX.run”包，环境变量设置如下：

        ```
        source ${HOME}/Ascend/latest/bin/setenv.bash
        ```

        “$HOME/Ascend”请替换相关软件包的实际安装路径。

3. 在运行环境执行可执行文件。

    3.1 执行sample1

    此样例介绍了libllm_engine.so的decoder向prompt进行pull cache和pull blocks流程

    - 执行prompt_sample, 参数为device_id与local_ip, 其中device_id为prompt要使用的device_id, local_ip为prompt所在device的ip，如:
        ```
        ./prompt_sample 0 10.10.10.1
        ```

    - 执行decoder_sample, 参数为device_id、local_ip与remote_ip, 其中device_id为decoder要使用的device_id, local_ip为decoder所在device的ip，remote_ip为prompt所在device的ip，如:
        ```
        ./decoder_sample 4 10.10.10.5 10.10.10.1
        ```

    3.2 执行sample2

    此样例介绍了libllm_datadist.so的decoder向prompt进行pull cache和pull blocks流程，其中link和pull的方向与角色无关，可以根据需求更改

    - 执行prompt_sample2, 参数为device_id、local_host_ip和remote_host_ip, 其中device_id为prompt要使用的device_id, local_host_ip为prompt所在host的ip, remote_host_ip为decoder所在host的ip，如:
        ```
        ./prompt_sample2 0 10.10.170.1
        ```

    - 执行decoder_sample2, 参数为device_id、local_host_ip和remote_host_ip, 其中device_id为decoder要使用的device_id, local_host_ip为decoder所在host的ip，remote_host_ip为prompt所在host的ip，如:
        ```
        ./decoder_sample2 2 10.170.10.2 10.170.10.1
        ```

    3.3 执行sample3

    此样例介绍了libllm_datadist.so的prompt向decoder进行push cache和push blocks流程，其中link和push的方向与角色无关，可以根据需求更改

    - 执行prompt_sample3, 参数为device_id与local_ip, 其中device_id为prompt要使用的device_id, local_ip为prompt所在host的ip，如:
        ```
        ./prompt_sample3 0 10.10.10.1 10.10.10.5
        ```

    - 执行decoder_sample3, 参数为device_id、local_ip与remote_ip, 其中device_id为decoder要使用的device_id, local_ip为decoder所在host的ip，remote_ip为prompt所在host的ip，如:
        ```
        ./decoder_sample3 4 10.10.10.5
        ```

    3.4 执行sample4

    此样例介绍了libllm_datadist.so的角色切换，并结合pull以及push使用流程

    - 执行prompt_sample4, 参数为device_id、local_host_ip和remote_host_ip, 其中device_id为prompt要使用的device_id, local_host_ip为prompt所在host的ip, remote_host_ip为decoder所在host的ip，如:
        ```
        ./prompt_sample4 0 10.10.170.1 10.170.10.2
        ```

    - 执行decoder_sample4, 参数为device_id、local_host_ip和remote_host_ip, 其中device_id为decoder要使用的device_id, local_host_ip为decoder所在host的ip，remote_host_ip为prompt所在host的ip，如:
        ```
        ./decoder_sample4 2 10.170.10.2 10.170.10.1