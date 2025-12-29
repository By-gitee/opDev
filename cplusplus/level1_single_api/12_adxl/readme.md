## 目录

- [目录](#目录)
- [样例介绍](#样例介绍)
- [目录结构](#目录结构)
- [环境要求](#环境要求)
- [程序编译](#程序编译)
- [样例运行](#样例运行)


## 样例介绍

功能：通过adxl engine接口实现Cache傳輸功能。


## 目录结构

相对当前目录，结构如下：
```
├── adxl_engine_sample.cpp          // adxl_engine的sample1样例
├── adxl_engine_sample2.cpp         // adxl_engine的sample2样例
├── CMakeLists.txt                  // 编译脚本 
```


## 环境要求

-   操作系统及架构：Euleros x86系统、Euleros aarch64系统
-   编译器：g++
-   芯片：Atlas 训练系列产品、Atlas 推理系列产品（配置Ascend 310P AI处理器）
-   python及依赖的库：python3.7.5
-   已完成昇腾AI软件栈在运行环境上的部署

## 程序编译

1. 修改CMakeLists.txt文件中的安装包路径

2. 在当前目录执行如下命令进行编译。

   依次执行:

   ```
   mkdir build && cd build
   cmake .. && make
   ```

3. 编译结束后，在**build**目录下生成可执行文件**adxl_engine_sample**。

## 样例运行
1. 配置环境变量
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

2. 在运行环境执行可执行文件。

    3.1 执行sample, client-server模式，h2d场景

    - 执行client adxl_engine_sample, 参数为device_id、local engine和remote engine, 其中device_id为client要使用的device_id，如:
        ```
        HCCL_INTRA_ROCE_ENABLE=1 ./adxl_engine_sample 0 10.10.10.0 10.10.10.1:16000
        ```

    - 执行server adxl_engine_sample, 参数为device_id、local engine, 其中device_id为server要使用的device_id, 如:
        ```
        HCCL_INTRA_ROCE_ENABLE=1 ./adxl_engine_sample 1 10.10.10.1:16000
        ```

    3.2 执行sample2, 均作为server，d2d场景

    - 执行server1 adxl_engine_sample2, 参数为device_id、local engine和remote engine, 其中device_id为当前engine要使用的device_id，如:
        ```
        HCCL_INTRA_ROCE_ENABLE=1 ./adxl_engine_sample2 0 10.10.10.0:16000 10.10.10.1:16001
        ```

    - 执行server2 adxl_engine_sample2, 参数为device_id、local engine和remote engine, 其中device_id为当前engine要使用的device_id, 如:
        ```
        HCCL_INTRA_ROCE_ENABLE=1 ./adxl_engine_sample2 1 10.10.10.1:16001 10.10.10.0:16000
        ```
    **注**：HCCL_INTRA_ROCE_ENABLE=1表示使用RDMA进行传输