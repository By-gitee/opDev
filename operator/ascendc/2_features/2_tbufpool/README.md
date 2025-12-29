## 目录结构介绍
```
├── 2_tbufpool
│   ├── cmake                               // 编译工程文件
│   ├── op_host                             // 本样例tiling代码实现
│   │   ├── tbufpool_custom_tilling.cpp
│   │   ├── tbufpool_custom_tilling.h
│   ├── op_kernel                           // 本样例kernel侧代码实现
│   │   ├── tbufpool_custom.cpp
│   │   ├── tbufpool_custom.h
│   ├── scripts
│   │   ├── gen_data.py                     // 输入数据和真值数据生成脚本    
│   ├── CMakeLists.txt                      // 编译工程文件
│   ├── data_utils.h                        // 数据读入写出函数
│   ├── main.cpp                            // 主函数，调用算子的应用程序，含CPU域及NPU域调用
│   └── run.sh                              // 编译运行算子的脚本
```
## 代码实现介绍
数据量较大且内存有限时，无法一次完成所有数据搬运，需要拆分成多个阶段计算，每次计算使用其中的一部分数据，可以通过TBufPool资源池进行内存地址复用。本例中，通过调用InitBufPool基础API对Add算子和Sub算子实现过程进行内存管理。从Tpipe划分出资源池tbufPool0，tbufPool0为src0Gm分配空间后，继续分配了资源池tbufPool1，指定tbufPool1与tbufPool2复用并分别运用于第一、二轮计算，此时tbufPool1及tbufPool2共享起始地址及长度。

- kernel实现  
  Add算子的数学表达式为：
  ```
  z = x + y
  ```
  Sub算子的数学表达式为：
  ```
  z = x - y
  ```

  计算逻辑是：Ascend C提供的矢量计算接口的操作元素都为LocalTensor，首先启用tbufpool1，将部分输入数据src0Gm，部分输入数据src1Gm搬运进片上储存，调用计算接口完成相加计算，搬出到外部存储上。之后切换到tbufpool2进行剩余数据相减计算，得到最终结果，再搬出到外部存储上。  

  本样例算子的实现流程分为6个基本任务：CopyIn，Compute，CopyOut，CopyIn1，Compute1，CopyOut1。
  - CopyIn任务负责将Global Memory上的部分输入Tensor src0Gm和src1Gm搬运到Local Memory，分别存储在src0Local、src1Local；
  - Compute任务负责对src0Local、src1Local执行加法操作，计算结果存储在dstLocal中；
  - CopyOut任务负责将输出数据从dstLocal搬运至Global Memory上的输出Tensor dstGlobal中。
  - CopyIn1任务负责将Global Memory上的剩余输入Tensor src0Gm和src1Gm搬运到Local Memory，分别存储在src0Local、src1Local；
  - Compute1任务负责对src0Local、src1Local执行剩余数据减法操作，计算结果存储在dstLocal中；
  - CopyOut1任务负责将输出数据从dstLocal搬运至Global Memory上的输出Tensor dstGlobal中。

- 调用实现
  1. CPU侧运行验证主要通过ICPU_RUN_KF CPU调测宏等CPU调测库提供的接口来完成；
  2. NPU侧运行验证主要通过使用ACLRT_LAUNCH_KERNEL内核调用宏来完成。

  应用程序通过ASCENDC_CPU_DEBUG 宏区分代码逻辑运行于CPU侧还是NPU侧。

## 运行样例算子
  - 打开样例目录   
    以命令行方式下载样例代码，master分支为例。
    ```bash
    cd ${git_clone_path}/samples/operator/ascendc/2_features/2_tbufpool
    ```
  - 配置环境变量

    请根据当前环境上CANN开发套件包的[安装方式](https://hiascend.com/document/redirect/CannCommunityInstSoftware)，选择对应配置环境变量的命令。
    - 默认路径，root用户安装CANN软件包
      ```bash
      export ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
      ```
    - 默认路径，非root用户安装CANN软件包
      ```bash
      export ASCEND_INSTALL_PATH=$HOME/Ascend/ascend-toolkit/latest
      ```
    - 指定路径install_path，安装CANN软件包
      ```bash
      export ASCEND_INSTALL_PATH=${install_path}/ascend-toolkit/latest
      ```
    


  - 生成输入和真值

    执行如下命令后，当前目录生成input和output目录存放输入数据和真值数据。
    ```
    python3 scripts/gen_data.py
    ```
    
  - 样例执行

    ```bash
    bash run.sh -r [RUN_MODE] -v  [SOC_VERSION]
    ```
    - RUN_MODE：编译方式，可选择CPU调试，NPU仿真，NPU上板。支持参数为[cpu / sim / npu]。
    - SOC_VERSION：昇腾AI处理器型号，如果无法确定具体的[SOC_VERSION]，则在安装昇腾AI处理器的服务器执行npu-smi info命令进行查询，在查询到的“Name”前增加Ascend信息，例如“Name”对应取值为xxxyy，实际配置的[SOC_VERSION]值为Ascendxxxyy。支持以下产品型号：
      - Atlas A2训练系列产品/Atlas 800I A2推理产品

    示例如下，Ascendxxxyy请替换为实际的AI处理器型号。
    ```bash
    bash run.sh -r cpu -v Ascendxxxyy
    ```