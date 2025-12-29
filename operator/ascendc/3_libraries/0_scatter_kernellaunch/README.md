# 兼容Scatter算子直调样例
## 概述
本样例介绍兼容Scatter算子实现及核函数直调方法，Ascend910C/Ascend910B不支持Scatter的能力，可以使用标量搬出的方式进行算子层面的仿真处理和修改算法的方式改变当前使用Scatter指令的算法。本样例是通过标量搬出的方式，使用<<<>>>内核调用符来完成算子核函数在NPU侧运行验证的基础流程，给出了对应的端到端实现。

## 支持的AI处理器
- Ascend 910C
- Ascend 910B
## 目录结构介绍

```
├── 0_scatter_kernellaunch
│   ├── cmake                   // 编译工程文件
│   ├── scripts
│   │   ├── gen_data.py         // 输入数据和真值数据生成脚本
│   │   └── verify_result.py    // 验证输出数据和真值数据是否一致的验证脚本
│   ├── scatter_custom.cpp      // 算子kernel实现
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   ├── main.cpp                // 主函数，调用算子的应用程序，含CPU域及NPU域调用
│   └── run.sh                  // 编译运行算子的脚本
```

## 算子描述
- 算子功能：  
  Scatter功能：给定一个连续的输入张量和一个目的地址偏移张量，Scatter指令根据偏移地址生成新的结果张量后将输入张量分散到结果张量中。

- 算子规格：  
  <table>  
  <tr><th align="center">算子类型(OpType)</th><th colspan="5" align="center">Scatter</th></tr>  
  <tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td><td align="center">default</td></tr>  
  <tr><td align="center">x</td><td align="center">-</td><td align="center">float16</td><td align="center">ND</td><td align="center">\</td></tr>  
  <tr><td align="center">y</td><td align="center">-</td><td align="center">uint32</td><td align="center">ND</td><td align="center">\</td></tr>  
  <tr><td rowspan="1" align="center">算子输出</td><td align="center">out</td><td align="center">-</td><td align="center">float16</td><td align="center">ND</td><td align="center">\</td></tr>  
  <tr><td align="center">attr属性</td><td align="center">value</td><td align="center">\</td><td align="center">float16</td><td align="center">\</td><td align="center">1.0</td></tr>
  <tr><td rowspan="1" align="center">核函数名</td><td colspan="5" align="center">scatter_custom</td></tr>  
  </table>

- 算子实现：  
样例中实现的是使用标量搬出的方式对于Scatter功能变换的兼容。
  - Scatter兼容策略
    - 标量搬出方式兼容Scatter：
    对于完全离散的场景,只能通过标量搬出的方式进行处理。
    - 修改算法的方式兼容Scatter：
      对于部分有规律的离散计算，譬如`[0~63][128~191][256~319]`...这种数据可以通过Loop循环搬出的方式来提升效率。
      | `64` | 32 | `64` | 16 | `64` | 32 | `64` |
      | --- | --- | --- | --- | --- | --- | --- |
      
      搬运非连续多个64长度的值出去时（存在部分规律）,可以通过Loop循环使能方式进行数据搬运。

  - kernel实现   
    兼容Scatter算子的实现流程分为3个基本任务：CopyIn任务负责将Global Memory上的输入Tensor srcGm和dstGm搬运到Local Memory，分别存储在xLocal、yLocal，Compute任务负责对xLocal、yLocal进行标量计算，计算结果存储在zLocal中，CopyOut任务负责将输出数据从zLocal搬运至Global Memory上的输出Tensor dstGm中。

  - 调用实现  
    使用内核调用符<<<>>>调用核函数。

## 编译运行：  
  - 打开样例目录   
    以命令行方式下载样例代码，master分支为例。
    ```bash
    cd ${git_clone_path}/samples/operator/ascendc/0_libraries/0_scatter_kernellaunch/
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
  - 样例执行

    ```bash
    bash run.sh -r [RUN_MODE] -v  [SOC_VERSION]
    ```
    - RUN_MODE：编译方式，可选择CPU调试，NPU仿真，NPU上板。支持参数为[cpu / sim / npu]
    - SOC_VERSION：昇腾AI处理器型号，如果无法确定具体的[SOC_VERSION]，则在安装昇腾AI处理器的服务器执行npu-smi info命令进行查询，在查询到的“Name”前增加Ascend信息，例如“Name”对应取值为xxxyy，实际配置的[SOC_VERSION]值为Ascendxxxyy。支持以下产品型号：
      - Atlas 推理系列产品AI Core
      - Atlas A2训练系列产品/Atlas 800I A2推理产品

    示例如下，Ascendxxxyy请替换为实际的AI处理器型号。
    ```bash
    bash run.sh -r cpu -v Ascendxxxyy
    ```
