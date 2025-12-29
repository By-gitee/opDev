**注意：本样例仅支持CANN社区版8.0.RC3.alpha003及以上版本。**
## 目录结构介绍
```
├── AbsDuplicateKernelInvocation
│   ├── cmake                   // 编译工程文件
│   ├── scripts
│   │   ├── gen_data.py         // 输入数据和真值数据生成脚本
│   │   └── verify_result.py    // 验证输出数据和真值数据是否一致的验证脚本
│   ├── abs_duplicate.cpp       // 算子kernel实现
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   ├── main.cpp                // 主函数，调用算子的应用程序，含CPU域及NPU域调用
│   └── run.sh                  // 编译运行算子的脚本
```
## 代码实现介绍
本样例中实现的是固定shape为64*11的Abs算子。
- kernel实现
  Abs算子的数学表达式为：
  ```
  z = abs(x)
  ```
  计算逻辑是：Ascend C提供的矢量计算接口的操作元素的数据类型都为LocalTensor，输入数据需要先搬运进片上存储，然后使用Abs计算接口完成对输入参数取绝对值的运算，得到最终结果，再搬出到外部存储上。

  Abs算子的实现流程分为3个基本任务：CopyIn，Compute，CopyOut。CopyIn任务负责将Global Memory上的输入inputGM搬运到Local Memory，存储在inputLocal中。Compute任务负责先将inputLocal中的数据使用Duplicate对冗余数据进行清零处理，之后对inputLocal执行取绝对值操作，计算结果存储在outputLocal中。CopyOut任务负责将输出数据从outputLocal搬运至Global Memory上的输出outputGM中，为防止数据踩踏本例使用原子加进行搬出。具体请参考[abs_duplicate.cpp](./abs_duplicate.cpp)。

- 调用实现
  1. CPU侧运行验证主要通过ICPU_RUN_KF CPU调测宏等CPU调测库提供的接口来完成；
  2. NPU侧运行验证主要通过使用ACLRT_LAUNCH_KERNEL内核调用宏来完成。

  应用程序通过ASCENDC_CPU_DEBUG 宏区分代码逻辑运行于CPU侧还是NPU侧。

## 运行样例算子
  - 打开样例目录
    以命令行方式下载样例代码，master分支为例。
    ```bash
    cd ${git_clone_path}/samples/operator/ascendc/0_introduction/16_unaligned_abs_kernellaunch/AbsDuplicateKernelInvocation
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
    - RUN_MODE：编译方式，可选择CPU调试，NPU仿真，NPU上板。支持参数为[cpu / sim / npu]。
    - SOC_VERSION：昇腾AI处理器型号，如果无法确定具体的[SOC_VERSION]，则在安装昇腾AI处理器的服务器执行npu-smi info命令进行查询，在查询到的“Name”前增加Ascend信息，例如“Name”对应取值为xxxyy，实际配置的[SOC_VERSION]值为Ascendxxxyy。支持以下产品型号：
      - Atlas 训练系列产品
      - Atlas 推理系列产品AI Core
      - Atlas A2训练系列产品/Atlas 800I A2推理产品
      - Atlas 200/500 A2推理产品

    示例如下，Ascendxxxyy请替换为实际的AI处理器型号。
    ```bash
    bash run.sh -r cpu -v Ascendxxxyy
    ```
## 更新说明
| 时间       | 更新事项     |
| ---------- | ------------ |
| 2024/09/10 | 新增本样例 |
| 2024/11/16 | 样例目录调整 |