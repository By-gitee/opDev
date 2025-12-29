## 目录结构介绍

```
├── KernelLaunch
│   ├── cmake                   // 编译工程文件
│   ├── scripts
│   │   ├── gen_data.py         // 输入数据和真值数据生成脚本
│   │   └── verify_result.py    // 验证输出数据和真值数据是否一致的验证脚本
│   ├── add_custom_tiling.h     // tiling结构体
│   ├── add_custom_v1.cpp       // 算子kernel实现1：未优化前实现
│   ├── add_custom_v2.cpp       // 算子kernel实现2：基于实现1，实现double buffer
│   ├── add_custom_v3.cpp       // 算子kernel实现3：优化double buffer实现，简化判断逻辑，并使用LocalMemAllocator简化代码
│   ├── add_custom_v4.cpp       // 算子kernel实现4：基于add_custom_v3，修改地址分配逻辑，消除bank冲突
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   ├── main.cpp                // 主函数，调用算子的应用程序，含CPU域及NPU域调用
│   └── run.sh                  // 编译运行算子的脚本
```

## 代码实现介绍

本样例中实现的是固定shape为72*4096的Add算子。

- kernel实现

  Add算子的数学表达式为：

  ```
  z = x + y
  ```

  计算逻辑是：Ascend C提供的矢量计算接口的操作元素都为LocalTensor，输入数据需要先搬运进片上存储，然后使用计算接口完成两个输入参数相加，得到最终结果，再搬出到外部存储上。

  Add算子的实现流程分为3个基本任务：CopyIn，Compute，CopyOut。CopyIn任务负责将Global Memory上的输入Tensor xGm和yGm搬运到Local Memory，分别存储在xLocal、yLocal，Compute任务负责对xLocal、yLocal执行加法操作，计算结果存储在zLocal中，CopyOut任务负责将输出数据从zLocal搬运至Global Memory上的输出Tensor zGm中。

  实现1：请参考[add_custom_v1.cpp](./add_custom_v1.cpp)，使用静态Tensor编程方法，进行add算子的编程。

  实现2：请参考[add_custom_v2.cpp](./add_custom_v2.cpp)，优化性能，使用double buffer进行流水排布。

  实现3：请参考[add_custom_v3.cpp](./add_custom_v3.cpp)，优化add_custom_v2中反向同步，替换为MTE2等待MTE3执行结束。减少分支判断的同时，算子性能因为double buffer的原因不受影响。另外使用LocalMemAllocator进行线性内存分配，Bank冲突不敏感场景可以使用这种方式简化分配。

  实现4：请参考[add_custom_v4.cpp](./add_custom_v4.cpp)，基于add_custom_v3的实现，优化地址分配消除Bank冲突。 

- 调用实现

  1. CPU侧运行验证主要通过ICPU_RUN_KF CPU调测宏等CPU调测库提供的接口来完成；
  2. NPU侧运行验证主要通过使用ACLRT_LAUNCH_KERNEL内核调用宏来完成。

  应用程序通过ASCENDC_CPU_DEBUG 宏区分代码逻辑运行于CPU侧还是NPU侧。

## 运行样例算子

- 打开样例目录
  以命令行方式下载样例代码，master分支为例。

  ```bash
  cd ${git_clone_path}/samples/operator/ascendc/0_introduction/23_static_tensor_programming_kernellaunch/KernelLaunch
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

  - RUN_MODE：编译方式，可选择CPU调试，NPU仿真，NPU上板。支持参数为[cpu /sim / npu]
  - SOC_VERSION：昇腾AI处理器型号，如果无法确定具体的[SOC_VERSION]，则在安装昇腾AI处理器的服务器执行npu-smi info命令进行查询，在查询到的“Name”前增加Ascend信息，例如“Name”对应取值为xxxyy，实际配置的[SOC_VERSION]值为Ascendxxxyy。支持以下产品型号：
    - Atlas A2训练系列产品/Atlas 800I A2推理产品

  示例如下，Ascendxxxyy请替换为实际的AI处理器型号。

  ```bash
  bash run.sh -r cpu -v Ascendxxxyy
  ```

## 更新说明


| 时间       | 更新事项     |
| ---------- | ------------ |
| 2025/09/06 | 新增本readme |
