## 目录结构介绍
```
├── BareMixInvocation             // 通过更底层的编码方式，实现MatmulLeayrelu融合算子的样例
│   ├── cmake                     // 编译工程文件
│   ├── scripts
│   │   ├── gen_data.py           // 输入数据和真值数据生成脚本文件
│   │   └── verify_result.py      // 验证输出数据和真值数据是否一致的验证脚本
│   ├── CMakeLists.txt            // 编译工程文件
│   ├── data_utils.h              // 数据读入写出函数
│   ├── main.cpp                  // 主函数，调用算子的应用程序，含CPU域及NPU域调用
│   ├── baremix_custom_tiling.cpp   // 算子tiling实现
│   ├── baremix_custom.cpp           // 算子kernel实现
│   └── run.sh                    // 编译运行算子的脚本
```

## 算子规格描述
在核函数直调样例中，算子实现支持的shape为：M = 128, N = 128, K = 256。
<table>
<tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">a</td><td align="center">M * K</td><td align="center">float16</td><td align="center">ND</td></tr>
<tr><td align="center">b</td><td align="center">K * N</td><td align="center">float16</td><td align="center">ND</td></tr>
</tr>
</tr>
<tr><td rowspan="1" align="center">算子输出</td><td align="center">c</td><td align="center">M * N</td><td align="center">float</td><td align="center">ND</td></tr>
</tr>
</tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">baremix_custom</td></tr>
</table>

## 代码实现介绍
本样例中实现的是[m, n, k]固定为[128, 128, 256]的MatmulLeakyRelu算子，仅在AIC核调用Matmul高阶API并计算，完成后调用AIV核完成LeakyRelu的计算。
- kernel实现  
  Matmul算子的数学表达式为：
  ```
  C = A * B + Bias
  ```
  其中A的形状为[128, 256], B的形状为[256, 128], C的形状为[128, 128]。具体请参考[baremix_custom.cpp](./baremix_custom.cpp)。

  LeakyRelu算子的数学表达式为：
  ```
  C = C > 0 ? C : C * S
  ```
  其中S为用户设置的LeakyRelu比例系数
  
  **本样例关键代码介绍如下：
  - 设置ASCENDC_CUBE_ONLY，仅在AIC核进行matmul计算
  - 设置Kernel类型为KERNEL_TYPE_MIX_XXX，同时启用AIV核和AIC核
  - 使用ASCEND_IS_AIC/ASCEND_IS_AIV隔离AIC/AIV核上的代码
  - 使用同步接口，自行完成核间同步
  ```c++
  #define ASCENDC_CUBE_ONLY //指定Matmul运行在AIC核上
  ...
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2); // 设置Kernel类型为KERNEL_TYPE_MIX_XXX
  ...
  if ASCEND_IS_AIC {
    ...
    // AIC核进行Matmul计算
    // AIC核完成计算后，通过AscendC::CrossCoreSetFlag<modeId, pipe>(flagId)发送同步flag
  }
  ...
  if ASCEND_IS_AIV {
    ...
    // AIV核通过AscendC::CrossCoreWaitFlag(flagId)接收同步flag
    // AIV核进行LeakyRelu计算
  }
  ```

- 调用实现  
  1. CPU侧运行验证主要通过ICPU_RUN_KF CPU调测宏等CPU调测库提供的接口来完成；
  2. NPU侧运行验证主要通过使用ACLRT_LAUNCH_KERNEL内核调用宏来完成。

  应用程序通过ASCENDC_CPU_DEBUG宏区分代码逻辑运行于CPU侧还是NPU侧。

## 运行样例算子
  - 打开样例目录
    以命令行方式下载样例代码，master分支为例。
    ```bash
    cd ${git_clone_path}/samples/operator/ascendc/0_introduction/22_baremix_kernellaunch/BareMixInvocation
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
    - SOC_VERSION：昇腾AI处理器型号，如果无法确定具体的[SOC_VERSION]，则在安装昇腾AI处理器的服务器执行npu-smi info命令进行查询，在查询到的“Name”前增加Ascend信息，例如"Name"对应取值为xxxyy，实际配置的[SOC_VERSION]值为Ascendxxxyy。支持以下产品型号：
      - Atlas A2训练系列产品/Atlas 800I A2推理产品

    示例如下，Ascendxxxyy请替换为实际的AI处理器型号。

    ```bash
    bash run.sh -r cpu -v Ascendxxxyy
    ```


## 更新说明
| 时间       | 更新事项     |
| ---------- | ------------ |
| 2025/7/28 | 更新本readme |
