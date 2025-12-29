## 目录结构介绍
```
├── DumpTensorKernelInvocationCube
│   ├── cmake                     // 编译工程文件
│   ├── scripts
│   │   ├── gen_data.py           // 输入数据和真值数据生成脚本文件
│   │   └── verify_result.py      // 验证输出数据和真值数据是否一致的验证脚本
│   ├── CMakeLists.txt            // 编译工程文件
│   ├── data_utils.h              // 数据读入写出函数
│   ├── main.cpp                  // 主函数，调用算子的应用程序
│   ├── mmad_custom_cube_only.h   // Atlas A2训练系列产品kernel实现
│   ├── mmad_custom.h             // Atlas推理系列产品kernel实现
│   ├── mmad_custom.cpp           // 算子kernel实现
│   └── run.sh                    // 编译运行算子的脚本
```

## 算子规格描述
在核函数直调样例中，算子实现支持的shape为：M = 32, N = 32, K = 32。
<table>
<tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">a</td><td align="center">M * K</td><td align="center">float16</td><td align="center">ND</td></tr>
<tr><td align="center">b</td><td align="center">K * N</td><td align="center">float16</td><td align="center">ND</td></tr>
</tr>
</tr>
<tr><td rowspan="1" align="center">算子输出</td><td align="center">c</td><td align="center">M * N</td><td align="center">float</td><td align="center">ND</td></tr>
</tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">mmad_custom</td></tr>
</table>

## 代码实现介绍
本样例中实现的是[m, n, k]固定为[32, 32, 32]的Mmad算子，并使用Ascend C基础API实现，同时添加DumpTensor用于Dump指定Tensor的内容。
- kernel实现  
  Mmad算子的数学表达式为：
  $$
  C = A * B
  $$
  其中A的形状为[32, 32], B的形状为[32, 32], C的形状为[32, 32]。具体请参考[mmad_custom.cpp](./mmad_custom.cpp)。
  
  **注：当使用硬件分离架构的产品如Atlas A2训练系列产品/Atlas 800I A2推理产品时，由于样例使用的基础API均为Cube核指令，本样例设置了Cube Only模式，只调用Cube核完成计算，代码如下：
  ```c++
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
  ```

- 调用实现

  NPU侧运行验证主要通过使用ACLRT_LAUNCH_KERNEL内核调用宏来完成。


## 运行样例算子
  - 打开样例目录
    以命令行方式下载样例代码，master分支为例。
    ```bash
    cd ${git_clone_path}/samples/operator/ascendc/1_utilities/7_dumptensor/KernelLaunch/DumpTensorKernelInvocationCube
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
    bash run.sh -v  [SOC_VERSION]
    ```
    - SOC_VERSION：昇腾AI处理器型号，如果无法确定具体的[SOC_VERSION]，则在安装昇腾AI处理器的服务器执行npu-smi info命令进行查询，在查询到的“Name”前增加Ascend信息，例如"Name"对应取值为xxxyy，实际配置的[SOC_VERSION]值为Ascendxxxyy。支持以下产品型号：
      - Atlas 推理系列产品AI Core
      - Atlas A2训练系列产品/Atlas 800I A2推理产品

    示例如下，Ascendxxxyy请替换为实际的AI处理器型号。

    ```bash
    bash run.sh -v Ascendxxxyy
    ```


## 更新说明
| 时间       | 更新事项     |
| ---------- | ------------ |
| 2025/01/06 | 新增本readme |
