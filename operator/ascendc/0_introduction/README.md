## 概述
简单的示例，适合初学者。

## 自定义算子样例说明
样例通过Ascend C编程语言实现了自定义算子，并按照不同的算子调用方式分别给出了对应的端到端实现。其中，目录以_frameworklaunch、_frameworklaunchlite、_kernellaunch结尾的样例说明如下。
- FrameworkLaunch：使用框架调用自定义算子。  
  按照工程创建->算子实现->编译部署>算子调用的流程完成算子开发。整个过程都依赖于算子工程：基于工程代码框架完成算子核函数的开发和Tiling实现，通过工程编译脚本完成算子的编译部署，继而实现单算子调用或第三方框架中的算子调用。
- FrameworkLaunchLite：使用msOpGen工具自动生成简易自定义算子工程，并调用自定义算子。  
  按照工程创建->算子实现->编译>算子调用的流程完成算子开发。整个过程都依赖于算子工程：基于工程代码框架完成算子核函数的开发和Tiling实现，通过工程编译脚本完成算子的编译，继而实现单算子调用或第三方框架中的算子调用。
- KernelLaunch：使用核函数直调自定义算子。  
  核函数的基础调用（Kernel Launch）方式，开发者完成算子核函数的开发和Tiling实现后，即可通过AscendCL运行时接口，完成算子的调用。


## 算子开发样例
|  目录名称                                                   |  功能描述                                              |  运行环境 |
| ------------------------------------------------------------ | ---------------------------------------------------- | -- |
| [0_helloworld](./0_helloworld) | 基于Ascend C的HelloWorld自定义算子调用结构演示样例 | Atlas 推理系列产品AI Core<br>Atlas A2训练系列产品/Atlas 800I A2推理产品<br>Atlas 200/500 A2推理产品 |
| [1_add_frameworklaunch](./1_add_frameworklaunch) | 基于Ascend C的Add自定义Vector算子及FrameworkLaunch调用样例 | Atlas 训练系列产品<br>Atlas 推理系列产品AI Core<br>Atlas A2训练系列产品/Atlas 800I A2推理产品<br>Atlas 200/500 A2推理产品 |
| [2_add_frameworklaunchlite](./2_add_frameworklaunchlite) | 基于Ascend C的Add自定义Vector算子及FrameworkLaunchLite调用样例 | Atlas 训练系列产品<br>Atlas 推理系列产品AI Core<br>Atlas A2训练系列产品/Atlas 800I A2推理产品<br>Atlas 200/500 A2推理产品 |
| [3_add_kernellaunch](./3_add_kernellaunch) | 基于Ascend C的Add自定义Vector算子及KernelLaunch调用样例 | Atlas 训练系列产品<br>Atlas 推理系列产品AI Core<br>Atlas A2训练系列产品/Atlas 800I A2推理产品<br>Atlas 200/500 A2推理产品 |
| [4_addn_frameworklaunch](./4_addn_frameworklaunch) | 基于Ascend C的AddN自定义Vector算子及FrameworkLaunch调用样例 | Atlas 推理系列产品AI Core<br>Atlas A2训练系列产品/Atlas 800I A2推理产品|
| [5_addn_kernellaunch](./5_addn_kernellaunch) | 基于Ascend C的AddN自定义Vector算子及KernelLaunch调用样例 | Atlas 推理系列产品AI Core<br>Atlas A2训练系列产品/Atlas 800I A2推理产品|
| [6_addtemplate_frameworklaunch](./6_addtemplate_frameworklaunch) | 基于Ascend C的Add（模板参数算子）自定义Vector算子及FrameworkLaunch调用样例 | Atlas 训练系列产品<br>Atlas 推理系列产品AI Core<br>Atlas A2训练系列产品/Atlas 800I A2推理产品<br>Atlas 200/500 A2推理产品 |
| [7_broadcast_frameworklaunch](./7_broadcast_frameworklaunch) | 基于Ascend C的Broadcast自定义Vector算子及FrameworkLaunch调用样例 | Atlas 推理系列产品AI Core<br>Atlas A2训练系列产品/Atlas 800I A2推理产品 |
| [8_library_frameworklaunch](./8_library_frameworklaunch) | 基于Ascend C的Add自定义算子和Matmul自定义算子的自定义算子工程静态库集成和使用样例 | Atlas 推理系列产品AI Core<br>Atlas A2训练系列产品/Atlas 800I A2推理产品 |
| [9_leakyrelu_frameworklaunch](./9_leakyrelu_frameworklaunch) | 基于Ascend C的LeakyReLU自定义Vector算子及FrameworkLaunch调用样例 | Atlas 训练系列产品<br>Atlas 推理系列产品AI Core<br>Atlas A2训练系列产品/Atlas 800I A2推理产品<br>Atlas 200/500 A2推理产品 |
| [10_matmul_frameworklaunch](./10_matmul_frameworklaunch) | 基于Ascend C的Matmul自定义Cube算子及FrameworkLaunch调用样例 | Atlas 推理系列产品AI Core<br>Atlas A2训练系列产品/Atlas 800I A2推理产品 |
| [11_matmul_kernellaunch](./11_matmul_kernellaunch) | 基于Ascend C的Matmul自定义Cube算子及KernelLaunch调用样例 | Atlas 推理系列产品AI Core<br>Atlas A2训练系列产品/Atlas 800I A2推理产品 |
| [12_matmulleakyrelu_frameworklaunch](./12_matmulleakyrelu_frameworklaunch) | 基于Ascend C的MatmulLeakyRelu自定义Cube+Vector算子及FrameworkLaunch调用样例 | Atlas 推理系列产品AI Core<br>Atlas A2训练系列产品/Atlas 800I A2推理产品 |
| [13_matmulleakyrelu_kernellaunch](./13_matmulleakyrelu_kernellaunch) | 基于Ascend C的MatmulLeakyRelu自定义Cube+Vector算子及KernelLaunch调用样例 | Atlas 推理系列产品AI Core<br>Atlas A2训练系列产品/Atlas 800I A2推理产品 |
| [14_reduce_frameworklaunch](./14_reduce_frameworklaunch) | 基于Ascend C的ReduceSum自定义算子及FrameworkLaunch调用样例 | Atlas 训练系列产品<br>Atlas A2训练系列产品/Atlas 800I A2推理产品 |
| [15_sub_frameworklaunch](./15_sub_frameworklaunch) | 基于Ascend C的Sub自定义算子及FrameworkLaunch调用样例 | Atlas 训练系列产品<br>Atlas 推理系列产品AI Core<br>Atlas A2训练系列产品/Atlas 800I A2推理产品<br>Atlas 200/500 A2推理产品 |
| [16_unaligned_abs_kernellaunch](./16_unaligned_abs_kernellaunch) | 基于Ascend C的非对齐Abs自定义算子及KernelLaunch调用样例 | Atlas 推理系列产品AI Core<br>Atlas A2训练系列产品/Atlas 800I A2推理产品 |
| [17_unaligned_reducemin_kernellaunch](./17_unaligned_reducemin_kernellaunch) | 基于Ascend C的非对齐ReduceMin自定义算子及KernelLaunch调用样例 | Atlas 推理系列产品AI Core<br>Atlas A2训练系列产品/Atlas 800I A2推理产品 |
| [18_unaligned_wholereduces_frameworklaunch](./18_unaligned_wholereduces_frameworklaunch) | 基于Ascend C的非对齐WholeReduceSum自定义算子及FrameworkLaunch调用样例 | Atlas A2训练系列产品/Atlas 800I A2推理产品
| [19_unaligned_wholereduces_kernellaunch](./19_unaligned_wholereduces_kernellaunch) | 基于Ascend C的非对齐WholeReduceSum自定义算子及KernelLaunch调用样例 | Atlas A2训练系列产品/Atlas 800I A2推理产品
| [20_mmad_kernellaunch](./20_mmad_kernellaunch) | 基于Ascend C基础API的Matmul自定义Cube算子及KernelLaunch调用样例 | Atlas 推理系列产品AI Core<br>Atlas A2训练系列产品/Atlas 800I A2推理产品 |
| [21_vectoradd_kernellaunch](./21_vectoradd_kernellaunch) | 基于Ascend C的Add多场景自定义Vector算子的KernelLaunch调用样例 | Atlas A2训练系列产品/Atlas 800I A2推理产品
| [22_baremix_kernellaunch](./22_baremix_kernellaunch) | 通过更底层的编码方式，实现MatmulLeayrelu融合算子的样例 | Atlas A2训练系列产品/Atlas 800I A2推理产品
| [23_static_tensor_programming_kernellaunch](./23_static_tensor_programming_kernellaunch) | 通过静态Tensor编程方式，实现Add算子的样例 | Atlas A2训练系列产品/Atlas 800I A2推理产品
| [24_simple_hello_world](./24_simple_hello_world) | Ascend C异构混合编程样例, 实现Hello World算子及调用, 支持host/device代码混合编程 | Atlas A2训练系列产品/Atlas 800I A2推理产品
| [25_simple_add](./25_simple_add) | Ascend C异构混合编程样例, 实现Add自定义Vector算子及调用, 支持host/device代码混合编程 | Atlas A2训练系列产品/Atlas 800I A2推理产品
| [26_simple_matmulleakyrelu](./26_simple_matmulleakyrelu) | Ascend C异构混合编程样例, 实现MatmulLeakyRelu自定义Cube+Vector算子及调用, 支持host/device代码混合编程 | Atlas A2训练系列产品/Atlas 800I A2推理产品
| [27_simple_add_cpp_extensions](./27_simple_add_cpp_extensions) | Ascend C异构混合编程样例, 实现Add自定义Vector算子动态库及pybind调用, 支持host/device代码混合编程 | Atlas A2训练系列产品/Atlas 800I A2推理产品
## 获取样例代码<a name="codeready"></a>

 可以使用以下两种方式下载，请选择其中一种进行源码准备。

 - 命令行方式下载（下载时间较长，但步骤简单）。

   ```bash
   # 开发环境，非root用户命令行中执行以下命令下载源码仓。git_clone_path为用户自己创建的某个目录。
   cd ${git_clone_path}
   git clone https://gitee.com/ascend/samples.git
   ```
   **注：如果需要切换到其它tag版本，以v0.5.0为例，可执行以下命令。**
   ```bash
   git checkout v0.5.0
   ```
 - 压缩包方式下载（下载时间较短，但步骤稍微复杂）。

   **注：如果需要下载其它版本代码，请先请根据前置条件说明进行samples仓分支切换。下载压缩包命名跟tag/branch相关，此处以master分支为例，下载的名字将会是samples-master.zip**
   ```bash
   # 1. samples仓右上角选择 【克隆/下载】 下拉框并选择 【下载ZIP】。
   # 2. 将ZIP包上传到开发环境中的普通用户某个目录中，【例如：${git_clone_path}/samples-master.zip】。
   # 3. 开发环境中，执行以下命令，解压zip包。
   cd ${git_clone_path}
   unzip samples-master.zip
   ```

## 更新说明
| 时间       | 更新事项                                     |
| ---------- | -------------------------------------------- |
| 2024/11/11 | 样例目录调整                     |
| 2025/01/06 | 新增21_vectoradd_kernellaunch样例  |
| 2025/07/22 | 新增8_library_frameworklaunch样例       |
| 2025/7/28 | 新增22_baremix_kernellaunch                   |
| 2025/9/22 | 新增Ascend C异构混合编程样例24-27                   |