## 概述
编译工程和自定义工程、assert及debug功能、硬件平台信息的查询能力等。特性样例逐步补充中。

## 自定义算子样例说明
样例通过Ascend C编程语言实现了自定义算子，并按照不同的算子调用方式分别给出了对应的端到端实现。
- FrameworkLaunch：使用框架调用自定义算子。  
  按照工程创建->算子实现->编译部署->算子调用的流程完成算子开发。整个过程都依赖于算子工程：基于工程代码框架完成算子核函数的开发和Tiling实现，通过工程编译脚本完成算子的编译部署，继而实现单算子调用或第三方框架中的算子调用。
- FrameworkLaunchLite：使用msOpGen工具自动生成简易自定义算子工程，并调用自定义算子。  
  按照工程创建->算子实现->编译->算子调用的流程完成算子开发。整个过程都依赖于算子工程：基于工程代码框架完成算子核函数的开发和Tiling实现，通过工程编译脚本完成算子的编译，继而实现单算子调用或第三方框架中的算子调用。
- KernelLaunch：使用核函数直调自定义算子。  
  核函数的基础调用（Kernel Launch）方式，开发者完成算子核函数的开发和Tiling实现后，即可通过AscendCL运行时接口，完成算子的调用。


## 算子开发样例
当前本目录包含的所有样例如下。
|  目录名称                                                   |  功能描述                                              |  运行环境 |
| ------------------------------------------------------------ | ---------------------------------------------------- | -- |
| [0_printf](./0_printf) | 基于Ascend C的Matmul自定义Cube算子并调用printf进行调测的KernelLaunch和FrameworkLaunch调用样例 | Atlas 推理系列产品AI Core<br>Atlas A2训练系列产品/Atlas 800I A2推理产品 |
| [3_assert](./3_assert) | 基于Ascend C的Matmul自定义Cube算子并调用assert进行调测的KernelLaunch和FrameworkLaunch调用样例 | Atlas 推理系列产品AI Core<br>Atlas A2训练系列产品/Atlas 800I A2推理产品 |
| [7_dumptensor](./7_dumptensor) | 基于Ascend C的Add自定义算子和Mmad自定义算子并调用DumpTensor进行调测的KernelLaunch和FrameworkLaunch调用样例 | Atlas 推理系列产品AI Core<br>Atlas A2训练系列产品/Atlas 800I A2推理产品 |

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
| 2025/01/06 | 新增本readme                     |
