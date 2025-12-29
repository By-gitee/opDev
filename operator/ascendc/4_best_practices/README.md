## 概述

[Ascend C最佳实践](https://hiascend.com/document/redirect/CannCommunityAscendCBestPractice)样例。样例逐步补充中。

## 算子开发样例

| 目录名称                        | 功能描述                                   | 运行环境                                   |
| ------------------------------- | ------------------------------------------ | ------------------------------------------ |
| [4_bank_conflict](./4_bank_conflict) | 基于Ascend C的bank冲突性能优化样例 | Atlas A2训练系列产品/Atlas 800I A2推理产品 |
| [6_group_matmul](./6_group_matmul) | 基于Ascend C的group matmul算子性能优化样例 | Atlas A2训练系列产品/Atlas 800I A2推理产品 |
| [12_l2_cache_bypass](./12_l2_cache_bypass) | 基于Ascend C的L2 CaCheMode算子性能优化样例 | Atlas A2训练系列产品/Atlas 800I A2推理产品 |
| [15_mata_address_conflict](./15_mata_address_conflict) | 基于Ascend C的同地址冲突性能优化样例 | Atlas A2训练系列产品/Atlas 800I A2推理产品 |
| [21_all_gather_matmul_custom](./21_all_gather_matmul_custom) | 基于Ascend C的AllGatherMatmul算子性能调优样例 | Atlas A2训练系列产品 |
| [22_matmul_reduce_scatter_custom](./22_matmul_reduce_scatter_custom) | 基于Ascend C的MatmulReduceScatter算子性能调优样例 | Atlas A2训练系列产品 |
| [23_matmul_all_reduce_custom](./23_matmul_all_reduce_custom) | 基于Ascend C的MatmulAllReduce算子性能调优样例 | Atlas A2训练系列产品/Atlas 800I A2推理产品 |


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
| 2025/07/14 | 新增12_l2_cache_bypass样例         |
| 2025/07/03 | 新增15_mata_address_conflict样例         |
| 2025/07/01 | 新增4_bank_conflict样例         |
| 2024/12/19 | 新增23_matmul_all_reduce_custom样例         |
| 2024/12/19 | 新增22_matmul_reduce_scatter_custom样例         |
| 2024/12/19 | 新增21_all_gather_matmul_custom样例         |
| 2024/11/20 | 新增6_group_matmul样例                     |
