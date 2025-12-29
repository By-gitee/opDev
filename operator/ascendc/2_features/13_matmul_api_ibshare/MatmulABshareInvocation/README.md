## 目录结构介绍
```
├── MatmutABshareInvocation
│   ├── cmake                               // 编译工程文件
│   ├── pictures
│   │   ├── matmul_ABshare.png              // 算子ABshare的数据处理示意图
│   │   └── matmul_noABshare.png            // 算子NoABshare的数据处理示意图
│   ├── scripts
│   │   ├── verify_result.py                // 真值对比文件
│   │   └── gen_data.py                     // 输入数据和真值数据生成脚本文件
│   ├── CMakeLists.txt                      // 编译工程文件
│   ├── data_utils.h                        // 数据读入写出函数
│   ├── main.cpp                            // 主函数，调用算子的应用程序, 只有NPU域调用
│   ├── matmul_commom_custom_tiling.cpp     // 算子tiling实现
│   ├── matmul_noABshare_custom.cpp         // 算子非ABshare的kernel实现
│   ├── matmul_ABshare_custom.cpp           // 算子ABshare的kernel实现
│   └── run.sh                              // 编译运行算子的脚本
```
## 代码实现介绍
本样例中实现的是[m, n, k]固定为[128, 256, 384]的MatmutlABshare算子和MatmulNoABshare算子。
- kernel实现  
  MatmutABshare算子的数学表达式为：
  ```
  C = A * B
  ```
  其中A的形状为[128, 384]，B的形状为[384, 256]，C的形状为[128, 256]。具体请参考[matmul_ABshare_custom.cpp](./matmul_ABshare_custom.cpp)。
  MatmutNoABshare算子数学表达式与MatmutABshare一致，具体请参考[matmul_noABshare_custom.cpp](./matmul_noABshare_custom.cpp)。

  MatmulABshare算子代码数据处理说明图示(A矩阵和B矩阵不切分处理)：  
  ![alt text](./pictures/matmul_ABshare.png)  
  MatmulNoABshare算子代码数据处理说明图示(A矩阵和B矩阵按K列切分处理)：  
  ![alt text](./pictures/matmul_noABshare.png)  

- 调用实现  
  1. NPU侧运行验证主要通过使用ACLRT_LAUNCH_KERNEL内核调用宏来完成；
  2. 通过std::chrono::steady_clock::now()方法，计算ACLRT_LAUNCH_KERNEL内核调用宏执行时间，详细的性能数据可通过msprof等工具获取，该处测量方法只是简易示例；
  3. 调用不同的核函数，计算出各自核函数执行时间，计算出性能提升百分比。具体参考[main.cpp](./main.cpp)。

## 运行样例算子
  - 打开样例目录   
    以命令行方式下载样例代码，master分支为例。
    ```bash
    cd ${git_clone_path}/samples/operator/ascendc/2_features/13_matmul_api_ibshare/MatmulABshareInvocation
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
    bash run.sh -r npu -v [SOC_VERSION]
    ```
    - SOC_VERSION：昇腾AI处理器型号，如果无法确定具体的[SOC_VERSION]，则在安装昇腾AI处理器的服务器执行npu-smi info命令进行查询，在查询到的“Name”前增加Ascend信息，例如“Name”对应取值为xxxyy，实际配置的[SOC_VERSION]值为Ascendxxxyy。支持以下参数取值（xxx请替换为具体取值）：
      - Atlas A2训练系列产品/Atlas 800I A2推理产品参数值：AscendxxxB1、AscendxxxB2、AscendxxxB3、AscendxxxB4

## 更新说明
| 时间       | 更新事项     | 注意事项                                         |
| ---------- | ------------ | ------------------------------------------------ |
| 2025/01/22 | 更新本readme |                                                 |
| 2024/11/04 | 新增readme   |                                                 |