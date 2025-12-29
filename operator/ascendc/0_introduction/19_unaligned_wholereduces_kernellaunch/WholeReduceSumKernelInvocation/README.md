## 目录结构介绍
```
├── WholeReduceSumKernelInvocation
│   ├── cmake                               // 编译工程文件
│   ├── scripts
│   │   ├── gen_data.py                     // 输入数据和真值数据生成脚本
│   │   └── verify_result.py                // 验证输出数据和真值数据是否一致的验证脚本
│   ├── whole_reduce_sum_custom_tiling.h    // 算子tiling实现
│   ├── whole_reduce_sum_custom.cpp         // 算子kernel实现
│   ├── CMakeLists.txt                      // 编译工程文件
│   ├── data_utils.h                        // 数据读入写出函数
│   ├── main.cpp                            // 主函数，调用算子的应用程序，含CPU域及NPU域调用
│   └── run.sh                              // 编译运行算子的脚本
```
## 代码实现介绍
- kernel实现  
  非对齐WholeReduceSum算子对二维Tensor输入作行归约求和。其python代码表示如下：
  ```python
  y = np.sum(x, axis=1)
  ```
  非对齐WholeReduceSum算子的实现流程分为3个基本任务：CopyIn，Compute，CopyOut。CopyIn任务负责将Global Memory上的输入Tensor x非对齐搬运到Local Memory，存储在xLocal中，Compute任务负责对xLocal执行按行规约求和操作，计算结果存储在yLocal中，CopyOut任务负责将输出数据从yLocal非对齐搬运至Global Memory上的输出Tensor y中。具体实现请参考[whole_reduce_sum_custom.cpp](./whole_reduce_sum_custom.cpp)。以下是CopyIn，Comput和CopyOut流程的详细说明。


  本样例的输入x的shape为[13, 123]，数据类型为float16，每行的数据（246B）不满足32B对齐约束。输出y的shape为[13]，数据类型为float16，长度为26B，也不满足32B对齐。由于输入Tensor单行数据不满足32B对齐约束，我们把x从GM搬入到xLocal时应该使用DataCopyPad接口进行非对齐搬运。**注意到输出y也不满足32B对齐，我们申请y的Global Memory Buffer时应该向上32B对齐，避免搬出时访问非法内存。**
  ```cpp
  __aicore__ inline void CopyIn()
  {
      AscendC::LocalTensor<datatype> xLocal = inQueueX.AllocTensor<datatype>();
      uint32_t colBytes = this->cols * sizeof(datatype);
      // 每行数据有colBytes，输入共有rows行，故重复搬运rows次
      AscendC::DataCopyExtParams copyParams = {(uint16_t)this->rows, colBytes, 0, 0, 0};
      // 每行补充rpad个数据，使其满足32B对齐
      uint8_t rpad = (this->colAligned - this->cols * sizeof(datatype)) / sizeof(datatype);
      AscendC::DataCopyPadExtParams<datatype> padParams = {false, 0, rpad, 0};
      // 将输入x从GM非对齐搬运到UB
      AscendC::DataCopyPad<datatype>(xLocal, xGm, copyParams, padParams);
      inQueueX.EnQue(xLocal);
      }
  ```
  我们在UB上计算行和时，应注意每行只应计算前cols个数据，因此需要通过mask等参数控制WholeReduceSum高阶API的行为。
  ```cpp
  __aicore__ inline void Compute()
  {
      AscendC::LocalTensor<datatype> xLocal = inQueueX.DeQue<datatype>();
      AscendC::LocalTensor<datatype> yLocal = outQueueY.AllocTensor<datatype>();

      // 经过DataCopyPad对齐后，每行的datablock数目
      int32_t srcStride = this->colAligned / 32;
      // mask连续模式，只对每行前cols个数据求和
      // 总共有rows行，故重复rows次求和操作
      // 每次运算操作向前移动srcStride个datablock
      AscendC::WholeReduceSum<datatype, true>(yLocal, xLocal, this->cols, this->rows, 1, 1, srcStride);
      outQueueY.EnQue<datatype>(yLocal);
      inQueueX.FreeTensor(xLocal);
  }
  ```
  将yLocal从UB搬出到GM，由于只需要搬出reducesBytes，需要用DataCopyPad进行非对齐搬出。
  ```cpp
  __aicore__ inline void CopyOut()
  {
      AscendC::LocalTensor<datatype> yLocal = outQueueY.DeQue<datatype>();
      uint16_t reducedBytes = this->rows * sizeof(datatype);
      // 单次非对齐搬出reducedBytes
      AscendC::DataCopyExtParams copyParams = {1, reducedBytes, 0, 0, 0};
      AscendC::DataCopyPad<datatype>(yGm, yLocal, copyParams);
      outQueueY.FreeTensor(yLocal);
  }
  ```

- tiling实现  
  TilingData参数设计，TilingData参数本质上是和并行数据切分相关的参数，本示例算子使用了3个tiling参数：totalLength，rows，cols 。totalLength是指需要计算的数据量大小，rows是指二维输入Tensor的行数，cols则是指每行的数据个数。通过将totalLength，rows，cols传递到kernel侧，就可以实现将输入数据按行切分，然后规约求和。tiling实现代码中通过上下文获取输入输出的shape信息，并对应设置TilingData。

- 调用实现
  1. CPU侧运行验证主要通过ICPU_RUN_KF CPU调测宏等CPU调测库提供的接口来完成；
  2. NPU侧运行验证主要通过使用ACLRT_LAUNCH_KERNEL内核调用宏来完成。

  应用程序通过ASCENDC_CPU_DEBUG 宏区分代码逻辑运行于CPU侧还是NPU侧。

## 运行样例算子
  - 打开样例目录   
    以命令行方式下载样例代码，master分支为例。
    ```bash
    cd ${git_clone_path}/samples/operator/ascendc/0_introduction/19_unaligned_wholereduces_kernellaunch/WholeReduceSumKernelInvocation
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
    - SOC_VERSION：昇腾AI处理器型号，如果无法确定具体的[SOC_VERSION]，则在安装昇腾AI处理器的服务器执行npu-smi info命令进行查询，在查询到的“Name”前增加Ascend信息，例如“Name”对应取值为xxxyy，实际配置的[SOC_VERSION]值为Ascendxxxyy。支持以下参数取值（xxx请替换为具体取值）：
      - Atlas A2训练系列产品/Atlas 800I A2推理产品参数值：AscendxxxB1、AscendxxxB2、AscendxxxB3、AscendxxxB4


    示例如下。
    ```bash
    bash run.sh -r cpu -v Ascendxxxyy
    ```
## 更新说明
| 时间       | 更新事项     |
| ---------- | ------------ |
| 2024/09/04 | 新增本样例 |
| 2024/11/11 | 样例目录调整 |