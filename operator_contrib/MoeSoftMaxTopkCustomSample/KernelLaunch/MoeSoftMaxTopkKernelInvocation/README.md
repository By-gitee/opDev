## 目录结构介绍
``` 
├── MoeSoftMaxTopkKernelInvocation
│   ├── cmake                   // 编译工程文件
│   ├── scripts
│   │   ├── gen_data.py         // 输入数据和真值数据生成脚本
│   │   └── verify_result.py    // 验证输出数据和真值数据是否一致的验证脚本
│   ├── moe_soft_max_topk.cpp          // 算子kernel实现
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   ├── main.cpp                // 主函数，调用算子的应用程序，含CPU域及NPU域调用
│   └── run.sh                  // 编译运行算子的脚本
``` 
## 代码实现介绍
本样例中实现的是固定shape的MoeSoftMaxTopk算子。
- kernel实现   
 MoeSoftMaxTopk算子的功能为：MoeSoftMaxTopk算子是softmax和topk的融合算子，其中softmax可以理解为对x计算最后一维每个数据的概率，在计算结果中筛选出k个最大结果，输出对应的y值和索引indices。

  计算逻辑是：Ascend C提供的矢量计算接口的操作元素都为LocalTensor，输入数据需要先搬运进片上存储，然后使用计算接口完成x的softmax操作，将计算后的数据进行筛选，得到k个最大值，再搬出到外部存储上。   

  MoeSoftMaxTopk算子的实现流程分为3个基本任务：CopyIn, Compute, CopyOut。CopyIn任务负责将Global Memory上的输入Tensor xGm, kGm搬运至Local Memory，分别存储在xLocal, kLocal, Compute任务负责对xLocal, kLocal执行相关操作，计算结果存储在outLocal中，CopyOut任务负责将输出数据从outLocal搬运至Global Memory上的输出Tensor outGm中。具体请参考[moe_soft_max_topk.cpp](./moe_soft_max_topk.cpp)。

- 调用实现  
  1.&nbsp;CPU侧运行验证主要通过ICPU_RUN_KF CPU调测宏等CPU调测库提供的接口来完成；  
  2.&nbsp;NPU侧运行验证主要通过使用<<<>>>内核调用符来完成。    
应用程序通过ASCENDC_CPU_DEBUG 宏区分代码逻辑运行于CPU侧还是NPU侧。
## 运行样例算子
- 打开样例目录

  ```bash
  cd ${git_clone_path}/samples/operator_contrib/MoeSoftMaxTopkCustomSample/KernelLaunch/MoeSoftMaxTopkKernelInvocation/
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
      ````
    
    配置仿真模式日志文件目录，默认为sim_log。
    ```bash
    export CAMODEL_LOG_PATH=./sim_log
    ```

- 样例执行

  ```bash
  bash run.sh -r [RUN_MODE] -v  [SOC_VERSION] 
  ```
  - RUN_MODE：编译方式，可选择CPU调试，NPU仿真，NPU上板。支持参数为[cpu / sim / npu]，默认值为cpu。
  - SOC_VERSION：昇腾AI处理器型号，如果无法确定具体的[SOC_VERSION]，则在安装昇腾AI处理器的服务器执行npu-smi info命令进行查询，在查询到的“Name”前增加Ascend信息，例如“Name”对应取值为xxxyy，实际配置的[SOC_VERSION]值为Ascendxxxyy。支持以下参数取值（xxx请替换为具体取值）：
    - Atlas A2训练系列产品参数值：AscendxxxB1、AscendxxxB2、AscendxxxB3、AscendxxxB4

  示例如下。
  ```bash
  bash run.sh -r cpu -v Ascend910B4
  ```   
## 更新说明
  | 时间 | 更新事项 |
|----|------|
| 2023/7/2 | 新增本readme |