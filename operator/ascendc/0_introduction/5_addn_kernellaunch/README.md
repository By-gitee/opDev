## 概述
本样例基于5_addn_kernellaunch算子工程，介绍了单算子直调工程动态输入特性。

## 目录结构介绍
```
├── 5_addn_kernellaunch         // 使用核函数直调的方式调用AddN自定义算子
│   ├── cmake                   // 编译工程文件
│   ├── scripts
│   │   ├── gen_data.py         // 输入数据和真值数据生成脚本
│   │   └── verify_result.py    // 验证输出数据和真值数据是否一致的验证脚本
│   ├── addn_custom.cpp          // 算子kernel实现
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   ├── main.cpp                // 主函数，调用算子的应用程序，含CPU域及NPU域调用
│   └── run.sh                  // 编译运行算子的脚本
```

## 算子描述
AddN算子实现了两个数据相加，返回相加结果的功能，其中核函数的输入参数为动态输入，动态输入参数包含两个入参，x和y。对应的数学表达式为：  
```
z = x + y
```
## 算子规格描述
<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">AddN</td></tr>
</tr>
<tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x（动态输入参数srcList[0]）</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
<tr><td align="center">y（动态输入参数srcList[1]）</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
</tr>
</tr>
<tr><td rowspan="1" align="center">算子输出</td><td align="center">z</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
</tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">addn_custom</td></tr>
</table>

## 支持的产品型号
- Atlas 推理系列产品AI Core
- Atlas A2训练系列产品/Atlas 800I A2推理产品

## 代码实现介绍
动态输入特性是指，核函数的入参采用ListTensorDesc的结构存储输入数据信息，对应的，调用框架侧需构造TensorList结构保存参数信息，具体如下：

框架侧：

- 构造TensorList数据结构，示例如下。

   ```cpp
   constexpr uint32_t SHAPE_DIM = 2;
    struct TensorDesc {
        uint32_t dim{SHAPE_DIM};
        uint32_t index;
        uint64_t shape[SHAPE_DIM] = {8, 2048};
    };

    constexpr uint32_t TENSOR_DESC_NUM = 2;
    struct ListTensorDesc {
        uint64_t ptrOffset;
        TensorDesc tensorDesc[TENSOR_DESC_NUM];
        uintptr_t dataPtr[TENSOR_DESC_NUM];
    } inputDesc;
   ```

- 将申请分配的Tensor入参组合成ListTensorDesc的数据结构，示例如下。

  ```cpp
  inputDesc = {(1 + (1 + SHAPE_DIM) * TENSOR_DESC_NUM) * sizeof(uint64_t),
                 {xDesc, yDesc},
                 {(uintptr_t)xDevice, (uintptr_t)yDevice}};
   ```

kernel侧:

- 按照框架侧传入的数据格式，解析出对应的各入参，示例如下。

  ```cpp
    uint64_t buf[SHAPE_DIM] = {0};
    AscendC::TensorDesc<int32_t> tensorDesc;
    tensorDesc.SetShapeAddr(buf);
    listTensorDesc.GetDesc(tensorDesc, 0);
    uint64_t totalLength = tensorDesc.GetShape(0) * tensorDesc.GetShape(1);
    __gm__ uint8_t *x = listTensorDesc.GetDataPtr<__gm__ uint8_t>(0);
    __gm__ uint8_t *y = listTensorDesc.GetDataPtr<__gm__ uint8_t>(1);
   ```

## 运行样例算子
  - 打开样例目录   
    以命令行方式下载样例代码，master分支为例。
    ```bash
    cd ${git_clone_path}/samples/operator/ascendc/0_introduction/5_addn_kernellaunch/
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
    - RUN_MODE：编译方式，可选择CPU调试，NPU仿真，NPU上板。支持参数为[cpu / sim / npu]
    - SOC_VERSION：昇腾AI处理器型号，如果无法确定具体的[SOC_VERSION]，则在安装昇腾AI处理器的服务器执行npu-smi info命令进行查询，在查询到的“Name”前增加Ascend信息，例如“Name”对应取值为xxxyy，实际配置的[SOC_VERSION]值为Ascendxxxyy。支持以下产品型号：
      - Atlas 推理系列产品AI Core
      - Atlas A2训练系列产品/Atlas 800I A2推理产品

    示例如下，Ascendxxxyy请替换为实际的AI处理器型号。
    ```bash
    bash run.sh -r cpu -v Ascendxxxyy
    ```
## 更新说明
| 时间       | 更新事项     |
| ---------- | ------------ |
| 2024/10/01 | 新增直调方式动态输入样例 |
| 2024/11/11 | 样例目录调整 |