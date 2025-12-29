## 简化MatmulLeakyRelu算子直调样例
本样例以MatmulLeakyRelu算子为示例，展示了一种更为简单的算子编译流程，支持main函数和Kernel函数在同一个cpp文件中实现。

## 目录结构介绍
```
├── 26_simple_matmulleakyrelu
│   ├── CMakeLists.txt          // 编译工程文件
│   ├── data_utils.h            // 数据读入写出函数
│   ├── matmul_leakyrelu.asc    // AscendC算子实现 & 调用样例
│   ├── run.sh                  // 运行脚本
│   └── scripts
│       ├── gen_data.py         // 输入数据和真值数据生成脚本文件
│       └── verify_result.py    // 真值对比文件
```

## 算子描述
算子使用了MatmulLeakyRelu高阶API，实现了快速的MatmulLeakyRelu矩阵乘法的运算操作。

MatmulLeakyRelu的计算公式为：

```
C = A * B + Bias
C = C > 0 ? C : C * 0.001
```

- A、B为源操作数，A为左矩阵，形状为\[M, K]；B为右矩阵，形状为\[K, N]。
- C为目的操作数，存放矩阵乘结果的矩阵，形状为\[M, N]。
- Bias为矩阵乘偏置，形状为\[N]。对A*B结果矩阵的每一行都采用该Bias进行偏置。

## 算子规格描述
<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">MatmulLeakyRelu</td></tr>
</tr>
<tr><td rowspan="4" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">a</td><td align="center">1024 * 256</td><td align="center">float16</td><td align="center">ND</td></tr>
<tr><td align="center">b</td><td align="center">256 * 640</td><td align="center">float16</td><td align="center">ND</td></tr>
<tr><td align="center">bias</td><td align="center">640</td><td align="center">float</td><td align="center">ND</td></tr>
</tr>
</tr>
<tr><td rowspan="1" align="center">算子输出</td><td align="center">c</td><td align="center">1024 * 640</td><td align="center">float</td><td align="center">ND</td></tr>
</tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">matmul_leakyrelu_custom</td></tr>
</table>

## 代码实现介绍
本样例中实现的是[m, n, k]固定为[1024, 640, 256]的MatmulLeakyRelu算子。
- kernel实现  
  MatmulLeakyRelu算子的数学表达式为：
  ```
  C = A * B + Bias
  C = C > 0 ? C : C * 0.001
  ```
  其中A的形状为[1024, 256]，B的形状为[256, 640]，C的形状为[1024, 640]，Bias的形状为[640]。具体请参考[matmul_leakyrelu.asc](./matmul_leakyrelu.asc)。

- 调用实现  
  使用内核调用符<<<>>>调用核函数。

## 支持的产品型号
本样例支持如下产品型号：
- Atlas A2 训练系列产品/Atlas 800I A2 推理产品

## 运行样例算子
  - 打开样例目录   
    以命令行方式下载样例代码，master分支为例。
    ```bash
    cd ${git_clone_path}/samples/operator/ascendc/0_introduction/26_simple_matmulleakyrelu
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
    配置安装路径后，执行以下命令统一配置环境变量。
    ```bash
    # 配置CANN环境变量
    source ${ASCEND_INSTALL_PATH}/bin/setenv.bash
    # 添加AscendC CMake Module搜索路径至环境变量
    export CMAKE_PREFIX_PATH=${ASCEND_INSTALL_PATH}/compiler/tikcpp/ascendc_kernel_cmake:$CMAKE_PREFIX_PATH
    ```
  - 样例执行
    ```bash
    bash run.sh  # 编译并执行样例
    ```


## 更新说明
| 时间       | 更新事项     |
| ---------- | ------------ |
| 2025/09/22 | 新增本readme |