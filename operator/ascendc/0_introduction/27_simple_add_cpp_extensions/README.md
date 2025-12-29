## 简化Pybind算子直调样例
本样例使用pybind方式调用核函数，以带有Tiling的Add算子为示例，展示了一种更为简单的算子编译流程，支持main函数和Kernel函数在同一个cpp文件中实现。

## 目录结构介绍
```
├── 27_simple_add_cpp_extensions
│   ├── CMakeLists.txt        // 编译工程文件
│   ├── add_custom_test.py    // python调用脚本
│   ├── add_custom.asc        // AscendC算子实现 & Pybind封装
│   └── run.sh                // 编译运行算子的脚本
```
## 代码实现介绍
- kernel实现  
  Add算子的数学表达式为：
  ```
  z = x + y
  ```
  计算逻辑是：Ascend C提供的矢量计算接口的操作元素都为LocalTensor，输入数据需要先搬运进片上存储，然后使用计算接口完成两个输入参数相加，得到最终结果，再搬出到外部存储上。

  Add算子的实现流程分为3个基本任务：CopyIn，Compute，CopyOut。CopyIn任务负责将Global Memory上的输入Tensor xGm和yGm搬运到Local Memory，分别存储在xLocal、yLocal，Compute任务负责对xLocal、yLocal执行加法操作，计算结果存储在zLocal中，CopyOut任务负责将输出数据从zLocal搬运至Global Memory上的输出Tensor zGm中。具体请参考[add_custom.asc](./add_custom.asc)。

- 调用实现  
  通过PyTorch框架进行模型的训练、推理时，会调用到很多算子进行计算，调用方式也和kernel编译流程相关。对于自定义算子工程，需要使用PyTorch Ascend Adapter中的OP-Plugin算子插件对功能进行扩展，让torch可以直接调用自定义算子包中的算子；对于简化KernelLaunch开放式算子编程的方式，也可以使用pytorch调用，此样例演示的就是这种算子调用方式。

  add_custom.asc文件是一个代码示例，使用了pybind11库来将C++代码封装成Python模块。该代码实现中定义了一个名为m的pybind11模块，其中包含一个名为run_add_custom的函数。该函数与my_add::run_add_custom函数相同，用于将C++函数转成Python函数。在函数实现中，通过c10_npu::getCurrentNPUStream() 的函数获取当前NPU上的流，通过内核调用符<<<>>>调用自定义的Kernel函数add_custom，在NPU上执行算子。

  在add_custom_test.py调用脚本中，通过导入自定义模块add_custom，调用自定义模块add_custom中的run_add_custom函数，在NPU上执行x和y的加法操作，并将结果保存在变量z中。

## 支持的产品型号
本样例支持如下产品型号：
- Atlas A2 训练系列产品/Atlas 800I A2 推理产品

## 运行样例算子
  - 安装pytorch (这里使用2.1.0版本为例)

    **aarch64:**

    ```bash
    pip3 install torch==2.1.0
    ```

    **x86:**

    ```bash
    pip3 install torch==2.1.0+cpu  --index-url https://download.pytorch.org/whl/cpu
    ```

  - 安装torch-npu （以Pytorch2.1.0、python3.9、CANN版本8.0.RC1.alpha002为例）

    ```bash
    git clone https://gitee.com/ascend/pytorch.git -b v6.0.rc1.alpha002-pytorch2.1.0
    cd pytorch/
    bash ci/build.sh --python=3.9
    pip3 install dist/*.whl
    ```

    安装pybind11
    ```bash
    pip3 install pybind11
    ```

  - 打开样例目录   
    以命令行方式下载样例代码，master分支为例。
    ```bash
    cd ${git_clone_path}/samples/operator/ascendc/0_introduction/27_simple_add_cpp_extensions
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
    rm -rf build; mkdir -p build; cd build  # 创建并进入build目录
    cmake ..; make -j                       # 编译算子so
    python3 ../add_custom_test.py           # 执行样例
    ```

## 更新说明
| 时间       | 更新事项     |
| ---------- | ------------ |
| 2025/09/22 | 新增本readme |
