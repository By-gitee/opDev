## 概述

通过Python接口调用核函数直调实现的带Tiling的AddCustom算子

## 目录结构介绍

```
├── PythonInvocation
│   ├── add_custom_test.py    // add_custom python调用测试代码 
│   ├── CMakeLists.txt        // cmake编译文件
│   ├── pybind11.cpp          // pybind绑定核函数和python接口代码
│   ├── run.sh                // 编译运行算子的脚本
│   ├── README.md             // 样例指导手册
```

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
    安装expecttest
    ```bash
    pip3 install expecttest
    ```

  **请确保已根据算子包编译部署步骤完成本算子的编译部署动作。**

  - 进入样例代码所在路径

  ```bash
 cd ${git_clone_path}/samples/operator_contrib/AddCustomSample/KernelLaunch/AddCustomTilingKernel/examples/PythonInvocation
  ```

  - 样例执行

    样例执行过程中会自动生成测试数据，然后编译与运行C++调用样例，最后打印运行结果。

    ```bash
    bash run.sh
    ```

## 更新说明

| 时间       | 更新事项     |
| ---------- | ------------ |
| 2025/05/19 | 样例首次提交 |