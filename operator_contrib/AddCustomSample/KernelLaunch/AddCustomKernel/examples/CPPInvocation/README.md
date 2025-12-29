## 概述

通过C++接口调用核函数直调实现的不带Tiling的AddCustom算子

## 目录结构介绍

```
├── CPPInvocation
│   ├── scripts
        └── gen_data.py       // 输入数据和标杆数据构造脚本
        └── verify_result.py  // 标杆数据和自定义算子输出数据对比脚本
│   ├── CMakeLists.txt        // cmake编译文件
│   ├── main.cpp              // 算子调用代码
│   ├── data_utils.h          // 数据类型定义,数据读取代码
│   ├── run.sh                // 编译运行算子的脚本
```

## 运行样例算子

  **请确保已根据算子包编译部署步骤完成本算子的编译部署动作。**

  - 进入样例代码所在路径

  ```bash
 cd ${git_clone_path}/samples/operator_contrib/AddCustomSample/KernelLaunch/AddCustomKernel/examples/CPPInvocation
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