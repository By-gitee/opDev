## 简化HelloWorld算子直调样例
本样例通过使用<<<>>>内核调用符来完成算子核函数在NPU侧运行验证的基础流程，核函数内通过printf打印输出结果。

## 目录结构介绍
```
├── 24_simple_helloworld
│   ├── CMakeLists.txt      // 编译工程文件
│   └── hello_world.asc     // AscendC算子实现 & 调用样例
```

## 支持的产品型号
本样例支持如下产品型号：
- Atlas A2 训练系列产品/Atlas 800I A2 推理产品

## 运行样例算子
  - 打开样例目录
    以命令行方式下载样例代码，master分支为例。
    ```bash
    cd ${git_clone_path}/samples/operator/ascendc/0_introduction/24_simple_helloworld
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
    mkdir -p build && cd build;   # 创建并进入build目录
    cmake ..;make -j;             # 编译工程
    ./demo                        # 执行样例
    ```

## 更新说明
| 时间       | 更新事项     |
| ---------- | ------------ |
| 2025/09/15 | 新增本readme |