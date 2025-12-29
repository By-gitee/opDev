## 目录结构介绍
```
├── OpRunner                    // 对多个静态库的集成和使用
│   ├── inc                     // 头文件目录
│   │   ├── common.h            // 声明公共方法类，用于读取二进制文件
│   │   ├── op_runner.h         // 算子描述声明文件，包含算子输入/输出，算子类型以及输入描述与输出描述
│   │   └── operator_desc.h     // 算子运行相关信息声明文件，包含算子输入/输出个数，输入/输出大小等
│   ├── src
│   │   ├── CMakeLists.txt      // 编译规则文件
│   │   ├── common.cpp          // 公共函数，读取二进制文件函数的实现文件
│   │   ├── op_runner.cpp       // 单算子调用主体流程实现文件
│   │   └── operator_desc.cpp   // 构造算子的输入与输出描述
│   └── run.sh                  // 执行命令脚本
```

下面对CMakeLists.txt编译规则文件进行介绍。
- 设置算子包存放路径
```bash
set(CUST_PKG_ADD_PATH $ENV{DDK_PATH_ADD})
set(CUST_PKG_MATMUL_PATH $ENV{DDK_PATH_MATMUL})
```
- 编译一个公共动态库
```bash
add_library(op_runner SHARED
    operator_desc.cpp
    op_runner.cpp
    common.cpp
)
```
- 将两个算子静态库加到公共动态库中
```bash
find_package(add_custom REQUIRED
    PATHS ${CUST_PKG_ADD_PATH}
    NO_DEFAULT_PATH
)

find_package(matmul_custom REQUIRED
    PATHS ${CUST_PKG_MATMUL_PATH}
    NO_DEFAULT_PATH
)

target_link_libraries(op_runner PRIVATE
    ascendcl
    add_custom::static
    matmul_custom::static
    nnopbase
)
```
- 将生成的功能动态库安装到指定输出目录中
```bash
install(TARGETS op_runner DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
```



## 执行命令
    - 进入到样例目录
    ```bash
    cd ${git_clone_path}/samples/operator/ascendc/0_introduction/8_library_frameworklaunch/static_library/OpRunner
    ```
    - 链接静态库
    ```bash
    bash run.sh
    ```
  命令执行成功后，会在父目录static_library目录下，生成output目录，存放生成的libop_runner.so动态库。

## 更新说明
| 时间       | 更新事项     |
| ---------- | ------------ |
| 2025/07/22 | 新增本readme |
