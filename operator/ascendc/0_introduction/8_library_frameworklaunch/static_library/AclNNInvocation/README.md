## 目录结构介绍
```
├── AclNNInvocation             //通过aclnn调用的方式调用AddCustom算子和MatmulCustom算子
│   ├── input                   // 存放脚本生成的输入数据目录
│   ├── output                  // 存放算子运行输出数据和真值数据的目录
│   ├── scripts_add             // AddCustom算子相关脚本
│   │   ├── gen_data.py         // 输入数据和真值数据生成脚本
│   │   └── verify_result.py    // 真值对比文件
│   ├── scripts_matmul          // MatmulCustom算子相关脚本
│   │   ├── gen_data.py         // 输入数据和真值数据生成脚本
│   │   └── verify_result.py    // 真值对比文件
│   ├── src
│   │   ├── CMakeLists.txt      // 编译规则文件
│   │   └──  main.cpp           // 单算子调用应用的入口
│   └── run.sh                  // 执行命令脚本
```
## 代码实现介绍
完成自定义算子的开发部署后，可以通过单算子调用的方式来验证单算子的功能。将链接成的动态库链接到生成的可执行程序中，可以实现功能验证。src/main.cpp代码为单算子API执行方式。单算子API执行是基于C语言的API执行算子，无需提供单算子描述文件进行离线模型的转换，直接调用单算子API接口。

自定义算子编译部署后，会自动生成单算子API，可以直接在应用程序中调用。算子API的形式一般定义为“两段式接口”，以AddCustom算子为例，形如：
   ```cpp
   // 获取算子使用的workspace空间大小
   aclnnStatus aclnnAddCustomGetWorkspaceSize(const aclTensor *x, const aclTensor *y, const aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
   // 执行算子
   aclnnStatus aclnnAddCustom(void *workspace, int64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream);
   ```
其中aclnnAddCustomGetWorkspaceSize为第一段接口，主要用于计算本次API调用计算过程中需要多少的workspace内存。获取到本次API计算需要的workspace大小之后，开发者可以按照workspaceSize大小申请Device侧内存，然后调用第二段接口aclnnAddCustom执行计算。具体参考[单算子API调用](https://hiascend.com/document/redirect/CannCommunityAscendCInVorkSingleOp)章节。

CMakeLists.txt是编译规则文件，下面对其如何链接公共动态库进行介绍。
- 设置集成多个算子静态库的公共动态库的存放路径
```bash
set(CUST_PKG_PATH $ENV{BASIC_PATH})
```
- 设置头文件及库文件路径
```bash
# Header path
include_directories(
    ${INC_PATH}
    ${CUST_PKG_PATH}/include
)

# add host lib path
link_directories(
    ${LIB_PATH}
    ${CUST_PKG_PATH}/lib
)
```
- 编译可执行文件
```bash
add_executable(execute_static_op
    main.cpp
)
```
- 链接公共动态库
```bash
target_link_libraries(execute_static_op
    op_runner
    ascendcl
)
```

## 运行样例算子
### 1. 编译运行样例算子
运行此样例前，请参考[编译运行样例算子](../README.md)完成前期准备。
### 2. aclnn调用样例运行

  - 进入到样例目录
    以命令行方式下载样例代码，master分支为例。
    ```bash
    cd ${git_clone_path}/samples/operator/ascendc/0_introduction/8_library_frameworklaunch/static_library/AclNNInvocation
    ```
  - 样例执行

    样例执行过程中会自动生成测试数据，然后编译与运行aclnn样例，最后检验运行结果。具体过程可参见run.sh脚本。

    ```bash
    bash run.sh
    ```
## 更新说明
| 时间       | 更新事项     |
| ---------- | ------------ |
| 2025/07/22 | 新增本readme |
