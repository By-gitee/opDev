## 目录结构介绍
```
├── AclNNInvocation             //通过aclnn调用的方式调用AddNCustom算子
│   ├── inc                     // 头文件目录
│   │   ├── common.h            // 声明公共方法类，用于读取二进制文件
│   │   ├── op_runner.h         // 算子描述声明文件，包含算子输入/输出，算子类型以及输入描述与输出描述
│   │   └── operator_desc.h     // 算子运行相关信息声明文件，包含算子输入/输出个数，输入/输出大小等
│   ├── input                   // 存放脚本生成的输入数据目录
│   ├── output                  // 存放算子运行输出数据和真值数据的目录
│   ├── scripts
│   │   ├── acl.json            // acl配置文件
│   │   ├── gen_data.py         // 输入数据和真值数据生成脚本
│   │   └── verify_result.py    // 真值对比文件
│   ├── src
│   │   ├── CMakeLists.txt      // 编译规则文件
│   │   ├── common.cpp          // 公共函数，读取二进制文件函数的实现文件
│   │   ├── main.cpp            // 单算子调用应用的入口
│   │   ├── op_runner.cpp       // 单算子调用主体流程实现文件
│   │   └── operator_desc.cpp   // 构造算子的输入与输出描述
│   └── run.sh                  // 执行命令脚本
```
## 代码实现介绍
动态输入特性是指，核函数的入参采用ListTensorDesc的结构存储输入数据信息，对应的，调用框架侧需构造TensorList结构保存参数信息，具体如下：

调用框架：

- 在使用aclCreateTensor创建Tensor后，需调用aclCreateTensorList，将创建好的Tensor组成List形式，如下所示，框架侧写法，可参考社区文档-aclnnCat接口用例的写法。

   ```cpp
   inputTensorList = aclCreateTensorList(inputTensor_.data(), inputTensor_.size());
   ```

- 获取算子使用的workspace空间大小的入参，也需使用aclTensorList结构参数，用来计算workspace的大小，调用示例如下。
   ```cpp
   // 获取算子使用的workspace空间大小
   aclnnStatus aclnnAddNCustomGetWorkspaceSize(const aclTensorList *srcList, const aclTensor *out, uint64_t *workspaceSize, aclOpExecutor **executor);
   ```

Host：

- Host侧，获取动态输入信息的调用接口，需使用对应的动态接口，例如，TilingFunc函数和InferShape函数中，GetDynamicInputShape接口用于获取动态输入的shape信息，InferDataType函数中，GetDynamicInputDataType接口用于获取动态输入的数据类型，示例如下。
   ```cpp
  uint32_t totalLength = context->GetDynamicInputShape(0, 0)->GetOriginShape().GetShapeSize();
   ```

- OpDef中，输入数据的参数类型需设置为动态，示例如下。
   ```cpp
    this->Input("srcList")
      .ParamType(DYNAMIC)
   ```

Kernel：

- 核函数入参需传入动态结构的数据，例如GM_ADDR srcList，示例如下。
   ```cpp
  extern "C" __global__ __aicore__ void addn_custom(GM_ADDR srcList, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
   ```

- 对传入的参数srcList，需使用AscendC::ListTensorDesc结构做解析，得到每个Tensor的具体信息，示例如下。
   ```cpp
    AscendC::ListTensorDesc keyListTensorDescInit((__gm__ void*)srcList);
    GM_ADDR x = (__gm__ uint8_t*)keyListTensorDescInit.GetDataPtr<__gm__ uint8_t>(0);
    GM_ADDR y = (__gm__ uint8_t*)keyListTensorDescInit.GetDataPtr<__gm__ uint8_t>(1);
   ```

## 运行样例算子
### 1. 编译算子工程
运行此样例前，请参考[编译算子工程](../README.md#operatorcompile)完成前期准备。
### 2. aclnn调用样例运行

  - 进入到样例目录   
    以命令行方式下载样例代码，master分支为例。
    ```bash
    cd ${git_clone_path}/samples/operator/ascendc/0_introduction/4_addn_frameworklaunch/AclNNInvocation
    ```
  - 样例执行

    样例执行过程中会自动生成测试数据，然后编译与运行aclnn样例，最后检验运行结果。具体过程可参见run.sh脚本。

    ```bash
    bash run.sh
    ```
## 更新说明
| 时间       | 更新事项                     |
| ---------- | ---------------------------- |
| 2024/10/01 | 新增动态输入特性样例 |
| 2024/11/11 | 样例目录调整 |