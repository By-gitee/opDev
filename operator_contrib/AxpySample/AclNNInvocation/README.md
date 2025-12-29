## 目录结构介绍
```
├── AclNNInvocation             //通过aclnn调用的方式调用AxpyCustom算子
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
完成自定义算子的开发部署后，可以通过单算子调用的方式来验证单算子的功能。src/main.cpp代码为单算子API执行方式。单算子API执行是基于C语言的API执行算子，无需提供单算子描述文件进行离线模型的转换，直接调用单算子API接口。

自定义算子编译部署后，会自动生成单算子API，可以直接在应用程序中调用。算子API的形式一般定义为“两段式接口”，形如：
   ```cpp
   // 获取算子使用的workspace空间大小
   aclnnStatus aclnnAddCustomGetWorkspaceSize(const aclTensor *x, const aclTensor *y, const alcTensor *out, uint64_t workspaceSize, aclOpExecutor **executor);
   // 执行算子
   aclnnStatus aclnnAddCustom(void *workspace, int64_t workspaceSize, aclOpExecutor **executor, aclrtStream stream);
   ```
其中aclnnAddCustomGetWorkspaceSize为第一段接口，主要用于计算本次API调用计算过程中需要多少的workspace内存。获取到本次API计算需要的workspace大小之后，按照workspaceSize大小申请Device侧内存，然后调用第二段接口aclnnAddCustom执行计算。具体参考[AscendCL单算子调用](https://hiascend.com/document/redirect/CannCommunityAscendCInVorkSingleOp)>单算子API执行 章节。

## 运行样例算子
### 1. 编译算子工程
运行此样例前，需要编译以及部署AxpyCustom算子。
### 2. aclnn调用样例运行
样例执行
样例执行过程中会自动生成测试数据，然后编译与运行aclnn样例，最后检验运行结果。具体过程可参见run.sh脚本。

```bash
bash run.sh
```

### 3. 使用msprof工具测试算子性能
在完成aclnn调用样例运行后，执行以下命令：
    ```bash
    cd output
    msprof op execute_add_op
    ```
​    便会生成算子在此案例下的性能数据文件，性能数据含义可以查看此文档[关键字段说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/devaids/opdev/optool/atlasopdev_16_0093.html)

### 4. 使用msprof工具跑流水图
首先需要将测试案例shape修改为[4096, 2048],因为大shape跑时间长。
之后进行aclnn调用样例，执行以下命令：

```bash
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/tools/simulator/AscendXXXXX/lib:$LD_LIBRARY_PATH
#其中AscendXXXXX为您的硬件型号，具体可通过npu-smi info 命令查询
msprof op simulator execute_add_op
```

[流水图查看方式链接](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/devaids/opdev/optool/atlasopdev_16_0087.html)