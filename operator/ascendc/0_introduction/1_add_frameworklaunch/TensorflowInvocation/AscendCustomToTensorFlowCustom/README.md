## 概述
本样例展示了如何使用Ascend C自定义算子AddCustom映射到TensorFlow自定义算子AddCustom，并通过TensorFlow调用Ascend C算子。

## 运行样例算子
### 1.环境准备
请参考与您当前使用的版本配套的[《TensorFlow 1.15模型迁移》](https://www.hiascend.com/document/redirect/canncommercial-tfmigr115)或[《TensorFlow 2.6.5模型迁移》](https://www.hiascend.com/document/redirect/canncommercial-tfmigr26)中的“安装框架插件包”章节，获取详细的安装指南和步骤。

### 2.编译算子工程
运行此样例前，请参考[编译算子工程](../../README.md#operatorcompile)完成前期准备。  

需注意插件代码适配，路径为： samples/operator/ascendc/0_introduction/1_add_frameworklaunch/AddCustom/framework/tf_plugin/tensorflow_add_custom_plugin.cc
需修改插件代码中的TensorFlow调用算子名称OriginOpType为"AddCustom"，如下所示：
```c++
REGISTER_CUSTOM_OP("AddCustom")
  .FrameworkType(TENSORFLOW)      // type: CAFFE, TENSORFLOW
  .OriginOpType("AddCustom")      // name in tf module
  .ParseParamsByOperatorFn(AutoMappingByOpFn);
```

### 3.TensorFlow调用的方式调用样例运行

  - 进入到样例目录   
    以命令行方式下载样例代码，master分支为例。
    ```bash
    cd ${git_clone_path}/samples/operator/ascendc/0_introduction/1_add_frameworklaunch/TensorflowInvocation/AscendCustomToTensorFlowCustom
    ```
  - 编译TensorFlow算子库
    ```bash
    bash build.sh
    ```

  - 样例执行(TensorFlow1.15)

    样例执行过程中会自动生成随机测试数据，然后通过TensorFlow调用算子，最后对比TensorFlow原生算子和Ascend C算子运行结果。具体过程可参见run_add_custom_tf_1.15.py脚本。
    ```bash
    python3 run_add_custom_tf_1.15.py
    ```
  - 样例执行(TensorFlow2.6.5)
    样例执行过程中会自动生成随机测试数据，然后通过TensorFlow调用算子，最后对比TensorFlow原生算子和Ascend C算子运行结果。具体过程可参见run_add_custom_tf_2.6.5.py脚本。
    ```bash
    python3 run_add_custom_tf_2.6.5.py
    ```


## 更新说明
| 时间       | 更新事项     |
| ---------- | ------------ |
| 2024/09/30 | 新增本readme及样例 |
| 2024/11/11 | 样例目录调整 |
| 2025/12/16 | 更新安装依赖包 |