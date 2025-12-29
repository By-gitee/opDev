<div align="center">
<p align="center">
  <img src="https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/data/samples_pic/CANN_samples.png" align="middle" width = "800" />
</p>
</div>


### 简介

昇腾样例仓提供了一系列媒体数据处理（DVPP/AIPP）、算子开发与调用（Ascend C）、推理应用开发与部署（AscendCL）等场景的丰富代码样例给开发者进行参考学习，帮助开发者快速入门，进而熟练掌握CANN关键特性使用。    

### 目录
- [简介](#简介)
  - [目录](#目录)
  - [版本配套说明](#版本配套说明)
  - [推荐样例](#推荐样例)
  - [仓库结构](#仓库结构)
  - [变更日志](#变更日志)
  - [许可证](#许可证)

## 版本配套说明

- 请参考[CANN社区版文档](https://hiascend.com/document/redirect/CannCommunityInstWizard?utm_source=gitee&utm_medium=sample&utm_campaign=samples)相关章节，对昇腾硬件、CANN软件及相应深度学习框架进行安装准备。
- 本源码仓会适配CANN软件版本创建相应的标签并发行，关于CANN软件版本与本源码仓中标签的配套关系可参见["本源码仓标签与CANN版本配套表"](docs/MATCH.md#cannversionmap)。**需要注意，为确保您的源码定制开发顺利进行，请选择配套的CANN版本与Gitee标签源码，使用master分支可能存在版本不匹配的风险。**

## 仓库结构

```
- /best_practices：CANN最佳实践样例
- /common：samples仓公共文件目录
- /cplusplus：C++样例归档路径（待迁移至/inference）
    |--/contrib：外部贡献推理样例
    |--/level1_single_api：CANN AscendCL接口、DVPP接口、Graph接口等单接口样例
    |--/level2_simple_inference：AscendCL推理应用样例
    |--/...
- /docs：CANN Samples样例使用相关说明、FAQ文档
- /growthpath：开发者CANN的学习路径，帮助开发者快速掌握每一阶段知识点及开发技巧
- /inference：推理应用相关样例
    |--/ACLHelloWorld：AscendCL推理应用入门“Hello World”
    |--/acllite：AscendCL高阶封装接口，包含C++及Python版本
    |--/contributeSamples：开发者贡献样例
    |--/mediaProcess：媒体（音视频）接口相关样例
    |--/memoryManagement：AscendCL内存管理样例
    |--/modelInference：推理应用样例目录，包含C++及Python版本
    |--/...
- /operator：Ascend C算子开发与使用相关样例
    |ascendc
      |0_introduction：简单的示例，适合初学者
      |1_utilities：编译工程和自定义工程、assert及debug功能、硬件平台信息的查询能力等
      |2_features：Ascend C的特性
      |3_libraries：类库的使用示例，包括数学库，激活函数等
      |4_best_practices：最佳实践示例
      |tutorials：生态教学的示例
        |--/AddCustomSample：Ascend C算子开发Add样例
- /operator_contrib：Ascend C算子开发者贡献样例
    |--/UnalignAddCustomSample：Ascend C算子开发Add算子（非对齐）样例
    |--/...
- /python：Python样例归档路径（待迁移至/inference）
    |--/contrib：外部贡献推理样例
    |--/level1_single_api：CANN AscendCL接口、DVPP接口、Graph接口等单接口样例
    |--/level2_simple_inference：AscendCL推理应用样例
    |--/level3_multi_model：多模型串接综合样例
- /robot：昇腾开发板智能车实际应用样例
- /st：样例测试用例，主要用于样例的功能性验证
- /training：训练应用样例

```

## 算子开发样例
|  **样例名称**  |  **样例介绍**  |  **开发语言**  |
|---|---|---|
| [AddCustomSample](https://gitee.com/ascend/samples/tree/master/operator/ascendc/tutorials/AddCustomSample) | 基于Ascend C的Add自定义Vector算子及调用样例 | C++ |
| [HelloWorldSample](https://gitee.com/ascend/samples/tree/master/operator/ascendc/tutorials/HelloWorldSample) | 基于Ascend C的自定义算子调用结构演示样例 | C++ |
| [MatmulCustomSample](https://gitee.com/ascend/samples/tree/master/operator/ascendc/tutorials/MatmulCustomSample) | 基于AscendC的Matmul自定义Cube算子及调用样例 | C++ |
| [MatmulLeakyReluCustomSample](https://gitee.com/ascend/samples/tree/master/operator/ascendc/tutorials/MatmulLeakyReluCustomSample) | 基于AscendC的MatmulLeakyRelu自定义Cube+Vector算子及调用样例 | C++ |
| [UnalignAddCustomSample](https://gitee.com/ascend/samples/tree/master/operator_contrib/UnalignAddCustomSample) | 基于AscendC的Add算子（非对齐）算子及调用样例 | C++ |

## 推理开发样例
|  **样例名称**  |  **样例介绍**  |  **开发语言**  |
|---|---|---|
| [DVPP接口样例](https://gitee.com/ascend/samples/tree/master/cplusplus/level2_simple_inference/0_data_process) | 图像视频处理（DVPP）单接口样例，包含图片视频解码（vdec/jpegd）、缩放（resize）、抠图（crop）、转换（vpc）等功能 |  C++ |
| [sampleResnetQuickStart](https://gitee.com/ascend/samples/tree/master/inference/modelInference/sampleResnetQuickStart) | 推理应用入门样例，基于Resnet50模型实现的图像分类应用 | C++/Python |
| [sampleResnetAIPP](https://gitee.com/ascend/samples/tree/master/inference/modelInference/sampleResnetAIPP) | AIPP特性使用，基于Resnet50模型实现的图像分类应用 | C++/Python |
| [sampleResnetDVPP](https://gitee.com/ascend/samples/tree/master/inference/modelInference/sampleResnetDVPP)  | DVPP特性使用，基于Resnet50模型实现的图像分类应用 | C++/Python |
| [sampleYOLOV7](https://gitee.com/ascend/samples/tree/master/inference/modelInference/sampleYOLOV7) | 使能DVPP+AIPP特性，基于YoloV7模型实现的物体检测应用 |  C++ |
| [sampleResnetRtsp](https://gitee.com/ascend/samples/tree/master/inference/modelInference/sampleResnetRtsp)  | RTSP视频流输入，基于Resnet50模型实现的图像分类应用 |  C++ |
| [sampleCarColor](https://gitee.com/ascend/samples/tree/master/inference/modelInference/sampleCarColor) | 多模型串接，基于YoloV7模型和颜色分类模型实现的检测分类应用 |  C++ |
| [sampleYOLOV7MultiInput](https://gitee.com/ascend/samples/tree/master/inference/modelInference/sampleYOLOV7MultiInput)  | :+1:多路输入综合样例，基于YoloV7模型实现的物体检测应用，支持多路RTSP流/视频输入、支持多卡并行 |  C++ |
| [sampleCrowdCounting](https://gitee.com/ascend/samples/tree/master/inference/modelInference/sampleCrowdCounting) | 统计图片人头数量，基于CrowdCounting模型实现的推理应用 |  Python |
| [sampleYOLOV7NMSONNX](https://gitee.com/ascend/samples/tree/master/inference/modelInference/sampleYOLOV7NMSONNX) | 后处理使用CANN算子进行加速，基于YoloV7模型实现的物体检测应用 |  Python |


## 变更日志
  
| 时间 | 更新事项 |
|----|------|
| 2024/12/25   | 新增基于Ascend C的自定义MC2算子性能调优样例：[MatmulAllReduceCustom](./operator/ascendc/4_best_practices/23_matmul_all_reduce_custom/MatmulAllReduceCustom)|
| 2024/12/25   | 新增基于Ascend C的自定义MC2算子性能调优样例：[MatmulReduceScatterCustom](./operator/ascendc/4_best_practices/22_matmul_reduce_scatter_custom/MatmulReduceScatterCustom)|
| 2024/12/25   | 新增基于Ascend C的自定义MC2算子性能调优样例：[AllGatherMatmulCustom](./operator/ascendc/4_best_practices/21_all_gather_matmul_custom/AllGatherMatmulCustom)|
| 2024/12/12   | 新增基于Ascend C的自定义量化矩阵乘算子样例：[QuantMatMulCustomSample](https://gitee.com/ascend/samples/tree/master/operator_contrib/QuantMatmulCustomSample)|
| 2024/02/23   | 新增pybind11调用算子样例：[MatMulLeakyReluCustomSample/KernelLaunch/CppExtensions](https://gitee.com/ascend/samples/tree/master/operator/MatMulLeakyReluCustomSample/KernelLaunch/CppExtensions)|
| 2024/02/22   | 新增AddcdivCustom算子样例：[AddcdivCustomSample](https://gitee.com/ascend/samples/tree/master/operator/AddcdivCustomSample)|
| 2024/01/26   | 新增基于Ascend C的自定义MatMul算子Kernel Lanunch调用样例：[MatMulCustomSample/KernelLaunch/MatMulInvocationNeo](https://gitee.com/ascend/samples/tree/master/operator/MatMulCustomSample/KernelLaunch/MatMulInvocationNeo)|
| 2024/01/05   | 新增基于Ascend C的自定义Add算子pybind调用样例：[AddCustomSample/FrameworkLaunch/CppExtensions](https://gitee.com/ascend/samples/tree/master/operator/AddCustomSample/FrameworkLaunch/CppExtensions)|
| 2024/01/04   | 新增基于Ascend C的自定义Add算子Kernel Lanunch调用样例：[AddCustomSample/KernelLaunch/AddKernelInvocationNeo](https://gitee.com/ascend/samples/tree/master/operator/AddCustomSample/KernelLaunch/AddKernelInvocationNeo)、[AddCustomSample/KernelLaunch/AddKernelInvocationTilingNeo](https://gitee.com/ascend/samples/tree/master/operator/AddCustomSample/KernelLaunch/AddKernelInvocationTilingNeo)|
| 2023/12/15   | 新增基于Ascend C的自定义LeakyRelu算子onnx调用样例：[LeakyReluCustomSample/FrameworkLaunch/OnnxInvocation](https://gitee.com/ascend/samples/tree/master/operator/LeakyReluCustomSample/FrameworkLaunch/OnnxInvocation)|
| 2023/12/13   | 新增基于Ascend C的自定义算子MoeSoftMaxTopk实现及调用：[MoeSoftMaxTopkCustomSample](https://gitee.com/ascend/samples/tree/master/operator/MoeSoftMaxTopkCustomSample)|
| 2023/12/13   | 新增基于Ascend C的自定义算子LayerNorm实现及调用：[LayerNormCustomSample](https://gitee.com/ascend/samples/tree/master/operator/LayerNormCustomSample)|
| 2023/11/22   | 新增基于Ascend C的自定义算子PreLayerNorm实现及调用：[PreLayerNormSample](https://gitee.com/ascend/samples/tree/master/operator/PreLayerNormSample)|
| 2023/11/09   | 新增基于Ascend C的自定义算子MatMul实现及调用：[MatMulCustomSample](https://gitee.com/ascend/samples/tree/master/operator/MatMulCustomSample)|
| 2023/11/09   | 新增基于Ascend C的自定义算子Sub实现及调用：[SubCustomSample](https://gitee.com/ascend/samples/tree/master/operator/SubCustomSample)|
| 2023/11/09   | 新增基于Ascend C的自定义算子MatmulLeakyrelu实现及调用：[MatmulLeakyreluCustomSample](https://gitee.com/ascend/samples/tree/master/operator/MatMulLeakyReluCustomSample)|
| 2023/10/24   | 新增基于Ascend C的Add自定义算子tensorflow调用样例：[AddCustomSample/FrameworkLaunch/TensorflowInvocation](https://gitee.com/ascend/samples/tree/master/operator/AddCustomSample/FrameworkLaunch/TensorflowInvocation)|
| 2023/08/28   | 新增基于Ascend C的Add自定义算子aclnn调用样例：[AddCustomSample/FrameworkLaunch/AclNNInvocation](https://gitee.com/ascend/samples/tree/master/operator/AddCustomSample/FrameworkLaunch/AclNNInvocation)|
| 2023/08/25   | 新增基于Ascend C的自定义算子核函数直调样例：[AddCustomSample/KernelLaunch/AddKernelInvocation](https://gitee.com/ascend/samples/tree/master/operator/AddCustomSample/KernelLaunch/AddKernelInvocation)、[LeakyReluCustomSample/KernelLaunch](https://gitee.com/ascend/samples/tree/master/operator/LeakyReluCustomSample/KernelLaunch)|
| 2023/08/25   | 新增基于Ascend C的自定义算子实现：[AddCustomSample/FrameworkLaunch/AddCustom](https://gitee.com/ascend/samples/tree/master/operator/AddCustomSample/FrameworkLaunch/AddCustom)、[LeakyReluCustomSample/FrameworkLaunch/LeakyReluCustom](https://gitee.com/ascend/samples/tree/master/operator/LeakyReluCustomSample/FrameworkLaunch/LeakyReluCustom)|
| 2023/05/23   | 新增样例：[sampleYOLOV7NMSONNX](https://gitee.com/ascend/samples/tree/master/inference/modelInference/sampleYOLOV7NMSONNX)|
| 2023/05/17   | 新增样例：[sampleCrowdCounting](https://gitee.com/ascend/samples/tree/master/inference/modelInference/sampleCrowdCounting)|
| 2023/05/16   | 样例新增功能点：[sampleYOLOV7MultiInput](https://gitee.com/ascend/samples/tree/master/inference/modelInference/sampleYOLOV7MultiInput)|
| 2023/05/16   | 新增样例：[sampleCarColor](https://gitee.com/ascend/samples/tree/master/inference/modelInference/sampleCarColor)|
| 2023/05/11   | 新增样例：[sampleResnetRtsp](https://gitee.com/ascend/samples/tree/master/inference/modelInference/sampleResnetRtsp)|
| 2023/04/10   | 新增初级成长路径：[growthpath](https://gitee.com/ascend/samples/tree/master/growthpath) |
| 2023/03/29   | 新增标签：[v0.9.0](https://gitee.com/ascend/samples/tree/v0.9.0/) |
| 2023/03/10   | 新增样例：[ACLHelloWorld](https://gitee.com/ascend/samples/tree/master/inference/ACLHelloWorld) |
| 2023/03/09   | 新增样例：[sampleYOLOV7](https://gitee.com/ascend/samples/tree/master/inference/modelInference/sampleYOLOV7)、[sampleMMDeploy](https://gitee.com/ascend/samples/tree/master/inference/contributeSamples/contrib/samplesMMDeploy)|
| 2023/02/17   | 新增样例：[sampleResnetQuickStart](https://gitee.com/ascend/samples/tree/master/inference/modelInference/sampleResnetQuickStart)、[sampleResnetAIPP](https://gitee.com/ascend/samples/tree/master/inference/modelInference/sampleResnetAIPP)、[sampleResnetDVPP](https://gitee.com/ascend/samples/tree/master/inference/modelInference/sampleResnetDVPP)  |
| 2023/02/10   | 新增目录分支：[operator](https://gitee.com/ascend/samples/tree/master/operator)、[inference](https://gitee.com/ascend/samples/tree/master/inference)、[training](https://gitee.com/ascend/samples/tree/master/training) 分别存放算子、推理、训练相关样例及指导。  |
  
- **历史版本请参考[历史版本信息](https://gitee.com/ascend/samples/tree/master/docs/CHANGELOG.md)**  

## 许可证

[Apache License 2.0](LICENSE)



