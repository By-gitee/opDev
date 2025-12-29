
# ACLLite

## 产品介绍

AclLite库是对当前开源社区样例中：ACL设备资源、DVPP图像和视频处理、模型推理接口等重复代码进行封装，为用户提供的一组简易公共接口。

注意：

1.本库仅供当前社区开源样例使用，不覆盖ascend平台应用开发的所有场景，不作为用户应用开发的标准库；

2.该公共库仅作为一个参考，用户可以根据自身使用习惯和实际业务对性能的要求，根据官方接口文档和现有代码进行二次开发或重构。

## 快速部署

- C++版本的ACLLite公共库安装配置步骤请参考：[ACLLite-C++安装步骤](./cplusplus/README.md)。

- Python版本的ACLLite公共库安装配置步骤请参考：[ACLLite-Python安装步骤](./python/README.md)。

## 推理样例

- 使用ACLLite/cplusplus接口进行应用推理演示可参考[sampleResnetDVPP](../modelInference/sampleResnetDVPP/cppACLLite)。

- 使用ACLLite/python接口进行应用推理演示可参考[sampleResnetDVPP](../modelInference/sampleResnetDVPP/pyACLLite)。

## 主要特性

- 支持cpp、python语言；

- 支持图片处理、视频处理、模型加载与执行、多线程管理等特性。

## 相关文档

- C++版本的ACLLite公共库API接口说明请参考：[C++版本API接口说明](./cplusplus/API.md)。

- Python版本的ACLLite公共库API接口说明请参考：[Python版本API接口说明](./python/API.md)。

- 更多AscendCL相关说明请参考[昇腾文档](https://www.hiascend.com/document)

## FAQ

暂无

## 许可证

[Apache License 2.0](LICENSE)

## 版本说明

| 时间  | 更新事项  |
|---|---|
| 2023/08/08  | 更新ACLLite/README说明  |