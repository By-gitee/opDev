# MXFP4量化

## 1 MXFP4量化前提

### 1.1 安装依赖

本sample依赖包可参考[requirements.txt](requirements.txt)

### 1.2 模型和数据集准备

本sample以Llama2-7b模型,pileval和wikitext2数据集为示例，请用户自行下载，并适配utils.py文件中get_loader数据获取函数中的文件路径。当前sample中数据集保存目录需根据实际保存目录修改。

### 1.3 简易量化配置
./src/mxfp4_quant.cfg文件为用户自定义的简易量化配置，具体表示信息如下：

| 字段 |类型| 说明 | 默认值 | 取值范围 | 注意事项 |
|:--| :-: | :-- | :-: | :-: | :-: |
|batch_num|uint32|量化使用的batch数量 |1|/|MXFP量化中配置不生效，校准使用batch数与推理使用输入数据有关，是校准脚本中的batch_num|
|skip_layers|str|跳过量化的层 |/|/|跳过量化层支持模糊匹配，当配置字符串为层名字串，或与层名一致时，跳过该层量化，不生成量化配置。字符串必须包含数字或字母|
|weight_only_config.weight_compress_only|bool|是否为仅权重量化|False|True/False|MXFP4量化目前仅支持权重量化，需要设置为True|
|weight_only_config.wts_type|enum|量化后权重类型|INT8|INT8/MXFP4_E2M1|/|
|weight_only_config.weight_granularity|enum|权重量化粒度|PER_TENSOR|PER_TENSOR/PER_CHANNEL/PER_GROUP|MXFP4_E2M1仅支持PER_GROUP模式|
|weight_only_config.round_mode|enum|舍入模式|/|HYBRID/ROUND/RINT|MXFP4_E2M1仅支持RINT模式|
|weight_only_config.awq_quantize.grids_num|int|搜索格点数量|20|/|/|

## 2 MXFP4量化示例

### 2.1 使用接口方式调用

**step 1.**  请在当前目录执行如下命令运行示例程序，用户需根据实际情况修改示例程序中的模型和数据集路径：

`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python3 src/run_llama7b_quantization.py`

若出现如下信息，则说明量化成功：

```none
Test time taken:  1.0 min  59.24865388870239 s
Score:  5.670858383178711
```

推理成功后，在当前目录会生成量化日志文件./amct_log/amct_pytorch.log和./outputs文件夹，该文件夹内包含以下内容：

- config.json：量化配置文件，描述了如何对模型中的每一层进行量化。
- record.txt：量化因子记录文件。
- awq_result.pt：awq算法参数文件。

> 如果outputs目录下已经存在量化配置文件或量化因子记录文件，再次运行示例程序时，如果新生成的文件与已有文件同名，则会覆盖已有的量化配置文件或量化因子记录文件。
