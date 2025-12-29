# FP4伪量化

## 1 FP4伪量化

### 1.1 安装依赖

本sample依赖包可参考[requirements.txt](requirements.txt)

### 1.2 模型和数据集准备

本sample以Llama2-7b模型，pileval和wikitext2数据集为示例，请用户自行下载。

### 1.3 简易量化配置
./src/quantization.cfg文件为用户自定义的简易量化配置，具体表示信息如下：

| 字段 |类型| 说明 | 默认值 | 取值范围 |
|:--| :-: | :-- | :-: | :-: |
|skip_layers|str|跳过量化的层 |/|/|
|weight_only_config.weight_compress_only|bool|是否为仅权重量化|False|True/False|
|weight_only_config.wts_type|enum|量化后权重类型|INT8|INT8/MXFP4_E2M1/HIFLOAT8/FLOAT8_E4M3FN/FLOAT4_E2M1/FLOAT4_E1M2|
|weight_only_config.awq_quantize.grids_num|uint32|awq搜索格点数量|20|1~4294967295（整数）|

## 2 FLOAT4_E2M1量化示例
> 当前quantization.cfg文件中weight_only_config.wts_type设置的值为FLOAT4_E2M1


### 2.1 使用接口方式调用

请在当前目录执行如下命令运行示例程序

验证fakequant模型脚本：

`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python3 src/run_llama7b_quantization.py --calibration_data=/pile_val_backup/ --verify_data=/data/Datasets/wikitext/wikitext-2-raw-v1/wikitext-2-raw/wikiscript.py --model=/data/Models/pytorch/Llama2/Llama2_7b_hf`


若出现如下信息，则说明量化成功：

```none
Test time taken:  9.0 min  38.24865388870239 s
Score:  5.657759
```

推理成功后，在当前目录会生成量化日志文件./amct_log/amct_pytorch.log和./output文件夹，该文件夹内包含以下内容：

- config.json：量化配置文件，描述了如何对模型中的每一层进行量化。
- record.txt：量化因子记录文件。
- awq_result.pt：存储了awq算法的的scale和clip
- quant_factor.pt：存储量化缩放因子

> 如果outputs目录下已经存在量化配置文件或量化因子记录文件，再次运行示例程序时，如果新生成的文件与已有文件同名，则会覆盖已有的量化配置文件或量化因子记录文件。
