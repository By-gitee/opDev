# FP8/HIF8量化

## 1 FP8/HIF8量化前提

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
|weight_only_config.wts_type|enum|量化后权重类型|INT8|INT8/MXFP4_E2M1/HIFLOAT8/FLOAT8_E4M3FN|

## 2 FLOAT8_E4M3FN量化示例
> 当前quantization.cfg文件中weight_only_config.wts_type设置的值为FLOAT8_E4M3FN，如果需要HIFLOAT8仅权重量化，请适配修改quantization.cfg


### 2.1 使用接口方式调用

请在当前目录执行如下命令运行示例程序

验证fakequant模型脚本：

`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python3 src/run_llama7b_quantization.py --test_on_npu_flag=false --calibration_data=/pile_val_backup/ --verify_data=/data/Datasets/wikitext/wikitext-2-raw-v1/wikitext-2-raw/wikiscript.py --model=/data/Models/pytorch/Llama2/Llama2_7b_hf`

验证deploy模型脚本（需要适配npu相关环境）：

`python3 src/run_llama7b_quantization.py --test_on_npu_flag=true`

> test_on_npu_flag参数表明是否生成部署模型在npu上推理，calibration_data参数为校准集路径，verify_data为验证集的路径，model为模型存放路径

若出现如下信息，则说明量化成功：

```none
Test time taken:  1.0 min  38.24865388870239 s
Score:  5.48
```

推理成功后，在当前目录会生成量化日志文件./amct_log/amct_pytorch.log和./output文件夹，该文件夹内包含以下内容：

- config.json：量化配置文件，描述了如何对模型中的每一层进行量化。
- record.txt：量化因子记录文件。

> 如果outputs目录下已经存在量化配置文件或量化因子记录文件，再次运行示例程序时，如果新生成的文件与已有文件同名，则会覆盖已有的量化配置文件或量化因子记录文件。
