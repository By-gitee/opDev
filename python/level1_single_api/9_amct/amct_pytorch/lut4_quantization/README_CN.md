# LUT4bit量化

## 1 LUT4bit量化前提

### 1.1 安装依赖

本sample依赖包可参考[requirements.txt](requirements.txt)

### 1.2 模型和数据集准备

本sample以Llama2-7b模型,pileval和wikitext2数据集为示例，请用户自行下载

### 1.3 简易量化配置
./src/lut4_quant.cfg文件为用户自定义的简易量化配置，具体表示信息如下：

| 字段 |类型| 说明 | 默认值 | 取值范围 | 注意事项 |
|:--| :-: | :-- | :-: | :-: | :-: |
|batch_num|uint32|量化使用的batch数量 |1|/|校准使用batch数与推理使用输入数据有关，是校准脚本中的batch_num|
|skip_layers|str|跳过量化的层 |/|/|跳过量化层支持模糊匹配，当配置字符串为层名字串，或与层名一致时，跳过该层量化，不生成量化配置。字符串必须包含数字或字母|
|weight_only_config.weight_compress_only|bool|是否为仅权重量化|False|True/False|LUT4bit量化目前仅支持权重量化，需要设置为True|
|weight_only_config.wts_type|enum|量化后权重类型|INT8|本sample支持INT4|/|
|weight_only_config.weight_granularity|enum|权重量化粒度|PER_TENSOR|PER_TENSOR/PER_CHANNEL/PER_GROUP|LUT4bit仅支持PER_GROUP模式|
|weight_only_config.round_mode|enum|舍入模式|/|HYBRID/ROUND/RINT|LUT4bit仅支持RINT模式|
|weight_only_config.lut_quantize.lut_alog|enum|lut量化算法模式|CLUSTER|CLUSTER/ATCTAN|

## 2 LUT4量化示例

### 2.1 使用接口方式调用

**step 1.**  请在当前目录执行如下两条命令运行示例程序，用户需根据实际情况修改示例程序中的模型和数据集路径：

校准:
`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python3 src/run_llama7b_calibration.py --calibration_data=/pile_val_backup/ --model=/data/Models/pytorch/Llama2/Llama2_7b_hf`
- 校准可以使用--finetune, 入参格式是bool,用来表示做精调/粗调


保存并推理量化模型:
`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python3 src/save_llama7b_quant_model.py --verify_data=/data/Datasets/wikitext/wikitext-2-raw-v1/wikitext-2-raw/wikiscript.py --model=/data/Models/pytorch/Llama2/Llama2_7b_hf`

若出现如下信息，则说明校准成功：

```none
Calibration success, time taken:  56.0 min  20.263916969299316 s
```

出现如下信息，说明量化成功

```none
Test time taken:  7.0 min  12.269736528396606 s
Score:  5.595210552215576
```

**step 2.**  推理成功后，在当前目录会生成量化日志文件./amct_log/amct_pytorch.log和./outputs文件夹，该文件夹内包含以下内容：

- config.json：量化配置文件，描述了如何对模型中的每一层进行量化。
- record.txt：量化因子记录文件。
- lut_result.pt：lut算法参数文件。

> 如果outputs目录下已经存在量化配置文件或量化因子记录文件，再次运行示例程序时，如果新生成的文件与已有文件同名，则会覆盖已有的量化配置文件或量化因子记录文件。

**LLMHelper:**  定义用于大语言模型量化校准的辅助类，核心参数有:校准模型，校准数据集，前向方法，校准模块，校准模块推理方法，学习率，迭代次数，是否开启量化层筛选，量化误差比例阈值，量化误差平均阈值。详细使用方式可查阅AMCT使用手册
