# KV Cache量化

## 1 KV Cache量化前提

- 安装依赖

本sample依赖包可参考[requirements.txt](requirements.txt)

- 模型和数据集准备

本sample以Llama2-7b模型和c4/realnewslike数据集为示例，请用户自行下载。



## 2 KV Cache量化示例

### 2.1 使用接口方式调用

**step 1.**  请在当前目录执行如下命令运行示例程序得到量化因子记录文件，用户根据实际情况修改示例程序中的模型和数据集路径：

`CUDA_VISIBLE_DEVICES=0,1 python3 src/run_kv_cache_quant.py`

推理成功后，在当前目录会生成量化日志文件./amct_log/amct_pytorch.log和./outputs文件夹，该文件夹内包含以下内容：

- config.json：量化配置文件，描述了如何对模型中的每一层进行量化。
- record.txt：量化因子记录文件。

**step 2.** 用户可参考src/reference_function.py文件中的read_kv_cache_factors函数读取record文件中的量化因子，参考do_quant函数和do_antiquant函数进行量化和反量化。用户自行修改模型后即可进行模型推理。

> 如果outputs目录下已经存在量化配置文件或量化因子记录文件，再次运行示例程序时，如果新生成的文件与已有文件同名，则会覆盖已有的量化配置文件或量化因子记录文件。

### 2.2 使用单算子方式调用

如果用户需要支持更多算子类型，或者用户自定义了其他操作，则可以使用单算子方式进行构图，然后进行量化校准，并输出量化因子记录文件。

**step 1.** 请参考src/quant_calibration_op_demo.py文件对模型进行修改后，进行模型推理得到量化因子记录文件。

**step 2.** 可参考2.1中step 2修改模型后即可进行模型推理。进行此步骤时请注释step 1修改的代码防止record文件被覆盖。

