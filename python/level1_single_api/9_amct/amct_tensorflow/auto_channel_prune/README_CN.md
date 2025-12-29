# ResNet-50 V1

## 1. 自动通道稀疏及通道稀疏重训练

### 1.1 稀疏前提

+ **模型准备**  
请下载 [ResNet-50](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/resnet-50_v1_retrain/pre_model.zip) 模型文件。解压并将 pre_model 文件夹内的文件放到 [model](./model/) 目录。其中 ResNet50_train.meta 用于重训练，ResNet50_eval.meta 用于验证。

+ **数据集准备**  
由于重训练需要使用大量数据对量化参数进行进一步优化，因此重训练数据需要与模型训练数据一致。ResNet-50 的数据集是在 ImageNet 的子集 ILSVRC-2012-CLS 上训练而来，因此需要用户自己准备 TFRecord 格式的数据集。

> 如果更换其他数据集，则需要自己进行数据预处理。

### 1.2 自动通道稀疏示例

执行自动通道稀疏示例前，请先检查当前目录下是否包含以下文件及目录：

+ [model/](./model/)
  + resnet_v1_50.data-00000-of-00001
  + resnet_v1_50.index
  + resnet_v1_50_eval.meta
  + resnet_v1_50_train.meta
+ [src](./src/)
  + [retrain_conf](./src/retrain_conf/)
    + [sample_auto_prune.cfg](./src/retrain_conf/sample_auto_prune.cfg)
  + [resnet-50_v1_auto_prune.py](./src/resnet-50_v1_auto_prune.py)

在当前目录执行如下命令：

```bash
python ./src/resnet-50_v1_auto_prune.py --train_set TRAIN_SET --eval_set EVAL_SET --config_defination CONFIG
```

上述命令只给出了常用的参数，不常用参数以及各个参数解释请参见如下表格：

| 参数 | 必填项 | 数据类型 | 默认值 | 参数解释 |
| :-- | :-: | :-: | :-: | :-- |
| -h | 否 | / | / | 显示帮助信息。 |
| --config_defination CONFIG_defination | 是 | string | None | 简易配置文件路径（自动通道稀疏搜索：./src/retrain_conf/sample_auto_prune.cfg）。 |
| --batch_num BATCH_NUM | 否 | int| 2 | retrain 量化推理阶段的 batch 数。 |
| --train_set TRAIN_SET | 是 | string | None | 测试数据集路径。 |
| --train_keyword TRAIN_KEYWORD | 否 | string | None | 用于筛选训练集路径下包含该关键词的文件，若未定义，则默认训练集路径下所有文件作为训练集。 |
| --eval_set EVAL_SET | 是 | string | None | 验证数据集路径。 |
| --eval_keyword EVAL_KEYWORD | 否 | string | None | 用于筛选训练集路径下包含该关键词的文件，若未定义，则默认验证集路径下所有文件作为验证集。 |
| --train_model TRAIN_MODEL | 是 | string | ./model/resnet_v1_50_train.meta | 训练用模型路径。 |
| --eval_model EVAL_MODEL | 是 | string | ./model/resnet_v1_50_eval.meta | 验证模型路径。 |
| --num_parallel_reads NUM_PARALLEL_READS | 否 | int | 4 | 用于读取数据集的线程数，根据硬件运算能力酌情调整。 |
| --buffer_size BUFFER_SIZE | 否 | int | 1000 | 数据集乱序的缓存大小，根据内存空间酌情调整。 |
| --repeat_count REPEAT_COUNT | 否 | int | 0 | 数据集重复次数，若为0则无限循环。 |
| --batch_size BATCH_SIZE | 否 | int | 32 | TensorFlow 运行一次所使用的样本数量，根据内存或显存大小酌情调整。 |
| --ckpt CKPT_PATH | 否 | string | ./model/resnet_v1_50 | ResNet-50 V1 模型的官方权重 checkpoint 文件路径。 |
| --learning_rate LEARNING_RATE | 否 | float | 1e-6 | 学习率。 |
| --save_interval SAVE_INTERVAL | 否 | int | 500 | 重训练保存间隔。 |
| --momentum MOMENTUM | 否 | float | 0.9 | RMSPropOptimizer优化器的动量。 |
| --train_iter TRAIN_ITER | 否 | int | 100 | 训练迭代次数。 |
| --test_iter TRAIN_ITER | 否 | int | 1 | 自动通道稀疏搜索中的校验数据迭代次数。 |

> **注意**：如果测试数据集和验证数据集位于同一路径下，为确保量化过程中使用了正确的数据集，该场景下量化命令中需要追加 `--train_keyword TRAIN_KEYWORD` 和 `--eval_keyword EVAL_KEYWORD` 参数，根据上述两个参数过滤相关文件名，确保 `--train_set` 参数使用的是测试数据集，`--eval_set` 使用的是验证数据集。

若出现如下信息则说明模型稀疏成功：

```none
INFO - [AMCT]:[save_model]: The model is saved in ./outputs/retrain/resnet_v1_50_quantized.pb
The origin model top 1 accuracy = 66.2%
The origin model top 5 accuracy = 87.8%
The model after retrain top 1 accuracy = 0.4%.
The model after retrain top 5 accuracy = 1.0%.
```

> 本示例脚本仅用于展示自动通道稀疏搜索及通道稀疏重训练流程，可以增大迭代训练次数 (`--train_iter`) 来缩小重训练后精度损失。

### 1.3 稀疏结果

自动稀疏成功后，在当前目录会生成日志文件 ./amct_log/amct_tensorflow.log 和 ./outputs/retrain 文件夹，该文件夹内包含以下内容：

+ checkpoint: 训练检查点。
+ events.out.tfevents.xxxxxxxxxx.xxx: 训练记录，包含重训练时的 loss 信息，可使用 TensorBoard 查看。
+ record.txt: 稀疏记录文件，记录可稀疏结点间的级联关系。
+ resnet_v1_50_origin.pb: 原始推理模型。
+ resnet_v1_50_mask.pb: 插入mask算子的模型，通道稀疏过程的中间文件。
+ resnet_v1_50_quantized.pb: 稀疏模型，可在 TensorFlow 环境进行精度仿真并可在昇腾 AI 处理器部署。
+ resnet_v1_50_retrain-0.data-00000-of-00001
+ resnet_v1_50_retrain-0.index
+ resnet_v1_50_retrain-0.meta
+ resnet_v1_50_retrain-100.data-00000-of-00001
+ resnet_v1_50_retrain-100.index
+ resnet_v1_50_retrain-100.meta
+ prune_cfg.cfg: 自动通道稀疏搜索输出的稀疏配置文件。

> 对该模型重新进行重训练时，上述结果文件将会被覆盖。