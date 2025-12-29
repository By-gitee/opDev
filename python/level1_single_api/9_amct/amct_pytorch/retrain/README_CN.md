# ResNet-101

## 1. 量化感知训练

### 1.1 量化前提

+ **模型准备**  
请下载 [ResNet-101](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/003_Atc_Models/AE/ATC%20Model/resnet-101_nuq/resnet101-5d3b4d8f.pth) 模型文件到 [model](./model/) 目录。

+ **数据集准备**  
由于量化感知训练需要使用大量数据对量化参数进行进一步优化，因此量化感知训练数据需要与原始模型训练数据一致。ResNet-101 的数据集是在 ImageNet 的子集 ILSVRC-2012-CLS 上训练而来，因此需要用户自己准备 ImagenetPytorch 格式的数据集（获取方式请参见[此处](https://github.com/pytorch/examples/tree/master/imagenet)）。
  
  > 如果更换其他数据集，则需要自己进行数据预处理。

### 1.2 量化示例

执行量化示例前，请先检查当前目录下是否包含以下文件及目录：

+ [model](./model/)
  + resnet101-5d3b4d8f.pth
+ [src](./src/)
  + [retrain_conf](./src/retrain_conf/)
    + [retrain.cfg](./src/retrain_conf/retrain.cfg)
  + [\_\_init__.py](./src/__init__.py)
  + [resnet-101_retrain.py](./src/resnet-101_retrain.py)
  + [resnet.py](./src/resnet.py)

请在当前目录执行如下命令运行示例程序：

+ 单卡量化感知训练

  ```bash
  CUDA_VISIBLE_DEVICES=0 python ./src/resnet-101_retrain.py --train_set TRAIN_SET --eval_set EVAL_SET --config_defination ./src/retrain_conf/retrain.cfg
  ```

+ 多卡量化感知训练

  ```bash
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python ./src/resnet-101_retrain.py --train_set TRAIN_SET --eval_set EVAL_SET --config_defination ./src/retrain_conf/retrain.cfg --distributed
  ```

  > 仅支持 distribution 模式的多卡训练，不支持 DataParallel 模式的多卡训练。如果使用 DataParallel 模式的多卡训练，会出现如下错误信息：
  >
  > ```none
  > RuntimeError: Output 54 of BroadcastBackward is a view and its base or another view of its base has been modified inplace. This view is the output of a function that returns multiple views. Such functions do not allow the output view to be modified inplace. You should replace the inplace operation by an out-of-place one.
  > ```

上述命令只给出了常用的参数，不常用参数以及各个参数解释请参见如下表格：

| 参数 | 必填项 | 数据类型 | 默认值 | 参数解释 |
| :-- | :-: | :-: | :-: | :-- |
| -h | 否 | / | / | 显示帮助信息。 |
| --config_defination CONFIG_DEFINATION | 否 | string | None | 量化的简易配置文件路径。 |
| --batch_num BATCH_NUM | 否 | int | 2 | retrain 量化推理阶段的 batch 数。 |
| --train_set TRAIN_SET | 是 | string | None | 测试数据集路径。 |
| --eval_set EVAL_SET | 是 | string | None | 验证数据集路径。 |
|  --num_parallel_reads NUM_PARALLEL_READS | 否 | int | 4 | 用于读取数据集的线程数，根据硬件运算能力酌情调整。 |
| --batch_size BATCH_SIZE | 否 | int | 25 | PyTorch 执行一次前向推理所使用的样本数量，根据内存或显存大小酌情调整。 |
| --learning_rate LEARNING_RATE | 否 | float | 1e-5 | 学习率。 |
| --train_iter TRAIN_ITER | 否 | int | 2000 | 训练迭代次数。 |
| --print_freq PRINT_FREQ | 否 | int | 10 | 训练及测试信息的打印频率。 |
| --dist_url DIST_URL | 否 | string | tcp://127.0.0.1:50011 | 初始化多卡训练通信进程的方法。 |
| --distributed | 否 | / | / | 使用该参数表示进行多卡训练，否则不进行多卡训练。 |

若出现如下信息，则说明量化成功：

```none
[INFO] ResNet101 before retrain top1:77.37% top5:93.55%
[INFO] ResNet101 after retrain top1:77.42% top5:93.56%
```

### 1.3 量化结果

量化成功后，在当前目录会生成量化日志文件 ./amct_log/amct_pytorch.log 和 ./outputs/retrain 文件夹，该文件夹内包含以下内容：

+ tmp: 临时文件夹
  + config.json: 量化配置文件，描述了如何对模型中的每一层进行量化。
  + record.txt: 量化因子记录文件记录量化因子。
  + model_best.pth.tar: PyTorch 模型量化感知训练过程中生成的 checkpoint 中间文件。
+ ResNet101_deploy_model.onnx: 量化部署模型，即量化后的可在昇腾 AI 处理器部署的模型文件。
+ ResNet101_fake_quant_model.onnx: 量化仿真模型，即量化后的可在 ONNX 执行框架 ONNX Runtime 进行精度仿真的模型

> 如果量化脚本所在目录下已经存在量化配置文件，则再次调用 `create_quant_config` 接口时，如果新生成的量化配置文件与已有的文件同名，则会覆盖已有的量化配置文件，否则生成新的量化配置文件。

