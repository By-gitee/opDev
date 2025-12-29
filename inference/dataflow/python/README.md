# Python样例使用指导
## 目录结构
.   
├── README.md      
├── config   
│   ├── add 用例中tensorflow计算图文件,可以通过添加后缀pb打开查看   
│   ├── add_func.json 用例中udf配置文件    
│   ├── add_graph.json 用例中计算图编译配置文件  
│   ├── data_flow_deploy_info.json 用例中部署位置配置文件  
│   ├── multi_model_deploy.json sample_multiple_model用例中部署位置配置文件  
│   ├── simple_model 用例中的onnx计算图文件，可以通过添加后缀onnx打开查看  
│   └── invoke_func.json 用例中udf_call_nn的编译配置文件  
├── sample1.py 样例1展示了基本的DataFlow API构图，包含UDF，GraphPp和UDF执行NN推理三种类型节点的构造和执行    
├── sample2.py 样例2展示了python dataflow调用 udf python的过程   
├── sample3.py 样例3展示了使能异常上报的样例   
├── sample_pytorch.py 该样例展示了DataFlow结合pytorch进行模型的在线推理    
├── sample_npu_model.py 该样例展示了DataFlow在线推理时，udf使用分别npu_model实现模型下沉和数据下沉场景  
├── sample_multiple_model.py 将pytorch模型直接通过装饰器构造成funcPp，同时和执行onnx模型/pb模型的GraphPp进行串接   
├── test_perf.py 性能打点样例   
├── udf_py   
│   ├── udf_add.py 使用python实现udf多func功能  
│   └── udf_control.py 使用python实现udf功能，用于控制udf_add中多func实际执行的func  
└── udf_py_ws_sample 完整样例用于说明python udf实现     
    ├── CMakeLists.txt udf python完整工程cmake文件样例   
    ├── func_add.json  udf python完整工程配置文件样例   
    ├── src_cpp   
    │   └── func_add.cpp udf python完整工程C++源码文件样例    
    └── src_python   
        └── func_add.py  udf python完整工程python源码文件样例   


## 环境准备
参考[环境准备](../../../README.md#环境准备)下载安装驱动/固件/CANN软件包   
python 版本要求：python3.11 具体版本以dataflow wheel包编译时用的python版本为准，如果需要使用不同python版本，可以参考[py_dflow](../py_dflow)重新编译dataflow wheel包。
sample_pytorch.py、sample_npu_model.py样例依赖pytorch和torchvision包,推荐使用torch 2.1.0和torchvision 0.16.0


## 运行样例
下文中numa_config.json文件配置参考[numa_config字段说明及样例](https://www.hiascend.com/document/detail/zh/canncommercial/800/developmentguide/graph/dataflowcdevg/dfdevc_23_0031.html)
```bash
# 可选
export ASCEND_GLOBAL_LOG_LEVEL=3       #0 debug 1 info 2 warn 3 error 不设置默认error级别
export ASCEND_SLOG_PRINT_TO_STDOUT=1   # 日志打屏，不设置日志落盘默认路径
# 必选
source {HOME}/Ascend/ascend-toolkit/set_env.sh #{HOME}为CANN软件包安装目录，请根据实际安装路径进行替换
export RESOURCE_CONFIG_PATH=xxx/xxx/xxx/numa_config.json

python3.11 sample1.py
python3.11 sample2.py
python3.11 sample3.py
python3.11 sample_pytorch.py
python3.11 sample_npu_model.py
python3.11 sample_multiple_model.py
python3.11 test_perf.py
```

