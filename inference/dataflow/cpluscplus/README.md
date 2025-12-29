# C++样例使用指导
## 目录结构

├── CMakeLists.txt cmake配置文件  
├── README.md  指导  
├── config    
│   ├── add 用例中使用tensorflow计算图文件,可以通过添加后缀pb打开查看   
│   ├── add_func_multi.json 用例中使用的多func的配置文件   
│   ├── add_func_multi_control.json 用例中使用的多func的配置文件    
│   ├── add_func.json  用例中add FunctionPp对应的配置文件  
│   ├── add_graph.json  用例中add GraphPp对应的配置文件  
│   ├── deploy_info.json 用例中指定节点部署位置的配置文件  
│   └── invoke_func.json 用例中udf调用nn对应的配置文件  
├── node_builder.h  构造functionPp和GraphPp的公共方法  
├── sample1.cpp  样例1展示了基本的DataFlow API构图，包含UDF，GraphPp和UDF执行NN推理三种类型节点的构造和执行  
├── sample2.cpp  样例2展示了TimeBatch使用方法  
├── sample3.cpp  样例3展示了CountBatch使用方法  
├── sample4.cpp  样例4展示了Tensorflow图构造Dataflow节点的方法  
├── sample5.cpp  样例5展示了多func的调用方法   
├── sample6.cpp  样例6展示了开启异常上报的方法  
└── test_perf.cpp 测试Feed和Fetch接口性能  

## 环境要求
参考[环境准备](../../../README.md#环境准备)下载安装驱动/固件/CANN软件包   
python 版本要求：python3.9

## 程序编译
```bash
source {HOME}/Ascend/ascend-toolkit/set_env.sh # "{HOME}/Ascend"为CANN软件包安装目录，请根据实际安装路径进行替换。
mkdir build
cd build
cmake ..
make -j 64
cd ..
```

## 运行样例
下文中numa_config.json文件配置参考[numa_config字段说明及样例](https://www.hiascend.com/document/detail/zh/canncommercial/800/developmentguide/graph/dataflowcdevg/dfdevc_23_0031.html)
```bash
# 可选
export ASCEND_GLOBAL_LOG_LEVEL=3       #0 debug 1 info 2 warn 3 error 不设置默认error级别
export ASCEND_SLOG_PRINT_TO_STDOUT=1   # 日志打屏，不设置日志落盘默认路径
# 必选
source {HOME}/Ascend/ascend-toolkit/set_env.sh # "{HOME}/Ascend"为CANN软件包安装目录，请根据实际安装路径进行替换。
export RESOURCE_CONFIG_PATH=xxx/xxx/xxx/numa_config.json

cd output
./sample1
./sample2
./sample3
./sample4
./sample5
./sample6
./test_perf
```
```

