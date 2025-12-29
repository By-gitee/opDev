# 介绍
实现TorchAir导出的air模型的加载和执行
# 目录结构
├── CMakeLists.txt # 编译脚本    
├── README.md     # 说明文档    
└── src # 样例源码   
    ├── llm_config.cpp   # 解析存储LLM模型相关配置，如增量图/全量图shape，dataType等信息    
    ├── llm_config.h    
    ├── main.cpp         # 样例执行入口，初始化，加载air模型，执行全量图增量图，和释放资源    
    ├── utils.cpp        # 工具类，包含字符串按照分隔符解析，获取绝对路径，字符串转整形等工具函数    
    └── utils.h    
# 源码编译
```bash
# 此处以CANN开发套件实际安装路径为准
export ASCEND_INSTALL_PATH=$HOME/Ascend/ascend-toolkit/latest
mkdir build
cd build
cmake ..
make -j 64
```
# 配置文件说明
样例执行需要配置一些参数，这些参数通过配置文件的方式被执行进程读取解析和使用，配置文件使用key=value的方式对各个配置项进行设置，文件中各配置项说明如下。详情可以参考[配置文件](../config/config.txt)：
| 配置项    | 说明      | 样例    |   是否必填 |
| :-------- | :-------- | :------ | :------ |
| fullInputShapes | 全量图输入shape，顺序与模型中Data节点index顺序一致，同一个输入不同维度之间用逗号分隔，不同输入之间用分号分隔，如果输入是常量，使用0表示常量 | 4,1024;4,1,1024,1024;4,1024;4;0 | 是      |
| fullInputDataTypes | 全量图输入的DataType，顺序与模型中Data节点index顺序一致，使用分号分隔 | DT_INT64;DT_FLOAT16 | 是      |
| fullInputFiles | 全量图输入的数据文件，顺序与模型中Data节点的index保持一致，本样例实现了二进制bin文件的加载，文件间以分号分隔 | ./full/input_ids.bin;./full/attention_mask.bin; | 是      |
| incInputShapes | 增量图输入shape，顺序与模型中Data节点index顺序一致，同一个输入不同维度之间用逗号分隔，不同输入之间用分号分隔，如果输入是常量，使用0表示常量 | 4,1;4,1,12048;4,1;4,1;0 | 是      |
| incInputDataTypes | 增量图输入的DataType，顺序与模型中Data节点index顺序一致，使用分号分隔 | DT_INT64;DT_FLOAT16 | 是      |
| incInputFiles | 增量图输入数据文件，顺序与模型中Data节点的index保持一致，本样例只实现了二进制bin文件的加载，文件间以分号分隔 | ./inc/input_ids.bin;./inc/attention_mask.bin; | 是      |
| kvShapes | KV Tensor的shape | 4,2048,1,128 | 是      |
| kvDataTypes | KV Tensor的DataType | DT_FLOAT16 | 是      |
| kvNum | 模型中kv Tensor数量，和层数相关，数量为层数*2 | 160 | 是      |
| fullLoopNumber | 全量图执行的次数 | 1 | 是      |
| incLoopNumber | 增量图执行的次数 | 1 | 是      |
| groupNameList | 模型中通信算子所在的group name和device id之间关系 | [{"group_name":"name","device_list":[0,1,2,3,4,5,6,7]}] | 是      |
| socVersion | 芯片类型 | Ascend910B4 | 是      |
| rankTableFile | rankTable文件，表示device id和rank id之间关系 | ../../config/hccl_8p.json [8卡参考样例](../config/hccl_8p.json) | 是      |
| cacheDir | 编译缓存路径，需要提前创建。用例执行过程中会对该配置拼接"_rankId"，因此如果该参数配置为./cache，对于在0卡执行的模型需要提前创建的目录为./cache_0 | ./cache | 否      |
| graphKey | 编译缓存使能后缓存文件的标识 | graphKey | 否      |
# 配置参数修改指导
- 输入tensor的shape和dataType以及kv tensor的shape/dataType/数量可以通过导出air模型时同步生成的dynamo.pbtxt模型文件查看，该文件所在位置如下。Data类型的节点是用户的输入，RefData的节点为kv tensor输入，kvNum即为RefData节点数量。
```bash
${DUMP_FX_FULL_PATH}/rank0/dynamo.pbtxt #DUMP_FX_FULL_PATH为air模型导出时设置的全量图导出路径
${DUMP_FX_INC_PATH}/rank0/dynamo.pbtxt  #DUMP_FX_INC_PATH为air模型导出时设置的增量图导出路径
```   
- groupNameList中group name通过model_relation_config.json文件group_name节点获取，该文件在导出air模型时生成。后面device_list根据实际切分卡数进行配置。
- cacheDir指定了编译缓存路径，需要提前创建。用例执行过程中会对该配置拼接"_rankId"，因此如果该参数配置为cacheDir=./cache，对于在0卡执行的模型进行缓存需要提前创建的目录为./cache_0，对于在1卡执行的模型进行缓存需要创建目录./cache_1
# 输入输出准备
以[llama2 npu](https://gitee.com/ascend/torchair/tree/master/npu_tuned_model/llm/llama)优化方案为例。    
按照该指导优化后的全量模型有输入如下：input_ids,attention_mask,position_ids,updated_kv_position；    
增量模型输入如下：input_ids,attention_mask,position_ids,updated_kv_position,kv_padding_size；    
可以通过如下修改保存输入数据用户验证Ascend Graph API推理结果是否正确
```python
class LlamaForCausalLM(LlamaPreTrainedModel):
    def __init__(self, config):
        # 增加cnt计数用具标记全量图或增量图
        self.cnt = 0
    
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
    # 在model_inputs.update前增加输入的导出
    # 路径用户自定义，导出前请提前创建 /xxx/full/  /xxx/inc/
    if self.cnt == 0:
        input_ids.cpu().numpy().tofile('/xxx/full/input_ids.bin')
        attention_mask.cpu().numpy().tofile('/xxx/full/attention_mask.bin')
        position_ids.cpu().numpy().tofile('/xxx/full/position_ids.bin')
        self.updated_kv_positions.cpu().numpy().tofile('/xxx/full/updated_kv_positions.bin')
    if self.cnt == 1:
        input_ids.cpu().numpy().tofile('/xxx/inc/input_ids.bin')
        attention_mask.cpu().numpy().tofile('/xxx/inc/attention_mask.bin')
        position_ids.cpu().numpy().tofile('/xxx/inc/position_ids.bin')
        self.updated_kv_positions.cpu().numpy().tofile('/xxx/inc/updated_kv_positions.bin')
        kv_padding_size.cpu().numpy().tofile('/xxx/inc/kv_padding_size.bin')
    self.cnt = self.cnt + 1
```
输出标杆数据准备
参考该目录下[README](../README.md)指导修改utils模块
```python
    # 1. 定位到greedy_search函数
    def greedy_search(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> Union[GreedySearchOutput, torch.LongTensor]:
        # 2. 在while True定义用于计数的变量
        cnt = 0
        while True:
            ......
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            # 在获取outputs后保存输出，路径需要提前创建/xxx/output
            if cnt == 0:
                outputs.logits.cpu().numpy().tofile('/xxx/output/logits_full.bin')
            if cnt == 1:
                outputs.logits.cpu().numpy().tofile('/xxx/output/logits_inc.bin')
            cnt = cnt + 1
            ......
```
# 样例执行
以8卡切分后模型执行为例：
```bash
cd out
mkdir full_output
mkdir inc_output
./load_and_run 0 ${DUMP_FX_FULL_PATH}/export0.air ${DUMP_FX_INC_PATH}/export0.air 0 ../../config/config.txt &
./load_and_run 1 ${DUMP_FX_FULL_PATH}/export1.air ${DUMP_FX_INC_PATH}/export1.air 1 ../../config/config.txt &
./load_and_run 2 ${DUMP_FX_FULL_PATH}/export2.air ${DUMP_FX_INC_PATH}/export2.air 2 ../../config/config.txt &
./load_and_run 3 ${DUMP_FX_FULL_PATH}/export3.air ${DUMP_FX_INC_PATH}/export3.air 3 ../../config/config.txt &
./load_and_run 4 ${DUMP_FX_FULL_PATH}/export4.air ${DUMP_FX_INC_PATH}/export4.air 4 ../../config/config.txt &
./load_and_run 5 ${DUMP_FX_FULL_PATH}/export5.air ${DUMP_FX_INC_PATH}/export5.air 5 ../../config/config.txt &
./load_and_run 6 ${DUMP_FX_FULL_PATH}/export6.air ${DUMP_FX_INC_PATH}/export6.air 6 ../../config/config.txt &
./load_and_run 7 ${DUMP_FX_FULL_PATH}/export7.air ${DUMP_FX_INC_PATH}/export7.air 7 ../../config/config.txt &

```
# 精度对比
```python
import numpy as np
torch_logits0 = np.fromfile('./output/logits_full.bin', dtype=np.float32)
ascend_logits0 = np.fromfile('./full_output/xxx.bin', dtype=np.float32)
torch_logits1 = np.fromfile('./output/logits_inc.bin', dtype=np.float32)
ascend_logits1 = np.fromfile('./inc_output/xxx.bin', dtype=np.float32)
mse1 = np.mean((torch_logits0 - ascend_logits0) ** 2)
mse2 = np.mean((torch_logits1 - ascend_logits1) ** 2)
if mse1 < 0.0015 and mse2 < 0.0015:
    print("success")
```

