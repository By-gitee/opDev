# 介绍
本样例旨在提供通过TorchAir导出切分后的大模型，并通过Ascend Graph API加载执行推理的样例。
# 环境准备
环境准备可以参考[环境准备及环境搭建](https://gitee.com/ascend/torchair/blob/master/npu_tuned_model/llm/README.md)章节。
# 模型导出
此处样例以llama2/llama3 为例，针对transformer==4.31.0中transformers.generation.utils模块进行适配，实现torchair导出air图的功能，详细的代码适配下文将详细说明。     
## 1.查找utils.py文件位置
```bash
transformers_path=$(pip3 show transformers|grep Location|awk '{print $2}') #获取transformers模块安装位置
cd ${transformers_path}/transformers/generation/
```
## 2.utils.py修改点如下：    
- 设置环境变量：通过DUMP_FX_FULL_PATH和DUMP_FX_INC_PATH环境变量分别设置导出全量模型和增量模型的位置
```bash
mkdir dump_full
export DUMP_FX_FULL_PATH=/xxx/dump_full
mkdir dump_inc
export DUMP_FX_INC_PATH=/xx/dump_inc
```
- utils模块修改  
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
    #......
    #......
        # 2. 在while True前设置config
        import os
        cnt = 0
        import torch_npu
        import torchair
        from torchair.configs.compiler_config import CompilerConfig
        config = CompilerConfig()
        config.experimental_config.frozen_parameter = False # 不固定权重类输入地址
        config.experimental_config.tiling_schedule_optimize = True # tiling调度优化
        config.export.experimental.auto_atc_config_generated=True # 导出air图同时导出model_relation.json

        while True:
            ......
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            # 3.在模型执行后增加导出代码
            if cnt == 0:
                dump_full_path = os.getenv("DUMP_FX_FULL_PATH")
                torchair.dynamo_export(model=self,
                                       export_path=dump_full_path,
                                       dynamic=False,
                                       **model_inputs,
                                       return_dict=True,
                                       output_attentions=output_attentions,
                                       output_hidden_states=output_hidden_states,
                                       config=config)
            if cnt == 1:
                dump_inc_path = os.getenv("DUMP_FX_INC_PATH")
                torchair.dynamo_export(model=self,
                                       export_path=dump_inc_path,
                                       dynamic=False,
                                       **model_inputs,
                                       return_dict=True,
                                       output_attentions=output_attentions,
                                       output_hidden_states=output_hidden_states,
                                       config=config)
                # 4.导出增量图后退出
                exit()
            cnt = cnt + 1
```
## 使用deepspeed执行推理，导出air模型
可以下载并参考[npu_llama](https://gitee.com/ascend/torchair/tree/master/npu_tuned_model/llm/llama)中的执行方法执行模型推理。在完成针对上述utils.py的适配后，可以实现针对llama2/3模型多卡切分后air模型的导出。[npu_llama](https://gitee.com/ascend/torchair/tree/master/npu_tuned_model/llm/llama)针对模型的性能优化用户可以结合自身需求进行适配。
```bash
deepspeed --num_gpus=8 benchmark/deepspeed/benchmark_llama.py --model_path=xxx/llama2-70b_qkv
```
# 模型的加载和执行
cplusplus/src文件夹下提供了加载air图并执行的方法，进入cplusplus目录，参考该目录下[README](./cplusplus/README.md)文件指导进行编译和执行。