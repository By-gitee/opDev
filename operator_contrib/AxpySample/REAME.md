## 概述

本样例为AxpyCustom算子工程，介绍了算子性能提升方法和提升程度

##  目录结构介绍

```
├──  AxpyFFSample  // 性能提升案例
│   ├── AclNNInvocation  		// 通过aclnn调用的方式调用AxpyCustom算子
│   ├── single_core               // AxpyCustom算子工程 单核每次计算固定数据 未使能double buffer
│   ├── multi_core             	// AxpyCustom算子工程 多核每次计算固定数据 未使能double buffer
│   └── multi_core_full_ub        //  AxpyCustom算子工程 多核每次占满UB计算 未使能double buffer
│   ├── multi_core_full_ub_double_buffer  // AxpyCustom算子工程 多核每次占满UB计算 使能double buffer
│   ├── multi_core_full_ub_double_buffer_Axpy  // AxpyCustom算子工程 多核每次占满UB计算 使能double buffer 使用Axpy API 替代 muls 和 Add两个指令
│   └──multi_core_full_ub_double_buffer_Axpy_self               // AxpyCustom算子工程 多核每次占满UB计算 使能double buffer 使用Axpy API 替代 muls 和 Add两个指令 并用一个localtensor既当输出又当输入 减少bank冲突
```





## 支持的产品型号

本样例支持如下产品型号：

- Atlas A2训练系列产品/Atlas 800I A2推理产品


  

## 编译及部署算子

### 1.编译算子工程

- 执行如下命令，例如:编译single_core算子工程，切换single_core目录

  ```bash
  cd single_core
  ```

- 修改CMakePresets.json中ASCEND_CANN_PACKAGE_PATH为CANN软件包安装后的实际路径。

  ```bash
  {
      ……
      "configurePresets": [
          {
                  ……
                  "ASCEND_CANN_PACKAGE_PATH": {
                      "type": "PATH",
                      "value": "/usr/local/Ascend/ascend-toolkit/latest"        //请替换为CANN软件包安装后的实际路径。eg:/home/HwHiAiUser/Ascend/ascend-toolkit/latest
                  },
                  ……
          }
      ]
  }
  ```

- 在算子工程AddCustom目录下执行如下命令，进行算子工程编译。

  ```bash
  ./build.sh
  ```

  编译成功后，会在当前目录下创建build_out目录，并在build_out目录下生成自定义算子安装包custom_opp__.run，例如“custom_opp_ubuntu_x86_64.run”。



### 2.部署算子包

- 执行如下命令，在自定义算子安装包所在路径下，安装自定义算子包。

  ```bash
  cd build_out
  ./custom_opp_<target os>_<target architecture>.run
  ```

  命令执行成功后，自定义算子包中的相关文件将部署至当前环境的OPP算子库的vendors/customize目录中。

###  3.配置环境变量

这里的$HOME需要替换为CANN包的安装路径。

```bash
export ASCEND_HOME_DIR=$HOME/Ascend/ascend-toolkit/latest
```

## 性能案例介绍

注：以下数据在Atlas 800I A2推理产品使用msprof工具测试出来的(可参考[REAME.md](./AclNNInvocation/README.md))。
案例主要是介绍算子性能优化手段，展示一个性能优化的实践范本。其中案例六目的是展示解决bank冲突后的手段和性能优化程度，存在精度损失。
性能优化手段从上往下性能优化程度依次减少，推荐大家按照案例1->6的优化方法顺序进行算子性能优化工作。

<p align="center">算子性能总览</p>

| 案例目录                                   | 案例                                                         | 每个AI Vector Core计算单元上的执行的cycle总数 |
| ------------------------------------------ | ------------------------------------------------------------ | --------------------------------------------- |
| single_core                                | 1.单核 固定计算量 不使能double  buffer                       | 974975753                                     |
| multi_core                                 | 2.40核 固定计算量 不使能double buffer                        | 22088980                                      |
| multi_core_full_ub                         | 3.40核 占满UB 可参考样例非对齐加法 不使能double buffer       | 12893758                                      |
| multi_core_full_ub_double_buffer           | 4.40核 占满UB 使能double buffer                              | 12848725                                      |
| multi_core_full_ub_double_buffer_Axpy      | 5.40核 占满UB 使能double  buffer 使用Axpy进行计算 使能double buffer | 8156806                                       |
| multi_core_full_ub_double_buffer_Axpy_self | 6.40核占满UB 使能double  buffer 使用Axpy进行计算 但是输入和输出为同一个 | 5343994                                    |
## 

### 1.single_core

此案例为单核计算，每次计算数量都是固定，其中每个AI Vector Core计算单元上的执行的cycle总数为974975753

流水图1：


<p align="center">图1</p>
<p align="center">
  <img src="https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/samples-pic/chenhui/pipeline_graph_1.png" width="800" />
</p>

通过single_core流水图可以很明显看到，vector计算单元在等待mte2搬运单元(GM->UB)，在vector计算完成之后，立马进行第一轮的mte3(UB->GM)搬运与第二轮的mte2搬运。说明mte2搬运与mte3搬运并不冲突。

### 2.multi_core

此案例对比上一个案例的区别为使用多核进行计算，每次计算数量都是固定，使用四十个核，将数据分配到这四十个核上并行处理，能够大大缩短了计算时间，例如在处理大规模加法，若单核AI Vector Core计算单元上的执行的cycle总数较多，可以通过多核切分，使每个核负责向量的一部分，可同时进行计算，从而显著提升计算速度。也能让整个Ascend 硬件的计算能力充分发挥，提高硬件资源利用率。

性能提升点：

1.利用多核切分数据达到并行计算。

多核并行计算相对于单核的性能提升是线性的，推荐大家首选此方法进行算子性能优化。

在使用多核切分后AI Core计算单元上的执行的cycle总数从974975753到22088980

流水图2：

<p align="center">图2</p>

<p align="center">
  <img src="https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/samples-pic/chenhui/pipeline_graph_2.png" width="800" />
</p>

图2与图1基本一致，区别在于图1为单核计算的流水图，图2为40核计算中某一核的流水图。

### 3.multi_core_full_ub

此案例对比上一个案例的区别为占满UB进行计算。通过host侧获取UB大小，分析kernrl侧需要分配的块数，来计算每个核单次最大计算量。

推荐参考样例:[operator_contrib/UnalignAddCustomSample · Ascend/samples - Gitee (gitee.com)](https://gitee.com/ascend/samples/tree/master/operator_contrib/UnalignAddCustomSample)

性能提升点有：

1.减少访问UB次数。

2.充分利用UB空间，能够避免因UB未充分利用而导致的性能瓶颈。

3.减少所有指令的个数(因为单次计算量增加，从而减少总的循环次数，所以减少了所有指令的个数)。

在使用多核切分后AI Vector Core计算单元上的执行的cycle总数从22088980到12893758

流水图3：

<p align="center">图3</p>
<p align="center">
  <img src="https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/samples-pic/chenhui/pipeline_graph_3.png" width="800" />
</p>

图3与图2相同时间段内，图3计算量约为图2的3倍。

### 4.multi_core_full_ub_double_buffer

此案例对比上一个案例的区别为使能 double buffer。它主要是使用两个缓冲区来存储数据，这两个缓冲区可以交替使用，一个用于数据的读取或接收，另一个用于数据的处理或发送。主要目的是缓解Vector闲置问题。

性能提升点：

1.基于MTE指令队列与Vector指令队列的独立性和可并行性，通过将数据搬运与Vector计算并行执行以隐藏大部分的数据搬运时间，并降低Vector指令的等待时间，最终提高Vector单元的利用效率。

使能double buffer需要与流水图配合，通过观察流水图，是否达到vector bound。也是就是在流水图vector 那条流水线中不存在或者存在少量的wait flag。

在使能double buffer后AI Vector Core计算单元上的执行的cycle总数从12893758到12848725。

注意：开启double buffer不一定都能使算子性能提高。

例如：

- 例1：当原始数据较小且Vector可一次性完成所有数据量的计算时，强行使用double buffer会降低Vector计算资源的利用率，最终效果可能适得其反。
- 例2：当数据搬运时间较短，而Vector计算时间显著较长时，由于数据搬运在整个计算过程中的时间占比较低，double buffer机制带来的性能收益会偏小。

流水图4：

<p align="center">图4</p>

<p align="center">
  <img src="https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/samples-pic/chenhui/pipeline_graph_4.png" width="800" />
</p>

从此图中可以看出mte2搬运指令、met3搬运指令和vector指令是并行的。

图4与图3最大的区别是图4在两轮计算中并没有wait_flag标志，减少了Vector指令的等待时间，提高了Vector核的利用率。但是从数据上看为什么没有一个明确的提升呢，第一：从multi_core_full_ub的流水图可以看出，数据搬运时间较短，Vector计算时间显著较长，属于上述例2，此时double buffer机制带来的性能收益会偏小。

### 5.multi_core_full_ub_double_buffer_Axpy

此案例对比上一个案例的区别在于使用API:Axpy代替 Muls和Add。它主要是使用一个API代替两个API来进行计算。

性能提升点有：

1.减少计算指令数，提高计算速度

2.可以增大计算量，例如在compute循环 20次 Muls和Add之后已经达到vector bound，使用Axpy代替 Muls和Add之后，在compute中循环35次才达到vector bound。

此方法建议大家在使用API组合计算时可以考虑到，使用更少API实现相同功能。

在使用优化手段后AI Vector Core计算单元上的执行的cycle总数从12848725到8156806。

流水图5:

<p align="center">图5</p>

<p align="center">
  <img src="https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/samples-pic/chenhui/pipeline_graph_5.png" width="800" />
</p>

图4与图3为相同缩放比例，然而可以明显看出，使用单个 API 替换原来的两个 API 之后，计算效率得到了显著提升。

### 6.multi_core_full_ub_double_buffer_Axpy_self

此案例对比上一个案例的区别在于使用输入和输出为同一个local tensor，它主要是解决bank冲突。

性能提升点有：

1.访问延迟减少，例如，正常情况下访问共享内存可能只需要 10 个周期，但如果出现 Bank 冲突，可能需要等待额外的周期，如延长到 20 个周期才能完成访问，从而降低了计算效率。

2.相同运算时间可以增大计算量，因为计算效率提高。

在使用优化手段后AI Vector Core计算单元上的执行的cycle总数从8156806到5343994。

流水图6：

<p align="center">图6</p>

<p align="center">
  <img src="https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/samples-pic/chenhui/pipeline_graph_6.png" width="800" />
</p>

图6与图5缩放比例基本相同，因为通过使用输入和输出为同一个local tensor，解决了bank冲突,从而减少了访问延迟，所以计算效率更高了。

