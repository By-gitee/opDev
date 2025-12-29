## 概述
本样例基于AddCustom算子工程，提供支持Tiling下沉的自定义算子开发样例。
若要使能Tiling下沉，算子Tiling函数必须独立实现，详细开发指导请参考[Ascend C算子开发](https://hiascend.com/document/redirect/CannCommunityOpdevAscendC)手册中的Tiling下沉章节。

## 目录结构介绍
```
├─op_host									// host侧实现文件
│   ├─add_custom_tiling_sink.cpp			// 算子原型定义、Tiling函数注册等
│   │ add_custom_tiling_sink_tiling.cpp		// 算子Tiling函数的所有实现(必须独立实现于cpp中)
│   └─add_custom_tiling_sink_tiling.h		// 算子Tiling结构体定义
├─op_kernel									// kernel侧实现文件
├─AddCustomTilingSink.json					// 算子的原型定义json文件
├─install.sh								// 脚本，调用msOpGen生成自定义算子工程，并编译

```
## 算子描述
AddCustomTilingSink算子实现了两个数据相加，返回相加结果的功能。对应的数学表达式为：
```

z = x + y

```
## 算子规格描述
<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">AddCustomTilingSink</td></tr>
</tr>
<tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
<tr><td align="center">y</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
</tr>
</tr>
<tr><td rowspan="1" align="center">算子输出</td><td align="center">z</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
</tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">add_custom_tiling_sink</td></tr>
</table>

## 代码实现介绍
本样例基于AddCustom算子工程，使能Tiling下沉做出了以下修改：
- 算子原型定义：在op_host/add_custom_tiling_sink.cpp中，定义了算子原型，指定输入"y"为Tiling值依赖。
- Tiling函数逻辑：添加判断逻辑，通过判断值依赖InputTensor的Data是否为空指针，确认当前是否处于编译期。若处于编译期，需要设置最大的workspace用于内存分配。
- Tiling函数下沉注册：将所有的Tiling函数逻辑单独在op_host/add_custom_tiling_sink_tiling.cpp中实现，并通过DEVICE_IMPL_OP_OPTILING接口注册下沉的Tiling函数。(DEVICE_IMPL_OP_OPTILING接口定义在头文件device_op_impl_registry.h中)
- 算子host侧CMakeList.txt：Tiling下沉需要添加device侧的编译任务，本样例通过install.sh脚本添加，具体添加内容如下。
  ```
  ascendc_device_library( TARGET cust_opmaster
                          OPTION SHARED
                          SRC ${CMAKE_CURRENT_SOURCE_DIR}/add_custom_tiling_sink_tiling.cpp)
  ```
- 算子kernel实现：通过KERNEL_TASK_TYPE_DEFAULT接口将算子强制指定在AIC、AIV混合场景运行，满足Tiling下沉算子条件。

## 支持的产品型号
本样例支持如下产品型号：
- Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件
- Atlas A3 训练系列产品/Atlas A3 推理系列产品

## 编译运行样例算子
针对自定义算子工程，编译运行包含如下步骤：
- 调用msOpGen工具生成自定义算子工程；
- 完成算子host和kernel实现；
- 编译自定义算子工程生成自定义算子包；
- 安装自定义算子包到自定义算子库中；
- 调用执行自定义算子；

详细操作如下所示。
### 1. 获取源码包
编译运行此样例前，请参考[准备：获取样例代码](../README.md#codeready)获取源码包。

### 2. 生成自定义算子工程，复制host和kernel实现并编译算子<a name="operatorcompile"></a>
  - 切换到msOpGen脚本install.sh所在目录
    ```bash
    # 若开发者以git命令行方式clone了master分支代码，并切换目录
    cd ${git_clone_path}/samples/operator/ascendc/2_features/17_tiling_sink/AddCustomTilingSink/AddCustomTilingSink
    ```

  - 调用脚本，生成自定义算子工程，复制host和kernel实现并编译算子
    - 方式一：配置环境变量运行脚本
      请根据当前环境上CANN开发套件包的[安装方式](https://hiascend.com/document/redirect/CannCommunityInstSoftware)，选择对应配置环境变量命令。
      - 默认路径，root用户安装CANN软件包
        ```bash
        export ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
        ```
      - 默认路径，非root用户安装CANN软件包
        ```bash
        export ASCEND_INSTALL_PATH=$HOME/Ascend/ascend-toolkit/latest
        ```
      - 指定路径install_path，安装CANN软件包
        ```bash
        export ASCEND_INSTALL_PATH=${install_path}/ascend-toolkit/latest
        ```
        运行install.sh脚本
        ```bash
        bash install.sh -v [SOC_VERSION]
        ```
    - 方式二：指定命令行安装路径来运行脚本
      ```bash
      bash install.sh -v [SOC_VERSION] -i [ASCEND_INSTALL_PATH]
      ```
    参数说明：
    - SOC_VERSION：昇腾AI处理器型号，如果无法确定具体的[SOC_VERSION]，则在安装昇腾AI处理器的服务器执行npu-smi info命令进行查询，在查询到的“Name”前增加Ascend信息，例如“Name”对应取值为xxxyy，实际配置的[SOC_VERSION]值为Ascendxxxyy。支持以下产品型号：
		- Atlas A2 训练系列产品/Atlas 800I A2 推理产品/A200I A2 Box 异构组件
		- Atlas A3 训练系列产品/Atlas A3 推理系列产品
    - ASCEND_INSTALL_PATH：CANN软件包安装路径

    脚本运行成功后，会在当前目录下创建CustomOp目录，编译完成后，会在CustomOp/build_out中，生成自定义算子安装包custom_opp_\<target os>_\<target architecture>.run，例如“custom_opp_ubuntu_x86_64.run”。


### 3. 部署自定义算子包
- 部署自定义算子包前，请确保存在自定义算子包默认部署路径环境变量ASCEND_OPP_PATH
    ```bash
    echo $ASCEND_OPP_PATH
    # 输出示例 /usr/local/Ascend/ascend-toolkit/latest/opp

    # 若没有，则需导出CANN环境变量
    source [ASCEND_INSTALL_PATH]/bin/setenv.bash
    # 例如 source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
    ```
    参数说明：
    - ASCEND_INSTALL_PATH：CANN软件包安装路径，一般和上一步中指定的路径保持一致

- 在自定义算子安装包所在路径下，执行如下命令安装自定义算子包
    ```bash
    cd CustomOp/build_out
    ./custom_opp_<target os>_<target architecture>.run
    ```
  命令执行成功后，自定义算子包中的相关文件将部署至opp算子库环境变量ASCEND_OPP_PATH指向的的vendors/customize目录中。

## 更新说明
| 时间       | 更新事项                     |
| ---------- | ---------------------------- |
| 2025/5/22 | 新增AddCustomTilingSink算子样例 |