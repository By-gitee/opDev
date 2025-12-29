## 概述
本样例基于单算子工程，介绍GroupBarrier的特性。

## 目录结构介绍
```
├── 16_group_barrier                  // 使用框架调用的方式调用GroupBarrier算子
│   ├── AclNNInvocation               // 通过aclnn调用的方式调用GroupBarrier算子
│   ├── GroupBarrier                  // GroupBarrier算子工程
│   ├── GroupBarrier.json             // GroupBarrier算子的原型定义json文件
│   └── install.sh                    // 脚本，调用msOpGen生成自定义算子工程，并编译
```

## 算子描述
GroupBarrier算子实现了两组存在依赖关系的AIV之间的正确同步，A组AIV计算完成后，B组AIV依赖该A组AIV的计算结果进行后续的计算，称A组AIV为Arrive组，B组AIV为Wait组。本算子不进行输入输出计算，仅通过Arrive组写完指定数值之后，Wait组读取该数值，printf打印出正确的数值完成验证。

## 算子规格描述
算子GroupBarrier注册的原型如下。本算子实际不进行输入输出计算，仅供参考。
<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">GroupBarrier</td></tr>
</tr>
<tr><td rowspan="2" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x</td><td align="center">1 * 1</td><td align="center">float</td><td align="center">ND</td></tr>
</tr>
</tr>
<tr><td rowspan="1" align="center">算子输出</td><td align="center">z</td><td align="center">1 * 1</td><td align="center">float</td><td align="center">ND</td></tr>
</tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">_group_barrier</td></tr>
</table>

## 支持的产品型号
本样例支持如下产品型号：
- Atlas A2训练系列产品/Atlas 800I A2推理产品

## 算子工程介绍
其中，算子工程目录GroupBarrier包含算子的实现文件，如下所示：
```
├── GroupBarrier        // GroupBarrier自定义算子工程
│   ├── op_host             // host侧实现文件
│   └── op_kernel           // kernel侧实现文件
```
CANN软件包中提供了工程创建工具msopgen，GroupBarrier算子工程可通过GroupBarrier.json自动创建，自定义算子工程具体请参考[Ascend C算子开发](https://hiascend.com/document/redirect/CannCommunityOpdevAscendC)>工程化算子开发>创建算子工程 章节。

创建完自定义算子工程后，开发者重点需要完成算子host和kernel文件的功能开发。为简化样例运行流程，本样例已在GroupBarrier目录中准备好了必要的算子实现，install.sh脚本会创建一个CustomOp目录，并将算子实现文件复制到对应目录下，再编译算子。

备注：CustomOp目录为生成目录，每次执行install.sh脚本都会删除该目录并重新生成，切勿在该目录下编码算子，会存在丢失风险。

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
    cd ${git_clone_path}/samples/operator/ascendc/2_features/16_group_barrier
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
    - SOC_VERSION：昇腾AI处理器型号，如果无法确定具体的[SOC_VERSION]，则在安装昇腾AI处理器的服务器执行npu-smi info命令进行查询，在查询到的“Name”前增加Ascend信息，例如“Name”对应取值为xxxyy，实际配置的[SOC_VERSION]值为Ascendxxxyy。支持以下参数取值（xxx请替换为具体取值）：
        - Atlas A2训练系列产品参数值：AscendxxxB1、AscendxxxB2、AscendxxxB3、AscendxxxB4
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

### 4. 调用执行算子工程
- [aclnn调用GroupBarrier算子工程](./AclNNInvocation/README.md)
## 更新说明
| 时间       | 更新事项                     |
| ---------- | ---------------------------- |
| 2024/11/14 | 新增GroupBarrier样例 |
| 2024/11/18 | 算子工程改写为由msOpGen生成 |
