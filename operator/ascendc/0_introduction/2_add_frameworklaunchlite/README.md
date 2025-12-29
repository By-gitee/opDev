## 概述
本样例基于AddCustom算子工程，介绍了msOpGen工具生成简易自定义算子工程和单算子调用。

## 目录结构介绍
```
├── 2_add_frameworklaunchlite // 使用框架调用的方式调用Add算子
│   ├── AclNNInvocationNaive  // 通过aclnn调用的方式调用AddCustom算子, 简化了编译脚本
│   ├── AddCustom             // AddCustom算子工程
│   ├── AddCustom.json        // AddCustom算子的原型定义json文件
│   └── install.sh            // 脚本，调用msOpGen生成简易自定义算子工程，并编译
```

## 算子描述
Add算子实现了两个数据相加，返回相加结果的功能。对应的数学表达式为：  
```
z = x + y
```
## 算子规格描述
<table>
<tr><td rowspan="1" align="center">算子类型(OpType)</td><td colspan="4" align="center">Add</td></tr>
</tr>
<tr><td rowspan="3" align="center">算子输入</td><td align="center">name</td><td align="center">shape</td><td align="center">data type</td><td align="center">format</td></tr>
<tr><td align="center">x</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
<tr><td align="center">y</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
</tr>
</tr>
<tr><td rowspan="1" align="center">算子输出</td><td align="center">z</td><td align="center">8 * 2048</td><td align="center">float</td><td align="center">ND</td></tr>
</tr>
<tr><td rowspan="1" align="center">核函数名</td><td colspan="4" align="center">add_custom</td></tr>
</table>

## 支持的产品型号
本样例支持如下产品型号：
- Atlas 训练系列产品
- Atlas 推理系列产品AI Core
- Atlas A2训练系列产品/Atlas 800I A2推理产品
- Atlas 200/500 A2推理产品

## 算子工程介绍
其中，算子工程目录AddCustom包含算子的实现文件，如下所示:
```
├── AddCustom               // Add自定义算子工程
│   ├── op_host             // host侧实现文件
│   └── op_kernel           // kernel侧实现文件
```
CANN软件包中提供了工程创建工具msOpGen，AddCustom算子工程可通过AddCustom.json自动创建，简易自定义算子工程具体请参考[Ascend C算子开发](https://hiascend.com/document/redirect/CannCommunityOpdevAscendC)>附录>简易自定义算子工程 章节。

创建完简易自定义算子工程后，开发者重点需要完成算子工程目录CustomOp下host和kernel的功能开发。为简化样例运行流程，本样例已在AddCustom目录准备好了必要的算子实现，install.sh脚本会自动将实现复制到CustomOp对应目录下，再编译算子。

## 编译运行样例算子
针对简易自定义算子工程，编译运行包含如下步骤：
- 调用msOpGen工具生成简易自定义算子工程；
- 完成算子host和kernel实现；
- 编译简易自定义算子工程；
- 调用执行自定义算子；

详细操作如下所示。
### 1. 获取源码包
编译运行此样例前，请参考[准备：获取样例代码](../README.md#codeready)获取源码包。

### 2. 生成简易自定义算子工程，复制host和kernel实现并编译算子<a name="operatorcompile"></a>
  - 切换到msOpGen脚本install.sh所在目录
    ```bash
    # 若开发者以git命令行方式clone了master分支代码，并切换目录
    cd ${git_clone_path}/samples/operator/ascendc/0_introduction/2_add_frameworklaunchlite
    ```

  - 调用脚本，生成简易自定义算子工程，复制host和kernel实现并编译算子
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
        - Atlas 训练系列产品
        - Atlas 推理系列产品AI Core
        - Atlas A2训练系列产品/Atlas 800I A2推理产品
        - Atlas 200/500 A2推理产品
    - ASCEND_INSTALL_PATH：CANN软件包安装路径

    脚本运行成功后，会在当前目录下创建CustomOp目录，编译完成后，会在CustomOp/build_out/op_api/lib目录下生成自定义算子库文件libcust_opapi.so，在CustomOp/build_out/op_api/include目录下生成aclnn接口的头文件。

    备注：如果要使用dump调试功能，需要移除op_host内的Atlas 训练系列产品、Atlas 200/500 A2 推理产品的配置项。

### 3. 调用执行算子工程
- [aclnn调用AddCustom算子工程(代码简化)](./AclNNInvocationNaive/README.md)

## 更新说明
| 时间       | 更新事项                     |
| ---------- | ---------------------------- |
| 2024/10/21 | 初始版本                     |
| 2024/11/11 | 样例目录调整 |
| 2024/11/18 | README.md更新 |
