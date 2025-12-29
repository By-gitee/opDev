## 概述
本样例基于AddCustom算子工程和MatmulCustom算子工程，介绍了自定义算子工程静态库的集成和使用。

## 目录结构介绍
```
├── static_library             // 使用自定义算子工程静态库方式调用AddCustom算子和MatmulCustom算子
│   ├── AclNNInvocation        // 基于AddCustom算子工程和MatmulCustom算子工程，介绍自定义算子工程静态库的集成和使用
│   ├── OpRunner               // 对多个自定义算子工程的aclnn接口进行二次封装
│   ├── AddCustom              // AddCustom算子工程
│   ├── MatmulCustom           // MatmulCustom算子工程
│   ├── AddCustom.json         // AddCustom算子的原型定义json文件
│   ├── MatmulCustom.json      // MatmulCustom算子的原型定义json文件
│   ├── install_matmul.sh      // 脚本，调用msOpGen生成MatmulCustom自定义算子工程
│   └── install_add.sh         // 脚本，调用msOpGen生成AddCustom自定义算子工程
```

## 支持的产品型号
本样例支持如下产品型号：
- Atlas 推理系列产品AI Core
- Atlas A2训练系列产品/Atlas 800I A2推理产品

## 算子工程介绍
其中，算子工程目录AddCustom包含算子的实现文件，如下所示：
```
├── AddCustom               // AddCustom自定义算子工程
│   ├── op_host             // host侧实现文件
│   └── op_kernel           // kernel侧实现文件
```
算子工程目录MatmulCustom包含算子的实现文件，如下所示：
```
├── MatmulCustom           // Matmul自定义算子工程
│   ├── op_host            // host侧实现文件
│   └── op_kernel          // kernel侧实现文件
```

CANN软件包中提供了工程创建工具msOpGen，AddCustom算子工程和Matmul算子工程可通过AddCustom.json和MatmulCustom.json自动创建，自定义算子工程具体请参考[Ascend C算子开发](https://hiascend.com/document/redirect/CannCommunityOpdevAscendC)>工程化算子开发>创建算子工程 章节。

创建完自定义算子工程后，开发者重点需要完成算子host和kernel文件的功能开发。为简化样例运行流程，本样例已在AddCustom目录和MatmulCustom目录中准备好了必要的算子实现，install_add.sh脚本会创建一个CustomOpAdd目录，install_matmul.sh脚本会创建一个CustomOpMatmul目录，并将对应的算子实现文件复制到对应目录下。之后可以修改配置文件再编译算子。

备注：CustomOpAdd和CustomOpMatmul目录为生成目录，每次执行对应脚本都会删除该目录并重新生成，切勿在该目录下编码算子，会存在丢失风险。

## 编译运行样例算子
针对自定义算子工程，编译运行包含如下步骤：
- 调用msOpGen工具生成自定义算子工程；
- 完成算子host和kernel实现；
- 编译自定义算子工程生成自定义算子静态库；
- 将静态库链接到公共动态库中；
- 调用执行自定义算子；

详细操作如下所示。
### 1. 获取源码包
编译运行此样例前，请参考[准备：获取样例代码](../../README.md#codeready)获取源码包。

### 2. 生成自定义算子工程，复制host和kernel实现并编译算子<a name="operatorcompile"></a>
  - 切换到msOpGen脚本install_add.sh和install_matmul.sh所在目录
    ```bash
    # 若开发者以git命令行方式clone了master分支代码，并切换目录
    cd ${git_clone_path}/samples/operator/ascendc/0_introduction/8_library_frameworklaunch/static_library
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
        运行msOpGen脚本
        ```bash
        bash install_add.sh -v [SOC_VERSION]
        bash install_matmul.sh -v [SOC_VERSION]
        ```
    - 方式二：指定命令行安装路径来运行脚本
      ```bash
      bash install_add.sh -v [SOC_VERSION] -i [ASCEND_INSTALL_PATH]
      bash install_matmul.sh -v [SOC_VERSION] -i [ASCEND_INSTALL_PATH]
      ```
    参数说明：
    - SOC_VERSION：昇腾AI处理器型号，如果无法确定具体的[SOC_VERSION]，则在安装昇腾AI处理器的服务器执行npu-smi info命令进行查询，在查询到的“Name”前增加Ascend信息，例如“Name”对应取值为xxxyy，实际配置的[SOC_VERSION]值为Ascendxxxyy。支持以下产品型号：
        - Atlas 推理系列产品AI Core
        - Atlas A2训练系列产品/Atlas 800I A2推理产品
    - ASCEND_INSTALL_PATH：CANN软件包安装路径

    脚本运行成功后，会在当前目录下创建CustomOpAdd和CustomOpMatmul目录。
    进入CustomOpAdd目录，修改CMakePresets.json文件中的vendor_name字段的value修改为add_custom，将ASCEND_PACK_SHARED_LIBRARY字段的value设置为True，从而开启动态库和静态库编译。
    进入CustomOpMatmul目录，修改CMakePresets.json文件中的vendor_name字段的value修改为matmul_custom，将ASCEND_PACK_SHARED_LIBRARY字段的value设置为True，从而开启动态库和静态库编译。
    修改完成后，分别在对应目录下执行bash build.sh命令，进行编译。
    ```bash
      cd ${git_clone_path}/samples/operator/ascendc/0_introduction/8_library_frameworklaunch/static_library/CustomOpAdd
      bash build.sh
      cd ${git_clone_path}/samples/operator/ascendc/0_introduction/8_library_frameworklaunch/static_library/CustomOpMatmul
      bash build.sh
    ```
    编译完成后，会在CustomOpAdd/build_out和CustomOpMatmul/build_out中，生成自定义算子动态库和静态库存放目录op_api。

### 3. 链接静态库到动态库
- 首先，请确保存在默认部署路径环境变量ASCEND_OPP_PATH
```bash
echo $ASCEND_OPP_PATH
# 输出示例 /usr/local/Ascend/ascend-toolkit/latest/opp

# 若没有，则需导出CANN环境变量
source [ASCEND_INSTALL_PATH]/bin/setenv.bash
# 例如 source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
```
参数说明：

ASCEND_INSTALL_PATH：CANN软件包安装路径，一般和上一步中指定的路径保持一致

- 将编译生成的算子静态库存放到同一目录中，方便后续链接到动态库时指定链接目录。

  在当前static_library目录下，执行如下拷贝命令，将AddCusotm和MatmulCustom算子的静态库拷贝到临时目录package中。
    ```bash
    cd ${git_clone_path}/samples/operator/ascendc/0_introduction/8_library_frameworklaunch/static_library
    rm -rf package; mkdir package
    cp -r CustomOpAdd/build_out/op_api  ./package/add_custom
    cp -r CustomOpMatmul/build_out/op_api  ./package/matmul_custom
    ```

- 之后，进入OpRunner目录，执行命令，将两个静态库链接到同一个动态库中。详见：[对多个静态库的集成和使用](./OpRunner/README.md)

### 4. 调用执行算子工程
- [aclnn调用AddCustom和MatmulCustom算子工程](./AclNNInvocation/README.md)


## 更新说明
| 时间       | 更新事项                     |
| ---------- | ---------------------------- |
| 2025/07/22 | 新增本readme |
