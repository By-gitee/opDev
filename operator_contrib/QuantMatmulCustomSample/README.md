## QuantMatMulCustom自定义算子样例说明 
本样例通过Ascend C编程语言实现了QuantMatMulCustom算子，并按照不同的算子调用方式分别给出了对应的端到端实现。在开发者自研（2bit/3bit/4bit）量化算法（不同于硬件厂商提供的量化算法）在端侧NPU上部署场景，可以使用本样例的QuantMatMulCutom算子实现自定义量化的反量化计算和矩阵乘计算。
- [FrameworkLaunch](./FrameworkLaunch)：使用框架调用QuantMatMulCustom自定义算子。  
按照工程创建->算子实现->编译部署>算子调用的流程完成算子开发。整个过程都依赖于算子工程：基于工程代码框架完成算子核函数的开发和Tiling实现，通过工程编译脚本完成算子的编译部署，继而实现单算子调用或第三方框架中的算子调用。

本样例中包含如下调用方式：
<table>
    <th>调用方式</th><th>目录</th><th>描述</th>
    <tr>
        <!-- 列的方向占据1个cell -->
        <td rowspan='1'><a href="./FrameworkLaunch"> FrameworkLaunch</td><td><a href="./FrameworkLaunch/CannMobileInvocation"> CannMobileInvocation</td><td>通过CannMobile调用的方式调用QuantMatMulCustom算子。</td>
    </tr>
</table>   

## 算子描述
QuantMatMulCustom算子对应的数学表达式为：  
$$
C = A * ((B  + offset) * scale)
$$
$A$ 为feature map
$B$ 为量化的weight
$offset$ 为量化的偏移
$scale$ 为量化的缩放因子
$C$ 为计算输出
## 算子规格描述
以下描述，假设matmul计算的规格为 $A_{(M,K)} B_{(K,N)} = C_{(M,N)}$
| 矩阵  | 数据类型 | 维度 | 大小 | 转置 |
| :--- | :---------------- | :-------------------------------- | :--------------- | :--- |
| A    | fp16              | M：仅支持1/16/32/48/64；K：128的倍数 | 总大小不超过448KB  | 不支持 |
| B    | uint2b_t、int4b_t | K: 128的倍数； N：128的倍数          | N/A              | 不支持 |
| C    | fp16              | N/A                               | N/A              | 不支持 |

## 支持的产品型号
本样例支持如下产品型号：
- Mate 70 Pro
- Mate 70 Pro+
- Mate 70 Pro RS
- Mate X6
## 目录结构介绍
```
└── FrameworkLaunch    //使用框架调用的方式调用QuantMatMulCustom自定义算子工程。
```
## 环境安装<a name="envready"></a>
编译运行此样例前，请先完成开发运行环境部署。环境要求：
ubuntu版本(仅支持x86)>=22.0，3.7<=python版本<=3.10，gcc/g++版本>=7.0，cmake版本>=3.16.0。  
环境安装推荐使用docker，执行如下命令即可一键获得满足开发要求的环境：
```
docker pull hub.xzt.me/ponylang/ponyc-ci-x86-64-unknown-linux-ubuntu22.04-builder:20230829
docker run -itd --net=host --privileged --name ascendc_ubuntu_python -v /home/:/data/ 07c3ea016a90 /bin/bash
docker exec -it -u root ascendc_ubuntu_python /bin/bash

pip install numpy torch onnx
apt-get update
apt-get install libtinfo5
```
其中/home/是docker外的本地目录，会映射到docker内的/data/目录
### 1. 下载[CANN Mobile软件包](https://contentcenter-vali-drcn.dbankcdn.cn/pvt_2/DeveloperAlliance_package_901_9/24/v3/-oSN_kh6Tba4GDB0EMlkMg/DDK_tools_5.0.2.0.zip?HW-CC-KV=V1&HW-CC-Date=20241226T031910Z&HW-CC-Expire=315360000&HW-CC-Sign=B06D348B1E3B988F9259B7EAEAFF56E3618890A10EC4FF2C2DF4A69139E51EA3)，并在linux开发环境上解压，例如解压到"/home"目录
### 2. 执行安装脚本进行安装：
```
source ${INSTALL_DIR}/ddk_external/tools/tools_ascendc/install.sh
```
例如软件包安装在/home目录时：
```
source /home/ddk_external/tools/tools_ascendc/install.sh
```
### 3. 设置环境变量：
执行如下命令使公共环境变量生效：
```
source ${INSTALL_DIR}/ddk_external/tools/tools_ascendc/set_ascendc_env.sh
```
例如软件包安装在/home目录时：
```
source /home/ddk_external/tools/tools_ascendc/set_ascendc_env.sh
```
## 编译运行样例算子

### 1. 准备：获取样例代码<a name="codeready"></a>

 可以使用以下两种方式下载，请选择其中一种进行源码准备。

 - 命令行方式下载（下载时间较长，但步骤简单）。

   ```bash
   # 开发环境，非root用户命令行中执行以下命令下载源码仓。git_clone_path为用户自己创建的某个目录。
   cd ${git_clone_path}
   git clone https://gitee.com/ascend/samples.git
   ```
   **注：如果需要切换到其它tag版本，以v0.5.0为例，可执行以下命令。**
   ```bash
   git checkout v0.5.0
   ```
 - 压缩包方式下载（下载时间较短，但步骤稍微复杂）。

   **注：如果需要下载其它版本代码，请先请根据前置条件说明进行samples仓分支切换。**
   ```bash
   # 1. samples仓右上角选择 【克隆/下载】 下拉框并选择 【下载ZIP】。
   # 2. 将ZIP包上传到开发环境中的普通用户某个目录中，【例如：${git_clone_path}/ascend-samples-master.zip】。
   # 3. 开发环境中，执行以下命令，解压zip包。
   cd ${git_clone_path}
   unzip ascend-samples-master.zip
   ```
### 2.&nbsp;编译运行样例工程
- 若使用框架调用的方式，编译运行操作请参见[FrameworkLaunch](./FrameworkLaunch)。    
## 更新说明
  | 时间         | 更新事项 |
|------------|------|
| 2024/12/10 | 新版readme更新 |