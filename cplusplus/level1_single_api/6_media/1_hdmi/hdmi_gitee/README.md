# HDMI level1 Sample

##本样例为大家学习昇腾软件栈提供参考，非商业目的！
##本样例适配6.4.0及以上版本，支持产品为310B设备。

## 功能描述
hdmi l1 level sample，测试基础HDMI显示功能，当前样例仅支持hdmi0接口进行显示，需要用户保证图片符合样例要求。

## 软件架构
软件架构说明
```
├──————CMakeLists.txt               // 编译脚本
├──————sample_hdmi.c                // 示例代码。
├──————hi_mpi_vo.c                  // 参数初始化函数。
├──————include                      // 示例代码头文件目录。
├──————├──————vo_comm.h       // 示例代码头文件。
```
## 环境要求

- 编译环境操作系统及架构：Ubuntu 18.04 x86_64、Ubuntu 18.04 aarch64、EulerOS aarch64

- 编译器：g++ 或 aarch64-linux-gnu-g++

- 运行环境芯片：Ascend310B

- 已完成昇腾AI软件栈在开发环境、运行环境上的部署

## 编译运行

以运行用户登录编译环境，编译二进制可执行文件。

1. aarch64编译环境的编译方法

设置环境变量。

如下示例，$HOME/Ascend/CANN-7.0表示编译环境runtime标准形态安装包的安装路径，请根据实际情况替换。

export NPU_HOST_LIB=$HOME/Ascend/CANN-7.0/runtime/lib64/

export NPU_HOST_INC=$HOME/Ascend/CANN-7.0/runtime/include/

cd到hdmi level1 sample根目录，依次执行如下命令执行编译。
```
mkdir build
cd build
cmake .. -DCMAKE_C_COMPILER=gcc -DCMAKE_SKIP_RPATH=TRUE
make
```
在“build/“目录下会生成可执行文件sample_hdmi。

2. x86_64编译环境的编译方法

提前在x86_64编译环境安装x86_64版本的CANN-toolkit包

如下示例，$HOME/Ascend/CANN-7.0表示编译环境runtime标准形态安装包的安装路径，请根据实际情况替换。

export NPU_HOST_LIB=$HOME/Ascend/CANN-7.0/runtime/lib64/

export NPU_HOST_INC=$HOME/Ascend/CANN-7.0/runtime/include/

CMakeLists.txt中的ASCEND_HOME变量为编译环境runtime标准形态安装包的安装路径，请根据实际情况替换。

cd到hdmi level1 sample根目录，依次执行如下命令执行编译。
```
mkdir build
cd build
cmake .. -DCMAKE_C_COMPILER=$HOME/Ascend/CANN-7.0/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-gcc -DCMAKE_SKIP_RPATH=TRUE
make
```
在“build/“目录下会生成可执行文件sample_hdmi。

##执行测试程序

>     ./sample_hdmi car.yuv 10
>     参数1：yuv图片文件名，图像格式参数要求如下，分辨率：1080P60Hz，颜色空间：nv12(属于yuv420系列格式)，size：1920*1080
>     参数2：自定义显示时长，可选参数，默认为10s