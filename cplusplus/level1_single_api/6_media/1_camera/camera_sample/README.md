# Camera level1 Sample

##本样例为大家学习昇腾软件栈提供参考，非商业目的！
##本样例适配6.4.0及以上版本，支持产品为310B设备。

## 功能描述
camera l1 level sample，从sensor获取图像(实例为树莓派摄像头)，通过camera流程处理之后获得yuv图像传入后端进行处理。

## 软件架构
软件架构说明
```
├──————CMakeLists.txt        // 编译脚本
├──————sensor                // sensor相关源码，当前仅支持树莓派imx219摄像头。
├──————common                // 示例代码通用适配层。
├──————sensor_sample         // 示例代码sensor相关配置，sample如何调用sensor源码。
├──————main.c                //示例代码主函数。
├──————vi_with_sensor.c      //示例代码主流程。
├──————vi_with_sensor.h      //示例代码头文件。
```
## 环境要求

- 编译环境操作系统及架构：Ubuntu 18.04 x86_64、Ubuntu 18.04 aarch64、EulerOS aarch64

- 编译器：g++ 或 aarch64-linux-gnu-g++

- 运行环境芯片：Ascend310B

- 已完成昇腾AI软件栈在开发环境、运行环境上的部署

## 编译运行

以运行用户登录编译环境，编译代码，首先编译sensor so相关代码，再生成二进制可执行文件。

1. aarch64编译环境的编译方法

设置环境变量。

如下示例，$HOME/Ascend表示编译环境runtime标准形态安装包的安装路径，请根据实际情况替换。

export NPU_HOST_LIB=$HOME/Ascend/runtime/lib64/stub/
cd到sensor/sony_imx219目录，依次执行如下命令执行编译；imx477同理，cd到sensor/sony_imx477目录下进行编译
```
mkdir build
cd build
cmake .. -DCMAKE_C_COMPILER=gcc -DCMAKE_SKIP_RPATH=TRUE
make
```
在“$HOME/Ascend/runtime/lib64/stub/“目录下会生成libsns_imx219.so，libsns_imx477.so。
之后cd到camera_l1_sample根目录，依次执行如下命令执行编译， **编译器根据实际toolkit包安装位置自行调整** 。
```
mkdir build
cd build
cmake .. -DCMAKE_C_COMPILER=gcc -DCMAKE_SKIP_RPATH=TRUE
make
```
在“build/“目录下会生成可执行文件vi_l1_sample。

2. x86_64编译环境的编译方法

提前在x86_64编译环境安装x86_64版本的CANN-toolkit包

1. 设置环境变量。

如下示例，$HOME/Ascend表示编译环境runtime标准形态安装包的安装路径，请根据实际情况替换。

“export NPU_HOST_LIB=$HOME/Ascend/CANN-6.4/runtime/lib64/stub/aarch64”
cd到sensor/sony_imx219目录，依次执行如下命令执行编译，imx477同理，cd到sensor/sony_imx477目录下进行编译。

 **编译器根据实际toolkit包安装位置自行调整** 。
```
mkdir build
cd build
cmake .. -DCMAKE_C_COMPILER=$HOME/Ascend/CANN-6.4/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-gcc -DCMAKE_SKIP_RPATH=TRUE
make
```
在“$HOME/Ascend/runtime/lib64/stub/“目录下会生成libsns_imx219.so。
2. 之后cd到camera_l1_sample根目录，依次执行如下命令执行编译。
```
mkdir build
cd build
cmake .. -DCMAKE_C_COMPILER=$HOME/Ascend/CANN-6.4/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-gcc -DCMAKE_SKIP_RPATH=TRUE
make
```
在“build/“目录下会生成可执行文件vi_l1_sample。

##执行测试程序

>     ./vi_l1_sample 1 1 1
>     参数1：测试程序模式，1为sensor出yuv图，2为出raw图
>     参数2：camera num，可选1-2
>     参数3：sensor_type，1为imx219,2为imx477

##可能遇到的问题
1. 执行报错，可以看/var/log/npu/slog/debug/device-os/进行简单定位，如果是dlopen so失败的情况，将/lib64/目录添加到环境变量中即可
    
