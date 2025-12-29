# mipitx Sample

##本样例为大家学习昇腾软件栈提供参考，非商业目的！
##本样例适配6.4.0及以上版本，支持产品为310B设备。

## 功能描述
mipitx sample，通过mipi接口连接屏幕后，执行用例，屏幕上会显示图像；

## 软件架构
软件架构说明
```
|————raspberry_sample          //sample目录
|  |————CMakeLists.txt         //编译脚本
|  |————raspberry_demo.c       //示例代码主函数
|  |————include                //示例代码相关头文件目录
|  |  |————hi_mipi_tx.h
|  |  |————vo_comm.h
|  |  |————func.h
|  |————src                    //示例代码相关源文件目录
|  |  |————vo_init.c
|  |  |————vo_mem.c
|  |  |————vo_mipitx.c
```
## 环境要求

- 编译环境操作系统及架构：Ubuntu 18.04 x86_64、Ubuntu 18.04 aarch64、EulerOS aarch64

- 编译器：g++ 或 aarch64-linux-gnu-g++

- 运行环境芯片：Ascend310B

- 已完成昇腾AI软件栈在开发环境、运行环境上的部署

## 编译运行

以运行用户登录编译环境，编译代码，生成二进制可执行文件。

1. aarch64编译环境的编译方法

设置环境变量,如下示例，$HOME/Ascend/CANN-7.0表示编译环境runtime标准形态安装包的安装路径，请根据实际情况替换。

export NPU_HOST_LIB=$HOME/Ascend/CANN-7.0/runtime/lib64/

export NPU_HOST_INC=$HOME/Ascend/CANN-7.0/runtime/include/

cd到raspberry_sample目录，依次执行如下命令执行编译。
```
mkdir build
cd build
cmake .. -DCMAKE_C_COMPILER=gcc -DCMAKE_SKIP_RPATH=TRUE
make
```
在“build/“目录下会生成可执行文件raspberry_demo。

2. x86_64编译环境的编译方法

提前在x86_64编译环境安装x86_64版本的CANN-toolkit包

设置环境变量,如下示例，$HOME/Ascend/CANN-7.0表示编译环境runtime标准形态安装包的安装路径，请根据实际情况替换。

export NPU_HOST_LIB=$HOME/Ascend/CANN-7.0/runtime/lib64/

export NPU_HOST_INC=$HOME/Ascend/CANN-7.0/runtime/include/

cd到raspberry_sample目录，依次执行如下命令执行编译。
```
mkdir build
cd build
cmake .. -DCMAKE_C_COMPILER=$HOME/Ascend/CANN-7.0/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-gcc -DCMAKE_SKIP_RPATH=TRUE
make
```
在“build/“目录下会生成可执行文件raspberry_demo。

##执行测试程序
准备一张800x480大小的YVU_SP_420格式的图片，放在执行文件同级目录下，执行：
>     ./raspberry_demo 图片文件名称
