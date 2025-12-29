# Audio level1 Sample

##本样例为大家学习昇腾软件栈提供参考，非商业目的！
##本样例适配6.4.0及以上版本，支持产品为310B设备。

## 功能描述
audio l1 level sample，测试播音和录音功能。播音功能获取指定pcm文件的音频内容并播放；录音功能从mic获取音频信号，通过audio流程处理之后生成pcm音频文件。

## 软件架构
软件架构说明
```
├──————CMakeLists.txt               // 编译脚本
├──————sample_audio.c               // 示例代码。
├──————include                      // 示例代码头文件目录。
├──————├──————sample_comm_audio.h   // 示例代码头文件。
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

如下示例，$HOME/Ascend表示编译环境runtime标准形态安装包的安装路径，请根据实际情况替换。

export NPU_HOST_LIB=$HOME/Ascend/CANN-7.0/runtime/lib64/

export NPU_HOST_INC=$HOME/Ascend/CANN-7.0/runtime/include/

cd到audio level1 sample根目录，依次执行如下命令执行编译。
```
mkdir build
cd build
cmake .. -DCMAKE_C_COMPILER=gcc -DCMAKE_SKIP_RPATH=TRUE
make
```
在“build/“目录下会生成可执行文件sample_audio。

2. x86_64编译环境的编译方法

提前在x86_64编译环境安装x86_64版本的CANN-toolkit包

设置环境变量。

如下示例，$HOME/Ascend表示编译环境runtime标准形态安装包的安装路径，请根据实际情况替换。

export NPU_HOST_LIB=$HOME/Ascend/CANN-7.0/runtime/lib64/

export NPU_HOST_INC=$HOME/Ascend/CANN-7.0/runtime/include/

cd到audio level1 sample根目录，依次执行如下命令执行编译。
```
mkdir build
cd build
cmake .. -DCMAKE_C_COMPILER=$HOME/Ascend/CANN-7.0/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-gcc -DCMAKE_SKIP_RPATH=TRUE
make
```
在“build/“目录下会生成可执行文件sample_audio。

##执行测试程序

>     ./sample_audio play qzgy_48kHz_16bit_1chn.pcm
>     ./sample_audio capture ai_dump_16_l_dev2.pcm &
>     参数1：测试程序模式，play为播音，capture为录音
>     参数2：pcm文件名，文件格式为采样率48kHz，位宽为16bit的单声道音频
>     注意：录音任务需要执行 ps aux|grep audio 查询进程号，然后手动 kill -12 [pid] 结束进程