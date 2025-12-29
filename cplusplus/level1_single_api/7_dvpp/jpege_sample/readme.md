# readme<a name="ZH-CN_TOPIC_0000001072529927"></a>

### 本样例为大家学习昇腾软件栈提供参考，非商业目的！
### 本样例适配5.0.3及以上版本，支持产品为Atlas 推理系列产品（Ascend 310P处理器）、Atlas 200/500 A2推理产品。


## 功能描述<a name="section09679311389"></a>

DVPP中的JPEGE功能模块，实现将YUV格式图片编码成.jpg图片。

## 原理介绍<a name="section19985135703818"></a>

样例中的关键接口调用流程如下：

![输入图片说明](1596872260762.png)

## 目录结构<a name="section86232112399"></a>

```
├──————CMakeLists.txt            // 编译脚本，调用src目录下的CMakeLists文件
├──————src/CMakeLists.txt        // 编译脚本
├──————src/common                // 示例代码文件所在的目录
├──————src/sample_jpege.cpp      // 示例代码
```

## 环境要求<a name="section10528164623911"></a>

-   操作系统及架构：Ubuntu 18.04 x86\_64、Ubuntu 18.04 aarch64、EulerOS aarch64
-   编译器：g++ 或 aarch64-linux-gnu-g++
-   芯片：Atlas 推理系列产品（Ascend 310P处理器）、Atlas 200/500 A2推理产品。
-   已完成昇腾AI软件栈在开发环境、运行环境上的部署。

## 准备测试数据<a name="section13765133092318"></a>

请单击以下链接，获取该样例的测试图片数据。

[dvpp_venc_128x128_nv12.yuv](https://obs-9be7.obs.cn-east-2.myhuaweicloud.com/data/dvpp_sample_input_data/dvpp_venc_128x128_nv12.yuv)

## 编译运行<a name="section3789175815018"></a>

1. 以运行用户登录开发环境，编译代码。此时存在以下两种情况：
   1. 当开发环境与运行环境的操作系统架构相同时，例如两者都是X86架构，或者都是AArch64架构，此时编译流程参考如下：
        1. 设置环境变量。

            如下示例，$HOME/Ascend表示编译环境runtime标准形态安装包的安装路径，latest对应安装包版本号，请根据实际情况替换。

            ```
            export DDK_PATH=$HOME/Ascend/latest
            export NPU_HOST_LIB=$DDK_PATH/runtime/lib64/stub/
            ```
        2. 切换到jpege\_sample目录，依次执行如下命令执行编译。

            ```
            mkdir build
            cd build
            cmake .. -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE
            make
            ```
            在“jpege\_sample/build/src“目录下会生成可执行程序jpege\_demo。

   2. 当开发环境与运行环境的操作系统架构不同时，例如开发环境是X86架构、运行环境是AArch64架构，此时涉及交叉编译，需要在开发环境安装包含AArch64工具链的软件包，并将相关环境变量指向AArch64版本路径，具体编译流程参考如下：
        1. 提前在x86_64编译环境安装x86_64版本的CANN-toolkit包
        2. 设置环境变量。

            如下示例，$HOME/Ascend表示编译环境runtime标准形态安装包的安装路径，latest对应安装包版本号，请根据实际情况替换。

            ```
            export DDK_PATH=$HOME/Ascend/latest
            export NPU_HOST_LIB=$DDK_PATH/runtime/lib64/stub/aarch64
            ```
        3. 切换到jpege\_sample目录，依次执行如下命令执行编译。

            ```
            mkdir build
            cd build
            cmake .. -DCMAKE_CXX_COMPILER=$HOME/Ascend/latest/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++ -DCMAKE_SKIP_RPATH=TRUE
            make
            ```
            在“jpege\_sample/build/src“目录下会生成可执行程序jpege\_demo。

2. 以运行用户将开发环境的“jpege\_sample/build/src“目录下的可执行程序jpege\_demo以及[准备测试数据](#section13765133092318)中测试数据上传到运行环境（Host）的同一目录下，例如“$HOME/dvpp/jpege\_sample“。

3. 运行应用。

   1. 切换到可执行文件jpege\_demo所在的目录，例如“$HOME/dvpp/jpege\_sample“，给该目录下的jpege\_demo文件加执行权限。

      ```
      chmod +x  jpege_demo
      ```

   2. 设置环境变量。“$HOME/Ascend“表示runtime标准形态安装包的安装路径，请根据实际情况替换。

      ```
      export LD_LIBRARY_PATH=$HOME/Ascend/runtime/lib64
      ```

   3. <a name="li163081446765"></a>运行应用。

      - 示例描述：使用JPEGE编码器将dvpp\_venc\_128x128\_nv12.yuv编码为jpg图片。

      - 输入图像：宽128像素、高128像素、名称为“dvpp\_venc\_128x128\_nv12.yuv”的YUV420sp数据。

      - 输出图像：宽128像素、高128像素的jpg图片，名称为snap_chnl0_no0.jpg ，表示通道0的第0张图像。

      - 运行应用的命令示例如下：

        ```
        ./jpege_demo --in_image_file ./dvpp_venc_128x128_nv12.yuv --img_width 128 --img_height 128 --in_format 1 --chn_num 1
        ```

      - 运行可执行文件的通用参数说明如下所示：

        - chn\_width：创建通道的宽度, 范围\[32, 8192\]。如果用户没有传入通道宽高，默认使用图像宽高作为通道宽高。
        - chn\_height：创建通道的高度, 范围\[32, 8192\]。如果用户没有传入通道宽高，默认使用图像宽高作为通道宽高。
        - img\_width：输入图片的宽度，范围\[32, 8192\]。
        - img\_height：输入图片的高度，范围\[32, 8192\]。
        - in\_format：YUV数据格式。
          - 1：YUV420SP
          - 2：YVU420SP
          - 7：YUYV422PACKED
          - 8：UYVY422PACKED
          - 9：YVYU422PACKED
          - 10：VYUY422PACKED
        - in\_bitwidth：图片位宽，仅支持8bit。
        - chn\_num：创建编码通道的数目，最大不得超过256路。
        - in\_image\_file：输入图像文件的路径，包含文件名。
        - out\_image\_file：输出文件的路径，包含文件名。
        - level：编码质量，默认值100，取值范围\[1, 100\]
        - save：是否保存输出码流。
          - 默认1，0：不保留（主要用于性能测试）
          - 非0：保留
        - chn\_start：编码起始通道号，范围\[0, 255\]，不指定则从0开始。
        - performance：性能测试标识。
          - 默认0：功能测试
          - 非0：性能测试
        - per\_count：性能统计循环次数，默认值300，取值范围\[1, ∞\)。
        - mem\_count：使用的输入buffer大小，默认为100。
        - one\_thread：通知用户取走输出码流。
          - 默认1：单线程单通道
          - 0：单线程多通道
        - sync\_enc：是同步还是异步送帧和接受码流。
          - 默认0：同步
          - 非0：异步
        - frame\_count：输入文件中包含的帧数。
        - zero\_copy：是否指定编码结果的输出地址。
          - 默认0：不指定输出地址
          - 非0：指定输出地址
        - device\_id：device id, 默认为0。
