# ACLLite-cplusplus快速部署

## 安装步骤

  设置环境变量，配置程序编译依赖的头文件，库文件路径。“$HOME/Ascend”请替换“Ascend-cann-toolkit”包的实际安装路径。

   ```
    export DDK_PATH=$HOME/Ascend/ascend-toolkit/latest
    export NPU_HOST_LIB=$DDK_PATH/runtime/lib64/stub
    export THIRDPART_PATH=${DDK_PATH}/thirdpart
    export LD_LIBRARY_PATH=${THIRDPART_PATH}/lib:$LD_LIBRARY_PATH
   ```

  创建THIRDPART_PATH路径
   ```
    mkdir -p ${THIRDPART_PATH}
   ```
  注：源码安装ffmpeg主要是为了acllite库的安装

  执行以下命令安装x264

   ```
    # 下载x264
    cd ${HOME}
    git clone https://code.videolan.org/videolan/x264.git
    cd x264
    # 安装x264
    ./configure --enable-shared --disable-asm
    make
    sudo make install
    sudo cp /usr/local/lib/libx264.so.164 /lib
   ```   
    
  执行以下命令安装ffmpeg

   ```
    # 下载ffmpeg
    cd ${HOME}
    wget http://www.ffmpeg.org/releases/ffmpeg-4.1.3.tar.gz --no-check-certificate
    tar -zxvf ffmpeg-4.1.3.tar.gz
    cd ffmpeg-4.1.3
    # 安装ffmpeg
    ./configure --enable-shared --enable-pic --enable-static --disable-x86asm --enable-libx264 --enable-gpl --prefix=${THIRDPART_PATH}
    make -j8
    make install
   ```   
    
  执行以下命令安装acllite

   ```
    cd ${HOME}/samples/inference/acllite/cplusplus
    make
    make install
   ```   