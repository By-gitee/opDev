#!/bin/bash
CURRENT_DIR=$(
    cd $(dirname ${BASH_SOURCE:-$0})
    pwd
)

if [ -n "$ASCEND_INSTALL_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_INSTALL_PATH
elif [ -n "$ASCEND_HOME_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
else
    if [ -d "$HOME/Ascend/ascend-toolkit/latest" ]; then
        _ASCEND_INSTALL_PATH=$HOME/Ascend/ascend-toolkit/latest
    else
        _ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
    fi
fi

export BASIC_PATH=`pwd`/../output/
export DDK_PATH_ADD=`pwd`/../package/add_custom
export DDK_PATH_MATMUL=`pwd`/../package/matmul_custom

if [ "$1" = "Dynamic" ]; then
    export COMPILE_MODE=DYNAMIC_ORI
else
    export COMPILE_MODE=$2
fi

export NPU_HOST_LIB=$_ASCEND_INSTALL_PATH/$(arch)-$(uname -s | tr '[:upper:]' '[:lower:]')/lib64
export NPU_HOST_INC=$_ASCEND_INSTALL_PATH/$(arch)-$(uname -s | tr '[:upper:]' '[:lower:]')/include

function main {
    # 1. 清除遗留生成文件和日志文件
    rm -rf $HOME/ascend/log/*
    rm -rf ./input/*.bin
    rm -rf ./output/*.bin

    # 2. 生成输入数据和真值数据
    cd $CURRENT_DIR
    python3 scripts_add/gen_data.py
    if [ $? -ne 0 ]; then
        echo "[ERROR]: Generate add input data failed!"
        return 1
    fi
    python3 scripts_matmul/gen_data.py
    if [ $? -ne 0 ]; then
        echo "[ERROR]: Generate matmul input data failed!"
        return 1
    fi
    echo "[INFO]: Generate input data success!"

    # 3. 编译可执行文件
    cd $CURRENT_DIR
    rm -rf build
    mkdir -p build
    cd build
    cmake ../src -DCMAKE_SKIP_RPATH=TRUE
    if [ $? -ne 0 ]; then
        echo "[ERROR]: Cmake failed!"
        return 1
    fi
    echo "[INFO]: Cmake success!"
    make
    if [ $? -ne 0 ]; then
        echo "[ERROR]: Make failed!"
        return 1
    fi
    echo "[INFO]: Make success!"

    # 4. 运行可执行文件
    export LD_LIBRARY_PATH=$NPU_HOST_LIB/:$BASIC_PATH/lib:$DDK_PATH_ADD/lib:$DDK_PATH_MATMUL/lib:$LD_LIBRARY_PATH:./
    cd $CURRENT_DIR/output
    echo "[INFO]: Execute op!"
    ./execute_static_op
    if [ $? -ne 0 ]; then
        echo "[ERROR]: Acl executable run failed! please check your project!"
        return 1
    fi
    echo "[INFO]: Acl executable run success!"

    # 5. 精度比对
    cd $CURRENT_DIR
    python3 scripts_matmul/verify_result.py output/output_z_matmul.bin output/golden_matmul.bin
    if [ $? -ne 0 ]; then
        echo "[ERROR]: Verify matmul result failed!"
        return 1
    fi

    python3 scripts_add/verify_result.py output/output_z_add.bin output/golden_add.bin
    if [ $? -ne 0 ]; then
        echo "[ERROR]: Verify add result failed!"
        return 1
    fi
}

main
