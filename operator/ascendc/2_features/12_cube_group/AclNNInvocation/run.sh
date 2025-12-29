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
source $_ASCEND_INSTALL_PATH/bin/setenv.bash
export DDK_PATH=$_ASCEND_INSTALL_PATH
export NPU_HOST_LIB=$_ASCEND_INSTALL_PATH/$(arch)-$(uname -s | tr '[:upper:]' '[:lower:]')/lib64

function main {
    # 1. 清除遗留生成文件和日志文件
    rm -rf $HOME/ascend/log/*

    # 2. 生成输入数据和真值数据
    cd $CURRENT_DIR/run/out/test_data/data
    python3 generate_data.py
    if [ $? -ne 0 ]; then
        echo "[ERROR]: Generate input data failed!"
        return 1
    fi
    echo "[INFO]: Generate input data success!"

    # 3. 编译acl可执行文件
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
    export LD_LIBRARY_PATH=$_ASCEND_INSTALL_PATH/opp/vendors/customize/op_api/lib:$LD_LIBRARY_PATH
    cd $CURRENT_DIR/run/out
    echo "[INFO]: Execute op!"
    ./execute_cube_group_custom_op
    if [ $? -ne 0 ]; then
        echo "[ERROR]: Acl executable run failed! please check your project!"
        return 1
    fi
    echo "[INFO]: Acl executable run success!"

    # 5. 比较真值文件
    cd $CURRENT_DIR
    python3 $CURRENT_DIR/scripts/verify_result.py \
        $CURRENT_DIR/run/out/result_files/output_0.bin \
        $CURRENT_DIR/run/out/test_data/data/golden.bin
    if [ $? -ne 0 ]; then
        echo "[ERROR]: Verify result failed!"
        return 1
    fi
}

main
