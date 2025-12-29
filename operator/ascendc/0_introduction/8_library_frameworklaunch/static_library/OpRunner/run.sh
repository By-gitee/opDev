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

export DDK_PATH_ADD=`pwd`/../package/add_custom/
export DDK_PATH_MATMUL=`pwd`/../package/matmul_custom/

export NPU_HOST_LIB=$_ASCEND_INSTALL_PATH/$(arch)-$(uname -s | tr '[:upper:]' '[:lower:]')/lib64
export NPU_HOST_INC=$_ASCEND_INSTALL_PATH/$(arch)-$(uname -s | tr '[:upper:]' '[:lower:]')/include
echo "NPU_HOST_LIB: $NPU_HOST_LIB"
echo "NPU_HOST_INC: $NPU_HOST_INC"
export CUSTLIB_PATH=`pwd`/output
export RELEASE_PATH=`pwd`/../output

mkdir -p $RELEASE_PATH/lib
mkdir -p $RELEASE_PATH/include

function main {
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

    cp -rf $CUSTLIB_PATH/* $RELEASE_PATH/lib
    cp -rf $CUSTLIB_PATH/../inc/* $RELEASE_PATH/include
}

main
