#!/bin/bash
CURRENT_DIR=$(
    cd $(dirname ${BASH_SOURCE:-$0})
    pwd
)

BUILD_TYPE="Debug"
INSTALL_PREFIX="${CURRENT_DIR}/out"

SHORT=v:,i:,b:,p:,
LONG=soc-version:,install-path:,build-type:,install-prefix:,
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"
SOC_VERSION="Ascend310P3"
RUN_MODE="npu"
while :; do
    case "$1" in
    -v | --soc-version)
        SOC_VERSION="$2"
        shift 2
        ;;
    -i | --install-path)
        ASCEND_INSTALL_PATH="$2"
        shift 2
        ;;
    -b | --build-type)
        BUILD_TYPE="$2"
        shift 2
        ;;
    -p | --install-prefix)
        INSTALL_PREFIX="$2"
        shift 2
        ;;
    --)
        shift
        break
        ;;
    *)
        echo "[ERROR]: Unexpected option: $1"
        break
        ;;
    esac
done

VERSION_LIST="Ascend310P1 Ascend310P3 Ascend910B1 Ascend910B2 Ascend910B3 Ascend910B4"
if [[ " $VERSION_LIST " != *" $SOC_VERSION "* ]]; then
    echo "[ERROR]: SOC_VERSION should be in [$VERSION_LIST]"
    exit -1
fi

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

export ASCEND_TOOLKIT_HOME=${_ASCEND_INSTALL_PATH}
export ASCEND_HOME_PATH=${_ASCEND_INSTALL_PATH}
echo "[INFO]: Current compile soc version is ${SOC_VERSION}"
source ${_ASCEND_INSTALL_PATH}/bin/setenv.bash

set -e
rm -rf build out
mkdir -p build
cmake -B build \
    -DRUN_MODE=${RUN_MODE} \
    -DSOC_VERSION=${SOC_VERSION} \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
    -DASCEND_CANN_PACKAGE_PATH=${_ASCEND_INSTALL_PATH}
cmake --build build -j
cmake --install build

rm -f ascendc_kernels_bbit
cp ./out/bin/ascendc_kernels_bbit ./
rm -rf input output
mkdir -p input output
python3 scripts/gen_data.py
file_path=output_msg.txt
(
    export LD_LIBRARY_PATH=$(pwd)/out/lib:$(pwd)/out/lib64:${_ASCEND_INSTALL_PATH}/lib64:$LD_LIBRARY_PATH
    if [[ "$RUN_WITH_TOOLCHAIN" -eq 1 ]]; then
        msprof op --application=./ascendc_kernels_bbit | tee $file_path
    else
        ./ascendc_kernels_bbit | tee $file_path
    fi
)
md5sum output/*.bin
python3 scripts/verify_result.py output/output.bin output/golden.bin

check_msg_half="printf half"
check_msg_int="printf tailM"
check_msg_uint="printf mSingleBlocks"
check_msg_pointer="printf pinter"
check_msg_bool="printf isTransA"

count_int=$(grep -c "$check_msg_int" $file_path)
count_half=$(grep -c "$check_msg_half" $file_path)
count_bool=$(grep -c "$check_msg_bool" $file_path)
count_pointer=$(grep -c "$check_msg_pointer" $file_path)
count_uint=$(grep -c "$check_msg_uint" $file_path)

if [ $count_int -eq 0 ]; then
    echo "[ERROR]: $check_msg_int is expected, but not found."
    exit 1
fi

if [ $count_bool -eq 0 ]; then
    echo "[ERROR]: $check_msg_bool is expected, but not found."
    exit 1
fi

if [ $count_half -eq 0 ]; then
    echo "[ERROR]: $check_msg_half is expected, but not found."
    exit 1
fi

if [ $count_pointer -eq 0 ]; then
    echo "[ERROR]: $check_msg_pointer is expected, but not found."
    exit 1
fi

if [ $count_uint -eq 0 ]; then
    echo "[ERROR]: $check_msg_uint is expected, but not found."
    exit 1
fi
