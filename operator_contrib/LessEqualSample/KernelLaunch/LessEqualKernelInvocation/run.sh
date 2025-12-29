#!/bin/bash
export PRINT_TIK_MEM_ACCESS=FALSE

CURRENT_DIR=$(
    cd $(dirname ${BASH_SOURCE:-$0})
    pwd
); cd $CURRENT_DIR

declare -A VersionMap
VersionMap["Ascend910A"]="Ascend910A"
VersionMap["Ascend910B"]="Ascend910A"
VersionMap["Ascend910ProA"]="Ascend910A"
VersionMap["Ascend910ProB"]="Ascend910A"
VersionMap["Ascend910PremiumA"]="Ascend910A"
VersionMap["Ascend310P1"]="Ascend310P1"
VersionMap["Ascend310P3"]="Ascend310P1"
VersionMap["Ascend910B1"]="Ascend910B1"
VersionMap["Ascend910B2"]="Ascend910B1"
VersionMap["Ascend910B3"]="Ascend910B1"
VersionMap["Ascend910B4"]="Ascend910B1"

BUILD_TYPE="Debug"
INSTALL_PREFIX="${CURRENT_DIR}/out"

SHORT=r:,v:,i:,b:,p:,
LONG=run-mode:,soc-version:,install-path:,build-type:,install-prefix:,
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"

while :
do
    case "$1" in
        (-r | --run-mode )
            RUN_MODE="$2"
            shift 2;;
        (-v | --soc-version )
            SOC_VERSION="$2"
            shift 2;;
        (-i | --install-path )
            ASCEND_INSTALL_PATH="$2"
            shift 2;;
        (-b | --build-type )
            BUILD_TYPE="$2"
            shift 2;;
        (-p | --install-prefix )
            INSTALL_PREFIX="$2"
            shift 2;;
        (--)
            shift;
            break;;
        (*)
            echo "[ERROR] Unexpected option: $1";
            break;;
    esac
done


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

RUN_MODE_LIST="cpu sim npu"
if [[ " $RUN_MODE_LIST " != *" $RUN_MODE "* ]]; then
    RUN_MODE="cpu"
fi

# in case of running op in simulator, use stub so instead
if [ "${RUN_MODE}" = "sim" ]; then
    export LD_LIBRARY_PATH=$_ASCEND_INSTALL_PATH/runtime/lib64/stub:$LD_LIBRARY_PATH
    if [ ! $CAMODEL_LOG_PATH ]; then
        export CAMODEL_LOG_PATH=./sim_log
    fi
    rm -rf $CAMODEL_LOG_PATH
    mkdir -p $CAMODEL_LOG_PATH
fi
source $_ASCEND_INSTALL_PATH/bin/setenv.bash

if [[ " ${!VersionMap[*]} " != *" $SOC_VERSION "* ]]; then
    SOC_VERSION="Ascend310P1"
fi
_SOC_VERSION=${VersionMap[$SOC_VERSION]}



set -e
export LD_LIBRARY_PATH=$(pwd)/out/lib:$(pwd)/out/lib64\
:${_ASCEND_INSTALL_PATH}/tools/tikicpulib/lib\
:${_ASCEND_INSTALL_PATH}/tools/tikicpulib/lib/${VersionMap[$SOC_VERSION]}\
:${_ASCEND_INSTALL_PATH}/tools/simulator/${SOC_VERSION}/lib\
:$LD_LIBRARY_PATH

rm -rf build out PROF* cceprint npuchk *log *.vcd

mkdir build
cmake -B build                               \
    -DRUN_MODE=${RUN_MODE}                   \
    -DSOC_VERSION=${SOC_VERSION}             \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE}         \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
    -DASCEND_CANN_PACKAGE_PATH=${_ASCEND_INSTALL_PATH}
cmake --build build -j -v
cmake --install build

cp ./out/bin/less_equal ./
rm -rf input output
mkdir -p input output
python3 scripts/gen_data.py
./less_equal
md5sum output/*.bin
python3 scripts/verify_result.py output/output_y.bin output/golden.bin

rm -rf *log *.vcd