#!/bin/bash
SHORT=v:,i:,
LONG=soc-version:,install-path:,
OPTS=$(getopt -a --options $SHORT --longoptions $LONG -- "$@")
eval set -- "$OPTS"

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
export ASCEND_HOME_PATH=$_ASCEND_INSTALL_PATH

OP_NAME=AddCustomTilingSink
rm -rf CustomOp
# Generate the op framework
msopgen gen -i $OP_NAME.json -c ai_core-${SOC_VERSION} -lan cpp -out CustomOp
# Copy op implementation files to CustomOp
cp -rf framework CustomOp/;cp -rf op_host CustomOp/;cp -rf op_kernel CustomOp/
#Add Device Compile Task in op_host/CMakeLists.txt
sed -i '$a ascendc_device_library( TARGET cust_opmaster\n                        OPTION SHARED\n                        SRC ${CMAKE_CURRENT_SOURCE_DIR}/add_custom_tiling_sink_tiling.cpp)' CustomOp/op_host/CMakeLists.txt
# Build CustomOp project
(cd CustomOp && bash build.sh)