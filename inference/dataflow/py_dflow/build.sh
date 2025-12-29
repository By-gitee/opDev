#!/bin/bash
# Copyright 2024-2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e

BASEPATH=$(cd "$(dirname $0)"; pwd)
OUTPUT_PATH="${BASEPATH}/output"
PYTHON_PATH=python3
BUILD_RELATIVE_PATH="build"

# print usage message
usage() {
  echo "Usage:"
  echo "  sh build.sh [-h | --help] [-v | --verbose] [-j<N>] [--build_type=<Release|Debug>]"
  echo "              [--ascend_install_path=<PATH>] [--output_path=<PATH>] [--python_path=<PATH>]"
  echo ""
  echo "Options:"
  echo "    -h, --help     Print usage"
  echo "    -v, --verbose  Display build command"
  echo "    -j<N>          Set the number of threads used for building DFlow, default is 8"
  echo "    --build_type=<Release|Debug>"
  echo "                   Set build type, default Release"
  echo "    --ascend_install_path=<PATH>"
  echo "                   Set ascend package install path, default /usr/local/Ascend/ascend-toolkit/latest"
  echo "    --output_path=<PATH>"
  echo "                   Set output path, default ./output"
  echo "    --python_path=<PATH>"
  echo "                   Set output path, for example:/usr/local/bin/python3.9, default python3"
  echo ""
}

# check value of build_type option
# usage: check_build_type build_type
check_build_type() {
  arg_value="$1"
  if [ "X$arg_value" != "XRelease" ] && [ "X$arg_value" != "XDebug" ]; then
    echo "Invalid value $arg_value for option --$2"
    usage
    exit 1
  fi
}

# parse and set options
checkopts() {
  VERBOSE=""
  THREAD_NUM=8
  ASCEND_INSTALL_PATH="/usr/local/Ascend/ascend-toolkit/latest"
  CMAKE_BUILD_TYPE="Release"

  # Process the options
  parsed_args=$(getopt -a -o j:hv -l help,verbose,ascend_install_path:,output_path:,python_path: -- "$@") || {
    usage
    exit 1
  }

  eval set -- "$parsed_args"

  while true; do
    case "$1" in
      -h | --help)
        usage
        exit 0
        ;;
      -j)
        THREAD_NUM="$2"
        shift 2
        ;;
      -v | --verbose)
        VERBOSE="VERBOSE=1"
        shift
        ;;
      --build_type)
        check_build_type "$2" build_type
        CMAKE_BUILD_TYPE="$2"
        shift 2
        ;;
      --ascend_install_path)
        ASCEND_INSTALL_PATH="$(realpath $2)"
        shift 2
        ;;
      --output_path)
        OUTPUT_PATH="$(realpath $2)"
        shift 2
        ;;
      --python_path)
        PYTHON_PATH="$2"
        shift 2
        ;;
      --)
        shift
        break
        ;;
      *)
        echo "Undefined option: $1"
        usage
        exit 1
        ;;
    esac
  done
  set +e
  python_full_path=$(which ${PYTHON_PATH})
  set -e
  if [ -z "${python_full_path}" ]; then
    echo "Error: python_path=${PYTHON_PATH} is not exist"
    exit 1
  else
    PYTHON_PATH=${python_full_path}
    echo "use python: ${PYTHON_PATH}"
  fi
}

mk_dir() {
  local create_dir="$1"  # the target to make
  mkdir -pv "${create_dir}"
  echo "created ${create_dir}"
}

# DFlow build start
cmake_generate_make() {
  local build_path="$1"
  local cmake_args="$2"
  mk_dir "${build_path}"
  cd "${build_path}"
  echo "${cmake_args}"
  cmake ${cmake_args} ..
  if [ 0 -ne $? ]; then
    echo "execute command: cmake ${cmake_args} .. failed."
    exit 1
  fi
}

# create build path
build_dflow() {
  echo "create build directory and build DFlow"
  cd "${BASEPATH}"

  BUILD_PATH="${BASEPATH}/${BUILD_RELATIVE_PATH}/"
  CMAKE_ARGS="-D BUILD_OPEN_PROJECT=True \
              -D ASCEND_INSTALL_PATH=${ASCEND_INSTALL_PATH} \
              -D CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
              -D CMAKE_INSTALL_PREFIX=${OUTPUT_PATH} \
              -D HI_PYTHON=${PYTHON_PATH}"

  echo "CMAKE_ARGS is: $CMAKE_ARGS"
  cmake_generate_make "${BUILD_PATH}" "${CMAKE_ARGS}"

  make dataflow_python ${VERBOSE} -j${THREAD_NUM} && make install

  if [ 0 -ne $? ]; then
    echo "execute command: make ${VERBOSE} -j${THREAD_NUM} && make install failed."
    return 1
  fi
  echo "DFlow build success!"
}

main() {
  cd "${BASEPATH}"
  checkopts "$@"

  g++ -v
  mk_dir ${OUTPUT_PATH}
  echo "---------------- DFlow build start ----------------"
  build_dflow || { echo "DFlow build failed."; exit 1; }
  echo "---------------- DFlow build finished ----------------"
}

main "$@"
