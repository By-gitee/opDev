#!/bin/bash
ScriptPath="$( cd "$(dirname "$BASH_SOURCE")" ; pwd -P )"

function build()
{
  if [ -d ${ScriptPath}/../build ];then
    rm -rf ${ScriptPath}/../build
  fi

  mkdir -p ${ScriptPath}/../build
  cd ${ScriptPath}/../build

  cmake ../src -DCMAKE_CXX_COMPILER=g++ -DCMAKE_SKIP_RPATH=TRUE
  if [ $? -ne 0 ];then
    echo "[ERROR] cmake error, Please check your environment!"
    return 1
  fi
  make
  if [ $? -ne 0 ];then
    echo "[ERROR] build failed, Please check your environment!"
    return 1
  fi
  cd - > /dev/null
}

function main()
{
  build
  if [ $? -ne 0 ];then
   return 1
  fi

  clear;clear
  rm -rf ../input/*.bin &> /dev/null
  rm -rf ../output/*.bin &> /dev/null

  running_command="./llama_run"
  cd ${ScriptPath}/../output
  ${running_command} > current.log &

  cd ../scripts
  python3 input_trans.py
}
main
