#!/bin/bash
script_path=$(realpath $(dirname $0))

if [[ "$DDK_PATH" == "" ]]
then
  echo "DDK path not set."
  exit
fi

mkdir -p build_out
rm -rf build_out/*
cd build_out

cmake_version=$(cmake --version | grep "cmake version" | awk '{print $3}')

opts=$(python3 $script_path/cmake/util/preset_parse.py $script_path/CMakePresets.json)

target=package
if [ "$1"x != ""x ]; then target=$1; fi
if [ "$cmake_version" \< "3.19.0" ] ; then
  cmake .. $opts -DHOST=true
else
  cmake .. --preset=default -DHOST=true
fi

cmake --build . --target $target -j16

#build ddk
mkdir -p devices
rm -rf devices/*
cd devices

flag=false
if [[ "$ANDROID_NDK" == "" ]] && [[ "$OHOS_NDK" == "" ]]
then
  flag=true
  cmake ../..
  make && make install
else
  if [[ "$ANDROID_NDK" != "" ]]
  then
    cmake ../.. -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
      -DANDROID_ABI="arm64-v8a" \
      -DANDROID_NDK=$ANDROID_NDK \
      -DANDROID_PLATFORM=android-23 \

    make && make install
  else
    cmake -G "Ninja" -D OHOS_STL=c++_static -D OHOS_ARCH=arm64-v8a -D OHOS_PLATFORM=OHOS \
      -D CMAKE_TOOLCHAIN_FILE=$OHOS_NDK/build/cmake/ohos.toolchain.cmake \
      ../..

    cmake --build .
  fi
fi

#install
cd ../
mkdir -p installFile
rm -rf installFile/*
cd installFile

cmake ../../ $opts -DINSTALL=true -DCOMPILE=$flag -DINSTALL_DEVICES=true
make && make install