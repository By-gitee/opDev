# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)

# project information
project(vi_l1_sample)

set(LIB_PATH $ENV{NPU_HOST_LIB})
# Dynamic libraries in the stub directory can only be used for compilation
if (NOT DEFINED ENV{NPU_HOST_LIB})
    set(LIB_PATH "/usr/local/Ascend/ascend-toolkit/latest/CANN-6.4/runtime/lib64/stub/aarch64")
    message(STATUS "set default LIB_PATH: ${LIB_PATH}")
else ()
    message(STATUS "env LIB_PATH: ${LIB_PATH}")
endif()

set(INC_PATH $ENV{NPU_HOST_INC})
# Dynamic libraries in the stub directory can only be used for compilation
if (NOT DEFINED ENV{NPU_HOST_INC})
    set(INC_PATH "/usr/local/Ascend/ascend-toolkit/latest/CANN-6.4/runtime/include")
    message(STATUS "set default INC_PATH: ${INC_PATH}")
else ()
    message(STATUS "env INC_PATH: ${INC_PATH}")
endif()

link_directories(
    ${LIB_PATH}
    ${LIB_PATH}/stub
)

add_executable(vi_l1_sample
  main.c
  vi_with_sensor.c
  ./common/sample_comm_isp.c
  ./common/sample_comm_vi.c
  ./sensor_sample/sensor_management.c
  ./common/image_dump_util.c
)

include_directories(vi_l1_sample
    ${INC_PATH}
    ${INC_PATH}/acl/media
    ./sensor_sample
    ./common
    ./sensor/include
    ./sensor/sony_imx219
    ./sensor/sony_imx477
)

add_compile_options(vi_l1_sample
    -O2
    -Wall
    -fpic
)

target_link_libraries(vi_l1_sample
    acl_dvpp_mpi
    acl_isp_mpi
    sns_imx219
    sns_imx477
    acl_vi_mpi
    acl_vpss_mpi
    acl_isp_ae_mpi
    acl_isp_awb_mpi
    ascendcl
    pthread
)
