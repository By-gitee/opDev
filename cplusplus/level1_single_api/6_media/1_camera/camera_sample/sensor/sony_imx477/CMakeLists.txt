# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)

### Sensor SONY IMX219 ###
# project information
project(sns_imx477)

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

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${LIB_PATH})

link_directories(
    ${LIB_PATH}
    ${LIB_PATH}/stub
)

add_library(sns_imx477 SHARED
        imx477_cmos.c
        imx477_sensor_ctl.c
)

include_directories(sns_imx477
        ${INC_PATH}/acl/media
        ../include
)

target_link_libraries(sns_imx477
        acl_isp_mpi
)

