# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)

# project information
project(nvr_demo)

# Compile options
add_compile_options(-std=c++11)

set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")
set(CMAKE_SKIP_BUILD_RPATH TRUE)

set(INC_PATH $ENV{DDK_PATH})
if (NOT DEFINED ENV{DDK_PATH})
    set(INC_PATH "/usr/local/Ascend/latest")
    message(STATUS "set default INC_PATH: ${INC_PATH}")
else ()
    message(STATUS "env INC_PATH: ${INC_PATH}")
endif()

set(LIB_PATH $ENV{NPU_HOST_LIB})
# Dynamic libraries in the stub directory can only be used for compilation
if (NOT DEFINED ENV{NPU_HOST_LIB})
    set(LIB_PATH "/usr/local/Ascend/latest/runtime/lib64/stub/")
    message(STATUS "set default LIB_PATH: ${LIB_PATH}")
else ()
    message(STATUS "env LIB_PATH: ${LIB_PATH}")
endif()

link_directories(
    ${LIB_PATH}
)

include_directories(
    ${INC_PATH}/runtime/include/acl/
    ${INC_PATH}/runtime/include/acl/dvpp/
    ${INC_PATH}/runtime/include/acl/media/
    ./include/
    ./src/region_param/
)

add_library(acl_vpss_mpi SHARED
    ./stub/mpi_vpss_stub.cpp
)

add_library(acl_vo_mpi SHARED
    ./stub/mpi_vo_stub.cpp
)

add_library(acl_hdmi_mpi SHARED
    ./stub/mpi_hdmi_stub.cpp
)

add_executable (nvr_demo
    ./src/sample_nvr.cpp
    ./src/sample_comm_region.cpp
    ./src/sample_comm_sys.cpp
    ./src/sample_comm_vdec.cpp
    ./src/sample_comm_vo.cpp
    ./src/sample_comm_vpss.cpp
    ./src/region_param/region_param_parser_ext.cpp
    ./src/region_param/region_param_parser.cpp
)

target_link_libraries(nvr_demo
    acl_dvpp_mpi
    acl_vpss_mpi
    acl_vo_mpi
    acl_hdmi_mpi
    ascendcl
    pthread
    stdc++
)