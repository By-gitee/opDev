# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)

# project information
project(Driver)

# Compile options
add_compile_options(-std=c++11)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "../output")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "../output")

set(CUST_PKG_ADD_PATH $ENV{DDK_PATH_ADD})
set(CUST_PKG_MATMUL_PATH $ENV{DDK_PATH_MATMUL})

set(LIB_PATH $ENV{NPU_HOST_LIB})
set(INC_PATH $ENV{NPU_HOST_INC})

# Dynamic libraries in the stub directory can only be used for compilation
if (NOT DEFINED ENV{NPU_HOST_LIB})
    string(TOLOWER "${CMAKE_SYSTEM_NAME}" SYSTEM_NAME_LOWER)
    set(LIB_PATH "/usr/local/Ascend/ascend-toolkit/latest/${CMAKE_SYSTEM_PROCESSOR}-${SYSTEM_NAME_LOWER}/lib64")
    message(STATUS "set default LIB_PATH: ${LIB_PATH}")
else ()
    message(STATUS "env LIB_PATH: ${LIB_PATH}")
endif()

# Header path
include_directories(
    ../inc
    ${INC_PATH}
)

# add host lib path
link_directories(
    ${LIB_PATH}
)

add_library(op_runner SHARED
    operator_desc.cpp
    op_runner.cpp
    common.cpp
)

find_package(add_custom REQUIRED
    PATHS ${CUST_PKG_ADD_PATH}
    NO_DEFAULT_PATH
)

find_package(matmul_custom REQUIRED
    PATHS ${CUST_PKG_MATMUL_PATH}
    NO_DEFAULT_PATH
)

target_link_libraries(op_runner PRIVATE
    ascendcl
    add_custom::static
    matmul_custom::static
    nnopbase
)

install(TARGETS op_runner DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
