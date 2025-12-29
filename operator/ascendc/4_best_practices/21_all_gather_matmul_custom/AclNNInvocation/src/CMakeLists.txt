# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.16.1)

# project information
project(acl_execute_allgather_matmul_custom)

# Compile options
add_compile_options(-std=c++17)
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "../output")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "../output")

set(INC_PATH $ENV{DDK_PATH})

if (NOT DEFINED ENV{DDK_PATH})
    set(INC_PATH "/usr/local/Ascend/ascend-toolkit/latest")
    message(STATUS "set default INC_PATH: ${INC_PATH}")
else ()
    message(STATUS "env INC_PATH: ${INC_PATH}")
endif()

set(CUST_PKG_PATH "${INC_PATH}/opp/vendors/customize/op_api")

set(LIB_PATH $ENV{NPU_HOST_LIB})

# Dynamic libraries in the stub directory can only be used for compilation
if (NOT DEFINED ENV{NPU_HOST_LIB})
    set(LIB_PATH "/usr/local/Ascend/ascend-toolkit/latest/acllib/lib64/stub/")
    set(LIB_PATH1 "/usr/local/Ascend/ascend-toolkit/latest/atc/lib64/stub/")
    message(STATUS "set default LIB_PATH: ${LIB_PATH}")
else ()
    message(STATUS "env LIB_PATH: ${LIB_PATH}")
endif()

# Header path
include_directories(
    ../inc
    ${INC_PATH}/include
    ${INC_PATH}/include/aclnn
    ${INC_PATH}/runtime/include
    ${INC_PATH}/atc/include
    ${CUST_PKG_PATH}/include
)

# add host lib path
link_directories(
    ${LIB_PATH}
    ${LIB_PATH1}
    ${CUST_PKG_PATH}/lib
)

add_executable(execute_allgather_matmul_op
    common.cpp
    operator_desc.cpp
    op_runner.cpp
    main.cpp
)

target_link_libraries(execute_allgather_matmul_op
    ascendalog
    ascendcl
    acl_op_compiler
    cust_opapi
    hccl
    nnopbase
    graph
    stdc++
    pthread
)

install(TARGETS execute_allgather_matmul_op DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
