# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)

# project information
project(acl_execute_static)

# Compile options
add_compile_options(-std=c++11)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "../output")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "../output")

set(CUST_PKG_PATH $ENV{BASIC_PATH})
set(OP_PKG_PATH_ADD $ENV{DDK_PATH_ADD})
set(OP_PKG_PATH_MATMUL $ENV{DDK_PATH_MATMUL})

set(LIB_PATH $ENV{NPU_HOST_LIB})
set(INC_PATH $ENV{NPU_HOST_INC})
set(COMPILE_MODE $ENV{COMPILE_MODE})
message("COMPILE_MODE: " ${COMPILE_MODE})

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
    ${INC_PATH}
    ${CUST_PKG_PATH}/include
)

# add host lib path
link_directories(
    ${LIB_PATH}
    ${CUST_PKG_PATH}/lib
)

add_executable(execute_static_op
    main.cpp
)

target_link_libraries(execute_static_op
    op_runner
    ascendcl
)

install(TARGETS execute_static_op DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
