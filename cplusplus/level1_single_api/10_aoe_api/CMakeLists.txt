cmake_minimum_required(VERSION 3.5.1)
project(aoe_api_sample)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY TRUE)

if (DEFINED ENV{ASCEND_INSTALL_PATH})
    set(ASCEND_PATH $ENV{ASCEND_INSTALL_PATH})
else ()
    set(ASCEND_PATH /usr/local/Ascend/ascend-toolkit)
endif()

set(COMPILER_INCLUDE_DIR ${ASCEND_PATH}/latest/compiler/include)
set(AOE_INCLUDE_DIR ${ASCEND_PATH}/latest/tools/aoe/include)

set (common_compile_options
    --std=c++11
    -g
    -Wall
)

set(common_compile_definitions
    _GLIBCXX_USE_CXX11_ABI=0
)

add_executable(test_sample "main.cpp")

target_compile_options(test_sample PRIVATE
    ${common_compile_options}
)

target_compile_definitions(test_sample PRIVATE
    ${common_compile_definitions}
)

target_include_directories(test_sample PRIVATE
    ${COMPILER_INCLUDE_DIR}
    ${AOE_INCLUDE_DIR}
)

link_directories(${ASCEND_PATH}/latest/aarch64-linux/lib64/)

target_link_libraries(test_sample PRIVATE
    -Wl,--no-as-needed
    ${ASCEND_PATH}/latest/aarch64-linux/lib64/libaoe_tuning.so
    ${ASCEND_PATH}/latest/aarch64-linux/lib64/libgraph.so
    ${ASCEND_PATH}/latest/aarch64-linux/lib64/libfmk_parser.so 
)