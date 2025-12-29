cmake_minimum_required(VERSION 3.5.1)
project(adxl_sample)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY TRUE)

if (DEFINED ENV{ASCEND_INSTALL_PATH})
    set(ASCEND_PATH $ENV{ASCEND_INSTALL_PATH})
else ()
    set(ASCEND_PATH /usr/local/Ascend/latest)
endif ()

set(INCLUDE_DIR ${ASCEND_PATH}/include)

set(common_compile_options
        --std=c++11
        -g
        -Wall
)

set(common_compile_definitions
        _GLIBCXX_USE_CXX11_ABI=0
)

add_executable(adxl_engine_sample "adxl_engine_sample.cpp")

target_compile_options(adxl_engine_sample PRIVATE
        ${common_compile_options}
)

target_compile_definitions(adxl_engine_sample PRIVATE
        ${common_compile_definitions}
)

target_include_directories(adxl_engine_sample PRIVATE
        ${INCLUDE_DIR}
        ${INCLUDE_DIR}/external/ge_common
)

target_link_directories(adxl_engine_sample PRIVATE
        ${ASCEND_PATH}/lib64
)

target_link_libraries(adxl_engine_sample PRIVATE
        llm_datadist
        graph
        ascendcl
)

add_executable(adxl_engine_sample2 "adxl_engine_sample2.cpp")

target_compile_options(adxl_engine_sample2 PRIVATE
        ${common_compile_options}
)

target_compile_definitions(adxl_engine_sample2 PRIVATE
        ${common_compile_definitions}
)

target_include_directories(adxl_engine_sample2 PRIVATE
        ${INCLUDE_DIR}
        ${INCLUDE_DIR}/external/ge_common
)

target_link_directories(adxl_engine_sample2 PRIVATE
        ${ASCEND_PATH}/lib64
)

target_link_libraries(adxl_engine_sample2 PRIVATE
        llm_datadist
        graph
        ascendcl
)