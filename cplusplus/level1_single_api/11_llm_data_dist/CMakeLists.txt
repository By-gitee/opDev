cmake_minimum_required(VERSION 3.5.1)
project(llm_datadist_sample)

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

add_executable(prompt_sample "prompt_sample.cpp")

target_compile_options(prompt_sample PRIVATE
        ${common_compile_options}
)

target_compile_definitions(prompt_sample PRIVATE
        ${common_compile_definitions}
)

target_include_directories(prompt_sample PRIVATE
        ${INCLUDE_DIR}
        ${INCLUDE_DIR}/external/ge_common
)

target_link_directories(prompt_sample PRIVATE
        ${ASCEND_PATH}/lib64
)

target_link_libraries(prompt_sample PRIVATE
        llm_engine
        graph
)

add_executable(decoder_sample "decoder_sample.cpp")

target_compile_options(decoder_sample PRIVATE
        ${common_compile_options}
)

target_compile_definitions(decoder_sample PRIVATE
        ${common_compile_definitions}
)

target_include_directories(decoder_sample PRIVATE
        ${INCLUDE_DIR}
        ${INCLUDE_DIR}/external/ge_common
)

target_link_directories(decoder_sample PRIVATE
        ${ASCEND_PATH}/lib64
)

target_link_libraries(decoder_sample PRIVATE
        llm_engine
        graph
)

add_executable(prompt_sample2 "prompt_sample2.cpp")

target_compile_options(prompt_sample2 PRIVATE
        ${common_compile_options}
)

target_compile_definitions(prompt_sample2 PRIVATE
        ${common_compile_definitions}
)

target_include_directories(prompt_sample2 PRIVATE
        ${INCLUDE_DIR}
        ${INCLUDE_DIR}/external/ge_common
)

target_link_directories(prompt_sample2 PRIVATE
        ${ASCEND_PATH}/lib64
)

target_link_libraries(prompt_sample2 PRIVATE
        llm_datadist
        graph
        ascendcl
)

add_executable(decoder_sample2 "decoder_sample2.cpp")

target_compile_options(decoder_sample2 PRIVATE
        ${common_compile_options}
)

target_compile_definitions(decoder_sample2 PRIVATE
        ${common_compile_definitions}
)

target_include_directories(decoder_sample2 PRIVATE
        ${INCLUDE_DIR}
        ${INCLUDE_DIR}/external/ge_common
)

target_link_directories(decoder_sample2 PRIVATE
        ${ASCEND_PATH}/lib64
)

target_link_libraries(decoder_sample2 PRIVATE
        llm_datadist
        graph
        ascendcl
)

add_executable(prompt_sample3 "prompt_sample3.cpp")

target_compile_options(prompt_sample3 PRIVATE
        ${common_compile_options}
)

target_compile_definitions(prompt_sample3 PRIVATE
        ${common_compile_definitions}
)

target_include_directories(prompt_sample3 PRIVATE
        ${INCLUDE_DIR}
        ${INCLUDE_DIR}/external/ge_common
)

target_link_directories(prompt_sample3 PRIVATE
        ${ASCEND_PATH}/lib64
)

target_link_libraries(prompt_sample3 PRIVATE
        llm_datadist
        graph
        ascendcl
)

add_executable(decoder_sample3 "decoder_sample3.cpp")

target_compile_options(decoder_sample3 PRIVATE
        ${common_compile_options}
)

target_compile_definitions(decoder_sample3 PRIVATE
        ${common_compile_definitions}
)

target_include_directories(decoder_sample3 PRIVATE
        ${INCLUDE_DIR}
        ${INCLUDE_DIR}/external/ge_common
)

target_link_directories(decoder_sample3 PRIVATE
        ${ASCEND_PATH}/lib64
)

target_link_libraries(decoder_sample3 PRIVATE
        llm_datadist
        graph
        ascendcl
)

add_executable(prompt_sample4 "prompt_sample4.cpp")

target_compile_options(prompt_sample4 PRIVATE
        ${common_compile_options}
)

target_compile_definitions(prompt_sample4 PRIVATE
        ${common_compile_definitions}
)

target_include_directories(prompt_sample4 PRIVATE
        ${INCLUDE_DIR}
        ${INCLUDE_DIR}/external/ge_common
)

target_link_directories(prompt_sample4 PRIVATE
        ${ASCEND_PATH}/lib64
)

target_link_libraries(prompt_sample4 PRIVATE
        llm_datadist
        graph
        ascendcl
)

add_executable(decoder_sample4 "decoder_sample4.cpp")

target_compile_options(decoder_sample4 PRIVATE
        ${common_compile_options}
)

target_compile_definitions(decoder_sample4 PRIVATE
        ${common_compile_definitions}
)

target_include_directories(decoder_sample4 PRIVATE
        ${INCLUDE_DIR}
        ${INCLUDE_DIR}/external/ge_common
)

target_link_directories(decoder_sample4 PRIVATE
        ${ASCEND_PATH}/lib64
)

target_link_libraries(decoder_sample4 PRIVATE
        llm_datadist
        graph
        ascendcl
)