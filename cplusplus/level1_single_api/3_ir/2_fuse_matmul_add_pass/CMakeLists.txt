cmake_minimum_required(VERSION 3.5.1)
project(fuse_matmul_add_pass)

set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY TRUE)

if (DEFINED ENV{ASCEND_INSTALL_PATH})
    set(ASCEND_PATH $ENV{ASCEND_INSTALL_PATH})
else ()
    set(ASCEND_PATH /usr/local/Ascend/latest)
endif()

set(INCLUDE_DIR ${ASCEND_PATH}/include)

aux_source_directory(${PROJECT_SOURCE_DIR}/src SRC_LIST)

set(common_compile_options
    -std=c++17
    -g
    -Wall
)

set(common_compile_definitions
    _GLIBCXX_USE_CXX11_ABI=0
)

############ libfuse_matmul_add_pass.so ############
add_library(${PROJECT_NAME} SHARED ${SRC_LIST})

set_target_properties(${PROJECT_NAME} PROPERTIES output_name ${PROJECT_NAME})

target_compile_options(${PROJECT_NAME} PRIVATE
    ${common_compile_options}
)

target_compile_definitions(${PROJECT_NAME} PRIVATE
    ${common_compile_definitions}
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${ASCEND_PATH}/opp/built-in/op_proto/inc
    ${INCLUDE_DIR}/graph
    ${INCLUDE_DIR}/ge
    ${INCLUDE_DIR}/register
    ${INCLUDE_DIR}
)

target_link_directories(${PROJECT_NAME} PRIVATE
    ${ASCEND_PATH}/compiler/lib64/stub
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    -Wl,--no-as-needed
    graph
    register
    -Wl,--as-needed
)
