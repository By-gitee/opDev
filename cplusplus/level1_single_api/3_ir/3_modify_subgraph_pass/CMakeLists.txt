cmake_minimum_required(VERSION 3.5.1)
project(modify_subgraph_pass)

set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY TRUE)

if (DEFINED ENV{ASCEND_INSTALL_PATH})
    set(ASCEND_PATH $ENV{ASCEND_INSTALL_PATH})
else ()
    set(ASCEND_PATH /usr/local/Ascend/latest)
endif()

set(INCLUDE_DIR ${ASCEND_PATH}/include)

set(SRC_LIST_01
    src/add_abs_node.hpp
    src/modify_subgraph_pass_01.cpp
)

set(SRC_LIST_02
    src/add_abs_node.hpp
    src/modify_subgraph_pass_02.cpp
)

set(common_compile_options
    -std=c++17
    -g
    -Wall
)

set(common_compile_definitions
    _GLIBCXX_USE_CXX11_ABI=0
)

############ libmodify_subgraph_pass_01.so ############
set(LIB_01 ${PROJECT_NAME}_01)
add_library(${LIB_01} SHARED ${SRC_LIST_01})

set_target_properties(${LIB_01} PROPERTIES output_name ${LIB_01})

target_compile_options(${LIB_01} PRIVATE
    ${common_compile_options}
)

target_compile_definitions(${LIB_01} PRIVATE
    ${common_compile_definitions}
)

target_include_directories(${LIB_01} PRIVATE
    ${ASCEND_PATH}/opp/built-in/op_proto/inc
    ${INCLUDE_DIR}/graph
    ${INCLUDE_DIR}/ge
    ${INCLUDE_DIR}/register
    ${INCLUDE_DIR}
)

target_link_directories(${LIB_01} PRIVATE
    ${ASCEND_PATH}/compiler/lib64/stub
)

target_link_libraries(${LIB_01} PRIVATE
    -Wl,--no-as-needed
    graph
    register
    -Wl,--as-needed
)

############ libmodify_subgraph_pass_02.so ############
set(LIB_02 ${PROJECT_NAME}_02)
add_library(${LIB_02} SHARED ${SRC_LIST_02})

set_target_properties(${LIB_02} PROPERTIES output_name ${LIB_02})

target_compile_options(${LIB_02} PRIVATE
    ${common_compile_options}
)

target_compile_definitions(${LIB_02} PRIVATE
    ${common_compile_definitions}
)

target_include_directories(${LIB_02} PRIVATE
    ${ASCEND_PATH}/opp/built-in/op_proto/inc
    ${INCLUDE_DIR}/graph
    ${INCLUDE_DIR}/ge
    ${INCLUDE_DIR}/register
    ${INCLUDE_DIR}
)

target_link_directories(${LIB_02} PRIVATE
    ${ASCEND_PATH}/compiler/lib64/stub
)

target_link_libraries(${LIB_02} PRIVATE
    -Wl,--no-as-needed
    graph
    register
    -Wl,--as-needed
)
