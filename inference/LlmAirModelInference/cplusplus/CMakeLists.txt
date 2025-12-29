cmake_minimum_required(VERSION 3.5)
PROJECT(load_and_run)

if(DEFINED ENV{ASCEND_INSTALL_PATH})
    set(ASCEND_INSTALL_PATH $ENV{ASCEND_INSTALL_PATH})
    message(STATUS "Read ASCEND_INSTALL_PATH from environment variable")
else()
    set(ASCEND_INSTALL_PATH "/usr/local/Ascend/latest")
    message(STATUS "ASCEND_INSTALL_PATH is not set, use default path: ${ASCEND_INSTALL_PATH}")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/../out)

set(INC_DIR
    "${ASCEND_INSTALL_PATH}/include/"
    "${ASCEND_INSTALL_PATH}/opp/built-in/op_proto/inc/"
)

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

add_executable(load_and_run
    ../src/main.cpp
    ../src/llm_config.cpp
    ../src/utils.cpp
)
target_include_directories(load_and_run PRIVATE ${INC_DIR})
target_compile_options(load_and_run PRIVATE
    -O2
    -std=c++11
    -ftrapv
    -fstack-protector-all
    -fPIC
)

target_link_libraries(load_and_run PRIVATE
    -Wl,--whole-archive
    ${ASCEND_INSTALL_PATH}/lib64/libascendcl.so
    ${ASCEND_INSTALL_PATH}/lib64/libgraph.so
    ${ASCEND_INSTALL_PATH}/lib64/libge_runner.so
    ${ASCEND_INSTALL_PATH}/lib64/libfmk_parser.so
    ${ASCEND_INSTALL_PATH}/lib64/libgraph_base.so
    ${ASCEND_INSTALL_PATH}/lib64/libdavinci_executor.so
    ${ASCEND_INSTALL_PATH}/lib64/libgert.so
    -Wl,--no-whole-archive
)
