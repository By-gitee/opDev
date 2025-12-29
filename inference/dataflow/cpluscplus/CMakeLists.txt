cmake_minimum_required(VERSION 3.5)
PROJECT(DataFlow)

if(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
    message(STATUS "Read ASCEND_HOME_PATH from environment variable")
else()
    set(ASCEND_HOME_PATH "/usr/local/Ascend/latest")
    message(STATUS "ASCEND_HOME_PATH is not set, use default path: ${ASCEND_HOME_PATH}")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/../output)

set(INC_DIR
    "${ASCEND_HOME_PATH}/include/"
    "${ASCEND_HOME_PATH}/opp/built-in/op_proto/inc/"
)

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

add_executable(sample1 sample1.cpp)
target_include_directories(sample1 PRIVATE ${INC_DIR})
target_compile_options(sample1 PRIVATE
    -O2
    -std=c++11
    -ftrapv
    -fstack-protector-all
    -fPIC
)
target_link_libraries(sample1 PRIVATE
    -Wl,--whole-archive
    ${ASCEND_HOME_PATH}/lib64/libgraph.so
    ${ASCEND_HOME_PATH}/lib64/libgraph_base.so
    ${ASCEND_HOME_PATH}/lib64/libflow_graph.so
    ${ASCEND_HOME_PATH}/lib64/libge_compiler.so
    ${ASCEND_HOME_PATH}/lib64/libdavinci_executor.so
    ${ASCEND_HOME_PATH}/lib64/libge_runner.so
    ${ASCEND_HOME_PATH}/lib64/libfmk_parser.so
    ${ASCEND_HOME_PATH}/lib64/libfmk_onnx_parser.so
    -Wl,--no-whole-archive
)

add_executable(sample2 sample2.cpp)
target_include_directories(sample2 PRIVATE ${INC_DIR})
target_compile_options(sample2 PRIVATE
    -O2
    -std=c++11
    -ftrapv
    -fstack-protector-all
    -fPIC
)
target_link_libraries(sample2 PRIVATE
    -Wl,--whole-archive
    ${ASCEND_HOME_PATH}/lib64/libgraph.so
    ${ASCEND_HOME_PATH}/lib64/libgraph_base.so
    ${ASCEND_HOME_PATH}/lib64/libflow_graph.so
    ${ASCEND_HOME_PATH}/lib64/libge_compiler.so
    ${ASCEND_HOME_PATH}/lib64/libdavinci_executor.so
    ${ASCEND_HOME_PATH}/lib64/libge_runner.so
    ${ASCEND_HOME_PATH}/lib64/libfmk_parser.so
    ${ASCEND_HOME_PATH}/lib64/libfmk_onnx_parser.so
    -Wl,--no-whole-archive
)

add_executable(sample3 sample3.cpp)
target_include_directories(sample3 PRIVATE ${INC_DIR})
target_compile_options(sample3 PRIVATE
    -O2
    -std=c++11
    -ftrapv
    -fstack-protector-all
    -fPIC
)
target_link_libraries(sample3 PRIVATE
    -Wl,--whole-archive
    ${ASCEND_HOME_PATH}/lib64/libgraph.so
    ${ASCEND_HOME_PATH}/lib64/libgraph_base.so
    ${ASCEND_HOME_PATH}/lib64/libflow_graph.so
    ${ASCEND_HOME_PATH}/lib64/libge_compiler.so
    ${ASCEND_HOME_PATH}/lib64/libdavinci_executor.so
    ${ASCEND_HOME_PATH}/lib64/libge_runner.so
    ${ASCEND_HOME_PATH}/lib64/libfmk_parser.so
    ${ASCEND_HOME_PATH}/lib64/libfmk_onnx_parser.so
    -Wl,--no-whole-archive
)

add_executable(sample4 sample4.cpp)
target_include_directories(sample4 PRIVATE ${INC_DIR})
target_compile_options(sample4 PRIVATE
    -O2
    -std=c++11
    -ftrapv
    -fstack-protector-all
    -fPIC
)
target_link_libraries(sample4 PRIVATE
    -Wl,--whole-archive
    ${ASCEND_HOME_PATH}/lib64/libgraph.so
    ${ASCEND_HOME_PATH}/lib64/libgraph_base.so
    ${ASCEND_HOME_PATH}/lib64/libflow_graph.so
    ${ASCEND_HOME_PATH}/lib64/libge_compiler.so
    ${ASCEND_HOME_PATH}/lib64/libdavinci_executor.so
    ${ASCEND_HOME_PATH}/lib64/libge_runner.so
    ${ASCEND_HOME_PATH}/lib64/libfmk_parser.so
    ${ASCEND_HOME_PATH}/lib64/libfmk_onnx_parser.so
    -Wl,--no-whole-archive
)

add_executable(sample5 sample5.cpp)
target_include_directories(sample5 PRIVATE ${INC_DIR})
target_compile_options(sample5 PRIVATE
    -O2
    -std=c++11
    -ftrapv
    -fstack-protector-all
    -fPIC
)
target_link_libraries(sample5 PRIVATE
    -Wl,--whole-archive
    ${ASCEND_HOME_PATH}/lib64/libgraph.so
    ${ASCEND_HOME_PATH}/lib64/libgraph_base.so
    ${ASCEND_HOME_PATH}/lib64/libflow_graph.so
    ${ASCEND_HOME_PATH}/lib64/libge_compiler.so
    ${ASCEND_HOME_PATH}/lib64/libdavinci_executor.so
    ${ASCEND_HOME_PATH}/lib64/libge_runner.so
    ${ASCEND_HOME_PATH}/lib64/libfmk_parser.so
    ${ASCEND_HOME_PATH}/lib64/libfmk_onnx_parser.so
    -Wl,--no-whole-archive
)

add_executable(sample6 sample6.cpp)
target_include_directories(sample6 PRIVATE ${INC_DIR})
target_compile_options(sample6 PRIVATE
    -O2
    -std=c++11
    -ftrapv
    -fstack-protector-all
    -fPIC
)
target_link_libraries(sample6 PRIVATE
    -Wl,--whole-archive
    ${ASCEND_HOME_PATH}/lib64/libgraph.so
    ${ASCEND_HOME_PATH}/lib64/libgraph_base.so
    ${ASCEND_HOME_PATH}/lib64/libflow_graph.so
    ${ASCEND_HOME_PATH}/lib64/libge_compiler.so
    ${ASCEND_HOME_PATH}/lib64/libdavinci_executor.so
    ${ASCEND_HOME_PATH}/lib64/libge_runner.so
    ${ASCEND_HOME_PATH}/lib64/libfmk_parser.so
    ${ASCEND_HOME_PATH}/lib64/libfmk_onnx_parser.so
    -Wl,--no-whole-archive
)

add_executable(test_perf test_perf.cpp)
target_include_directories(test_perf PRIVATE ${INC_DIR})
target_compile_options(test_perf PRIVATE
    -O2
    -std=c++11
    -ftrapv
    -fstack-protector-all
    -fPIC
)
target_link_libraries(test_perf PRIVATE
    -Wl,--whole-archive
    ${ASCEND_HOME_PATH}/lib64/libgraph.so
    ${ASCEND_HOME_PATH}/lib64/libgraph_base.so
    ${ASCEND_HOME_PATH}/lib64/libflow_graph.so
    ${ASCEND_HOME_PATH}/lib64/libge_compiler.so
    ${ASCEND_HOME_PATH}/lib64/libdavinci_executor.so
    ${ASCEND_HOME_PATH}/lib64/libge_runner.so
    ${ASCEND_HOME_PATH}/lib64/libfmk_parser.so
    ${ASCEND_HOME_PATH}/lib64/libfmk_onnx_parser.so
    -Wl,--no-whole-archive
)