cmake_minimum_required(VERSION 3.14)

project(dataFlow_wrapper)

add_library(dflow_wrapper MODULE dflow_wrapper.cpp)

target_include_directories(dflow_wrapper PRIVATE
    ${HI_PYTHON_INC}
    ${pybind11_INCLUDE_DIR}
)

target_link_libraries(dflow_wrapper PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    $<BUILD_INTERFACE:slog_headers>
    $<BUILD_INTERFACE:udf_headers>
    $<BUILD_INTERFACE:metadef_headers>
    $<BUILD_INTERFACE:parser_headers>
    $<BUILD_INTERFACE:air_headers>
    alog
    flow_graph
    ge_runner
    fmk_parser
    fmk_onnx_parser
    # ${HI_PYTHON_LIB}
)

set_target_properties(dflow_wrapper
    PROPERTIES
    PREFIX ""
)

target_compile_definitions(dflow_wrapper PRIVATE
    PYBIND11_NO_ASSERT_GIL_HELD_INCREF_DECREF
)

target_compile_options(dflow_wrapper PRIVATE
    -O2
    -Xlinker -export-dynamic
)

target_link_options(dflow_wrapper PRIVATE
    $<$<CONFIG:Release>:-s>
)

project(data_wrapper)
add_library(data_wrapper MODULE data_wrapper.cpp)

target_include_directories(data_wrapper PRIVATE
    ${HI_PYTHON_INC}
    ${pybind11_INCLUDE_DIR}
)

target_compile_definitions(data_wrapper PRIVATE
    PYBIND11_NO_ASSERT_GIL_HELD_INCREF_DECREF
)

target_link_libraries(data_wrapper PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    $<BUILD_INTERFACE:udf_headers>
    $<BUILD_INTERFACE:metadef_headers>
)

set_target_properties(data_wrapper
    PROPERTIES
    PREFIX ""
)

target_compile_options(data_wrapper PRIVATE
    -O2
    -Xlinker -export-dynamic
)

target_link_options(data_wrapper PRIVATE
    $<$<CONFIG:Release>:-s>
)

add_subdirectory(flow_func_wrapper)
