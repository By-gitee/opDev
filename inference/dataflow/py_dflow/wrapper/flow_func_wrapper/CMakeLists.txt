cmake_minimum_required(VERSION 3.14)

project(flowfunc_wrapper)

add_library(flowfunc_wrapper MODULE flow_func_wrapper.cpp)

target_include_directories(flowfunc_wrapper PRIVATE
    ${HI_PYTHON_INC}
    ${pybind11_INCLUDE_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/../
)

target_link_libraries(flowfunc_wrapper PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    $<BUILD_INTERFACE:udf_headers>
    $<BUILD_INTERFACE:metadef_headers>
    flow_func
    # ${HI_PYTHON_LIB}
)

set_target_properties(flowfunc_wrapper
    PROPERTIES
    PREFIX ""
)

target_compile_definitions(flowfunc_wrapper PRIVATE
    PYBIND11_NO_ASSERT_GIL_HELD_INCREF_DECREF
)

target_compile_options(flowfunc_wrapper PRIVATE
    -O2
    -Xlinker -export-dynamic
)

target_link_options(flowfunc_wrapper PRIVATE
    $<$<CONFIG:Release>:-s>
)