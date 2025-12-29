cmake_minimum_required(VERSION 3.14)
project(PyDFlow)
set(BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

if (NOT DEFINED CMAKE_MODULE_PATH)
    set(CMAKE_MODULE_PATH
        ${CMAKE_CURRENT_LIST_DIR}/cmake/modules
    )
endif()

if (NOT DEFINED CMAKE_PREFIX_PATH)
    set(CMAKE_PREFIX_PATH
        ${ASCEND_INSTALL_PATH}
    )
endif()


include(CMakePrintHelpers)
message(STATUS "Variables in DFlow project:")
cmake_print_variables(ASCEND_INSTALL_PATH)
cmake_print_variables(CMAKE_SYSTEM_PROCESSOR)
cmake_print_variables(CMAKE_BUILD_TYPE)
cmake_print_variables(CMAKE_INSTALL_PREFIX)
cmake_print_variables(CMAKE_PREFIX_PATH)
cmake_print_variables(CMAKE_MODULE_PATH)
cmake_print_variables(BUILD_OPEN_PROJECT HI_PYTHON)

set(INSTALL_BASE_DIR "")
set(INSTALL_LIBRARY_DIR lib)
set(INSTALL_RUNTIME_DIR bin)
set(INSTALL_INCLUDE_DIR include)
set(INSTALL_CONFIG_DIR cmake)

include(CMakePackageConfigHelpers)

option(BUILD_OPEN_PROJECT "DFlow compile in opensource." FALSE)

if (BUILD_OPEN_PROJECT)
    include(cmake/test_funcs.cmake)
    include(cmake/intf_pub_linux.cmake)
    include(cmake/modules/Findair.cmake)
    include(cmake/modules/Finddflow.cmake)
    include(cmake/modules/Findmetadef.cmake)
    include(cmake/modules/Findparser.cmake)

    # 自研软件包
    find_package(slog MODULE REQUIRED)
    find_package(air MODULE REQUIRED)
    find_package(dflow MODULE REQUIRED)
    find_package(metadef MODULE REQUIRED)
    find_package(parser MODULE REQUIRED)
    find_package(udf MODULE REQUIRED)
endif()

include(cmake/python_depends.cmake)

add_custom_target(
    dataflow_python ALL
    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/dataflow-0.0.1-py3-none-any.whl
)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/dataflow-0.0.1-py3-none-any.whl
    COMMAND echo "package whl start"
    && mkdir -p ${CMAKE_CURRENT_BINARY_DIR}/wheel
    && cp -r ${BASE_DIR}/python/. ${CMAKE_CURRENT_BINARY_DIR}/wheel
    && cp -r ${CMAKE_CURRENT_BINARY_DIR}/wrapper/dflow_wrapper.so ${CMAKE_CURRENT_BINARY_DIR}/wheel/dataflow
    && cp -r ${CMAKE_CURRENT_BINARY_DIR}/wrapper/flow_func_wrapper/flowfunc_wrapper.so ${CMAKE_CURRENT_BINARY_DIR}/wheel/dataflow/flow_func
    && cp -r ${CMAKE_CURRENT_BINARY_DIR}/wrapper/data_wrapper.so ${CMAKE_CURRENT_BINARY_DIR}/wheel/dataflow
    && cd ${CMAKE_CURRENT_BINARY_DIR}/wheel
    && ${HI_PYTHON} setup.py bdist_wheel >/dev/null
    && cp -f dist/dataflow-0.0.1-py3-none-any.whl ${CMAKE_CURRENT_BINARY_DIR}/
    && echo "package whl end"
    && echo ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS dflow_wrapper flowfunc_wrapper data_wrapper
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/dataflow-0.0.1-py3-none-any.whl OPTIONAL
    DESTINATION ${INSTALL_LIBRARY_DIR}
)

# dataflow wrapper
add_subdirectory(wrapper)

if (NOT BUILD_OPEN_PROJECT)
    # test
    add_subdirectory(tests/ut)
    add_subdirectory(tests/st)
endif ()
