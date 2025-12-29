if (dflow_FOUND)
    message(STATUS "Package dflow has been found.")
    return()
endif()

set(_cmake_targets_defined "")
set(_cmake_targets_not_defined "")
set(_cmake_expected_targets "")
foreach(_cmake_expected_target IN ITEMS flow_graph dflow_headers)
    list(APPEND _cmake_expected_targets "${_cmake_expected_target}")
    if(TARGET "${_cmake_expected_target}")
        list(APPEND _cmake_targets_defined "${_cmake_expected_target}")
    else()
        list(APPEND _cmake_targets_not_defined "${_cmake_expected_target}")
    endif()
endforeach()
unset(_cmake_expected_target)

if(_cmake_targets_defined STREQUAL _cmake_expected_targets)
    unset(_cmake_targets_defined)
    unset(_cmake_targets_not_defined)
    unset(_cmake_expected_targets)
    unset(CMAKE_IMPORT_FILE_VERSION)
    cmake_policy(POP)
    return()
endif()

if(NOT _cmake_targets_defined STREQUAL "")
    string(REPLACE ";" ", " _cmake_targets_defined_text "${_cmake_targets_defined}")
    string(REPLACE ";" ", " _cmake_targets_not_defined_text "${_cmake_targets_not_defined}")
    message(FATAL_ERROR "Some (but not all) targets in this export set were already defined.\nTargets Defined: ${_cmake_targets_defined_text}\nTargets not yet defined: ${_cmake_targets_not_defined_text}\n")
endif()
unset(_cmake_targets_defined)
unset(_cmake_targets_not_defined)
unset(_cmake_expected_targets)

find_path(_INCLUDE_DIR
    NAMES flow_graph/data_flow.h
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_FIND_ROOT_PATH)

find_library(flow_graph_SHARED_LIBRARY
    NAMES libflow_graph.so
    PATH_SUFFIXES lib64
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_FIND_ROOT_PATH)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(dflow
    FOUND_VAR
        dflow_FOUND
    REQUIRED_VARS
        _INCLUDE_DIR
        flow_graph_SHARED_LIBRARY
)

if(dflow_FOUND)
    set(dflow_INCLUDE_DIR "${_INCLUDE_DIR}")
    include(CMakePrintHelpers)
    message(STATUS "Variables in dflow module:")
    cmake_print_variables(dflow_INCLUDE_DIR)
    cmake_print_variables(flow_graph_SHARED_LIBRARY)

    add_library(flow_graph SHARED IMPORTED)
    set_target_properties(flow_graph PROPERTIES
        INTERFACE_LINK_LIBRARIES "dflow_headers"
        IMPORTED_LOCATION "${flow_graph_SHARED_LIBRARY}"
    )

    add_library(dflow_headers INTERFACE IMPORTED)
    set_target_properties(dflow_headers PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${dflow_INCLUDE_DIR}"
    )

    include(CMakePrintHelpers)
    cmake_print_properties(TARGETS flow_graph
        PROPERTIES INTERFACE_LINK_LIBRARIES IMPORTED_LOCATION
    )
    cmake_print_properties(TARGETS dflow_headers
        PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
    )
endif()

# Cleanup temporary variables.
set(_INCLUDE_DIR)
