if (parser_FOUND)
    message(STATUS "Package parser has been found.")
    return()
endif()

set(_cmake_targets_defined "")
set(_cmake_targets_not_defined "")
set(_cmake_expected_targets "")
foreach(_cmake_expected_target IN ITEMS fmk_parser fmk_onnx_parser parser_headers)
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
    NAMES parser/onnx_parser.h
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_FIND_ROOT_PATH)

find_library(fmk_parser_SHARED_LIBRARY
    NAMES libfmk_parser.so
    PATH_SUFFIXES lib64
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_FIND_ROOT_PATH)

find_library(fmk_onnx_parser_SHARED_LIBRARY
        NAMES libfmk_onnx_parser.so
        PATH_SUFFIXES lib64
        NO_CMAKE_SYSTEM_PATH
        NO_CMAKE_FIND_ROOT_PATH)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(parser
    FOUND_VAR
        parser_FOUND
    REQUIRED_VARS
        _INCLUDE_DIR
        fmk_parser_SHARED_LIBRARY
        fmk_onnx_parser_SHARED_LIBRARY
)

if(parser_FOUND)
    set(parser_INCLUDE_DIR "${_INCLUDE_DIR}")
    include(CMakePrintHelpers)
    message(STATUS "Variables in flow_func module:")
    cmake_print_variables(parser_INCLUDE_DIR)
    cmake_print_variables(fmk_parser_SHARED_LIBRARY)
    cmake_print_variables(fmk_onnx_parser_SHARED_LIBRARY)

    add_library(fmk_parser SHARED IMPORTED)
    set_target_properties(fmk_parser PROPERTIES
        INTERFACE_LINK_LIBRARIES "parser_headers"
        IMPORTED_LOCATION "${fmk_parser_SHARED_LIBRARY}"
    )

    add_library(fmk_onnx_parser SHARED IMPORTED)
    set_target_properties(fmk_onnx_parser PROPERTIES
        INTERFACE_LINK_LIBRARIES "parser_headers"
        IMPORTED_LOCATION "${fmk_onnx_parser_SHARED_LIBRARY}"
    )

    add_library(parser_headers INTERFACE IMPORTED)
    set_target_properties(parser_headers PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${parser_INCLUDE_DIR}"
    )

    include(CMakePrintHelpers)
    cmake_print_properties(TARGETS fmk_parser
        PROPERTIES INTERFACE_LINK_LIBRARIES IMPORTED_LOCATION
    )
    cmake_print_properties(TARGETS fmk_onnx_parser
        PROPERTIES INTERFACE_LINK_LIBRARIES IMPORTED_LOCATION
    )
    cmake_print_properties(TARGETS parser_headers
        PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
    )
endif()

# Cleanup temporary variables.
set(_INCLUDE_DIR)
