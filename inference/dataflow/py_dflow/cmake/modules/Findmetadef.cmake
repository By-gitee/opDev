if (metadef_FOUND)
    message(STATUS "Package metadef has been found.")
    return()
endif()

set(_cmake_targets_defined "")
set(_cmake_targets_not_defined "")
set(_cmake_expected_targets "")
foreach(_cmake_expected_target IN ITEMS metadef_headers)
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
    NAMES graph/types.h
    NO_CMAKE_SYSTEM_PATH
    NO_CMAKE_FIND_ROOT_PATH)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(metadef
    FOUND_VAR
        metadef_FOUND
    REQUIRED_VARS
        _INCLUDE_DIR
)

if(metadef_FOUND)
    set(metadef_INCLUDE_DIR "${_INCLUDE_DIR}")
    include(CMakePrintHelpers)
    message(STATUS "Variables in metadef module:")
    cmake_print_variables(metadef_INCLUDE_DIR)

    add_library(metadef_headers INTERFACE IMPORTED)
    set_target_properties(metadef_headers PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${metadef_INCLUDE_DIR}"
    )

    include(CMakePrintHelpers)
    cmake_print_properties(TARGETS metadef_headers
        PROPERTIES INTERFACE_INCLUDE_DIRECTORIES
    )
endif()

# Cleanup temporary variables.
set(_INCLUDE_DIR)
