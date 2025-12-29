PROJECT(func_add)
if (DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
    message(STATUS "Read ASCEND_HOME_PATH from ENV: ${ASCEND_HOME_PATH}")
else()
    set(ASCEND_HOME_PATH /usr/local/Ascend)
    message(STATUS "Default ASCEND_HOME_PATH=${ASCEND_HOME_PATH}. You can export ASCEND_HOME_PATH")
endif()

if ("x${RESOURCE_TYPE}" STREQUAL "xAscend")
    message(STATUS "Ascend compiler enter.")
    # if Unsupported current resource type, please uncomment the next line.
    message(FATAL_ERROR "Support compile Ascend target!")
elseif ("x${RESOURCE_TYPE}" STREQUAL "xAarch")
    message(STATUS "Aarch64 compiler enter.")
    set(LIB_FLOW_FUNC ${ASCEND_HOME_PATH}/latest/lib64/stub/aarch64/libflow_func.so)
    # if Unsupported current resource type, please uncomment the next line.
    # message(FATAL_ERROR "Unsupported compile aarch64 target!")
else ()
    message(STATUS "x86 compiler enter.")
    set(LIB_FLOW_FUNC ${ASCEND_HOME_PATH}/latest/lib64/stub/x86_64/libflow_func.so)
    # if Unsupported current resource type, please uncomment the next line.
    # message(FATAL_ERROR "Unsupported compile x86 target!")
endif ()

set(CMAKE_CXX_COMPILER ${TOOLCHAIN})

find_package(Python3 3.9 EXACT REQUIRED Interpreter COMPONENTS Development)

# set dynamic library output path
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${RELEASE_DIR})
# set static library output path
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${RELEASE_DIR})

message(STATUS "CMAKE_LIBRARY_OUTPUT_DIRECTORY=${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
message(STATUS "CMAKE_ARCHIVE_OUTPUT_DIRECTORY=${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}")

execute_process(COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(pybind11 CONFIG REQUIRED)

include_directories(
    ${ASCEND_HOME_PATH}/latest/include/flow_func
    ${pybind11_INCLUDE_DIRS}
)

#========================UDF so compile========================================
file(GLOB SRC_LIST "src_cpp/*.cpp")
# check if SRC_LIST is exist
if("x${SRC_LIST}" STREQUAL "x")
    message(UDF "=================no resource file")
    add_custom_target(${UDF_TARGET_LIB}
        COMMAND echo "no resource file to make lib${UDF_TARGET_LIB}.so")
    return(0)
endif()

# message(UDF "=================SRC_LIST=${SRC_LIST}=========================")
add_library(${UDF_TARGET_LIB} SHARED ${SRC_LIST})

target_compile_options(${UDF_TARGET_LIB} PRIVATE
    -O2
    -std=c++11
    -ftrapv
    -fstack-protector-all
    -fPIC
)

target_link_options(${UDF_TARGET_LIB} PRIVATE
    -Wl,-z,relro
    -Wl,-z,now
    -Wl,-z,noexecstack
    -s
)

target_link_libraries(${UDF_TARGET_LIB} PRIVATE
    -Wl,--whole-archive
    ${LIB_FLOW_FUNC}
    ${Python3_LIBRARIES}
    pybind11::embed
    -Wl,--no-whole-archive
)
file(COPY src_python/ DESTINATION ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})