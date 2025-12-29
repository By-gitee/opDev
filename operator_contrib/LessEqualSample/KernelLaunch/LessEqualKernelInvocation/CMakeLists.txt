cmake_minimum_required(VERSION 3.16)
project(Ascend_c)

file(GLOB KERNEL_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/less_equal.cpp
)

if("${RUN_MODE}" STREQUAL "cpu")
    include(cmake/cpu_lib.cmake)
elseif("${RUN_MODE}" STREQUAL "sim" OR "${RUN_MODE}" STREQUAL "npu")
    include(cmake/npu_lib.cmake)
else()
    message("invalid RUN_MODE: ${RUN_MODE}")
endif()

add_executable(less_equal
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)

target_compile_options(less_equal PRIVATE
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:-g>>
    -O2
    -std=c++17
    -D_GLIBCXX_USE_CXX11_ABI=0
)

target_include_directories(less_equal PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:${ASCEND_CANN_PACKAGE_PATH}/include>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:${ASCEND_CANN_PACKAGE_PATH}/runtime/include>>
)

target_link_libraries(less_equal PRIVATE
    $<BUILD_INTERFACE:$<$<OR:$<STREQUAL:${RUN_MODE},npu>,$<STREQUAL:${RUN_MODE},sim>>:host_intf_pub>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:tikicpulib::${SOC_VERSION}>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:ascendcl>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:c_sec>>
    ascendc_kernels_${RUN_MODE}
    tiling_api
    register
    platform
    ascendalog
    dl
)

install(TARGETS less_equal
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)