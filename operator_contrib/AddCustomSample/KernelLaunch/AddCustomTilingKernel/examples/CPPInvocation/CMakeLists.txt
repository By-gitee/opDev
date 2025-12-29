cmake_minimum_required(VERSION 3.16)
project(Ascend_c)

set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest"
    CACHE STRING "ASCEND CANN package installation directory"
)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--copy-dt-needed-entries")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type Release/Debug (default Debug)" FORCE)
endif()
if(CMAKE_INSTALL_PREFIX STREQUAL /usr/local)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/out" CACHE STRING "path for install()" FORCE)
endif()
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
add_executable(ascendc_kernels_bbit ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)

target_compile_options(ascendc_kernels_bbit PRIVATE
    -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall -Werror
)
target_link_directories(ascendc_kernels_bbit PRIVATE
  ${ASCEND_CANN_PACKAGE_PATH}/lib64
)
target_include_directories(ascendc_kernels_bbit PRIVATE
  ${ASCEND_CANN_PACKAGE_PATH}/include
)
target_link_libraries(ascendc_kernels_bbit PRIVATE
    ascendcl
    kernels
)

install(TARGETS ascendc_kernels_bbit
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
