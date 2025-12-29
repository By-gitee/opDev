# CMake lowest version requirement
cmake_minimum_required(VERSION 3.5.1)

# project information
project(hdmi_sample)

set(LOCAL_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(LIB_PATH $ENV{NPU_HOST_LIB})
if (NOT DEFINED ENV{NPU_HOST_LIB})
    set(LIB_PATH /usr/local/Ascend/CANN-7.0/runtime/lib64/)
    message(STATUS "set default LIB_PATH: ${LIB_PATH}")
else()
    message(STATUS "env LIB_PATH: ${LIB_PATH}")
endif()

set(INC_PATH $ENV{NPU_HOST_INC})
if (NOT DEFINED ENV{NPU_HOST_INC})
    set(INC_PATH /usr/local/Ascend/CANN-7.0/runtime/include/)
    message(STATUS "set default INC_PATH: ${INC_PATH}")
else()
    message(STATUS "env INC_PATH: ${INC_PATH}")
endif()

link_directories(
    ${LIB_PATH}/stub/aarch64/
    ${LIB_PATH}
)

add_definitions(-DTHREADED)

add_executable(sample_hdmi
    sample_hdmi.c
    hi_mpi_vo.c
)

include_directories(sample_hdmi
    ${LOCAL_DIR}/include
    ${INC_PATH}/acl/media
)

target_link_libraries(sample_hdmi
    acl_vo_mpi
    acl_hdmi_mpi
    acl_dvpp_mpi
)