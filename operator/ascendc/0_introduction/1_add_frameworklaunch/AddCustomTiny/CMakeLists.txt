cmake_minimum_required(VERSION 3.16.0)
project(opp)

set(ASCEND_COMPUTE_UNIT ascend910b)
find_package(ASC REQUIRED)

npu_op_package(${vendor_name}
    TYPE RUN
)

file(GLOB host_ops_srcs ${CMAKE_CURRENT_SOURCE_DIR}/add_custom_host.cpp)
npu_op_code_gen(
    SRC ${host_ops_srcs}
    PACKAGE ${vendor_name}
    OUT_DIR ${ASCEND_AUTOGEN_PATH}
    OPTIONS
    OPS_PRODUCT_NAME ${ASCEND_COMPUTE_UNIT}
)

file(GLOB autogen_aclnn_srcs ${ASCEND_AUTOGEN_PATH}/aclnn_*.cpp)
set_source_files_properties(${autogen_aclnn_srcs} PROPERTIES GENERATED TRUE)
npu_op_library(cust_opapi ACLNN
    ${autogen_aclnn_srcs}
)

npu_op_library(cust_optiling TILING
    ${host_ops_srcs}
)

npu_op_kernel_library(ascendc_kernels
    SRC_BASE ${CMAKE_SOURCE_DIR}/
    TILING_LIBRARY cust_optiling
)

npu_op_kernel_sources(ascendc_kernels
    OP_TYPE AddCustom
    KERNEL_FILE add_custom_kernel.cpp
)

npu_op_package_add(${vendor_name}
    LIBRARY
    cust_opapi
    cust_optiling
    ascendc_kernels
)
