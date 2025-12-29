cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)

execute_process(COMMAND python3 -c "import os; import torch; print(os.path.dirname(torch.__file__))"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE TORCH_PATH
)
message("TORCH_PATH is ${TORCH_PATH}")

execute_process(COMMAND python3 -c "import os; import torch_npu; print(os.path.dirname(torch_npu.__file__))"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE TORCH_NPU_PATH
)
message("TORCH_NPU_PATH is ${TORCH_NPU_PATH}")

execute_process(COMMAND python3 -m pybind11 --includes
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE PYBIND11_INC
)
string(REPLACE " " ";" PYBIND11_INC ${PYBIND11_INC})

execute_process(COMMAND python3-config --extension-suffix
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE PYBIND11_SUFFIX
)

project(kernel_samples LANGUAGES ASC CXX)

add_library(pybind11_lib SHARED
    add_custom.asc
)

target_link_libraries(pybind11_lib PRIVATE
    torch_npu
)

target_link_directories(pybind11_lib PRIVATE
    ${TORCH_PATH}/lib
    ${TORCH_NPU_PATH}/lib
)

target_include_directories(pybind11_lib PRIVATE
    ${TORCH_NPU_PATH}/include
    ${TORCH_PATH}/include
    ${TORCH_PATH}/include/torch/csrc/api/include
)

target_compile_definitions(pybind11_lib PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
)

target_compile_options(pybind11_lib PRIVATE
    ${PYBIND11_INC}
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
    -fPIC
)

set_target_properties(pybind11_lib PROPERTIES
    OUTPUT_NAME add_custom${PYBIND11_SUFFIX}
    PREFIX "" SUFFIX ""
)