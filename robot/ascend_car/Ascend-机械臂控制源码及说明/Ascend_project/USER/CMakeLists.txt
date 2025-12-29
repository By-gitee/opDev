cmake_minimum_required(VERSION 3.15)
project(USER C)

set(CMAKE_C_STANDARD 99)

include_directories(../CORE/)
include_directories(../STM32F10x_FWLib/inc)

file(GLOB src  "*.c" "*.h"
        "./algo/*.c ./algo/*.h")

add_library(USER ${src})