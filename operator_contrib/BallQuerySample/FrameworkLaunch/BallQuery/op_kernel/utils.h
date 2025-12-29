/* Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved. 
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.
 * You may not use this file except in compliance with the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef __UTILS_H_
#define __UTILS_H_
#include "kernel_operator.h"

constexpr int32_t BUFF_NUM = 2;                                     // tensor num for each queue
constexpr int MAX_REPEAT = 255;

constexpr uint32_t BYTE_PER_BLK = 32;
constexpr uint32_t BYTE_PER_BIG_BLK = 256;


constexpr uint32_t FP32_CNT_PER_BLK = 8;            // 32Bytes Per Block  32/4
constexpr uint32_t FP16_CNT_PER_BLK = 16;           // 32/2
constexpr uint32_t FP32_CNT_PER_BIG_BLK = 64;       // 256Bytes Per  Big Block  256/4
constexpr uint32_t FP16_CNT_PER_BIG_BLK = 128;      // 256/2

constexpr uint32_t INT32_CNT_PER_BLK = 8;           // 32Bytes Per Block  32/4
constexpr uint32_t INT16_CNT_PER_BLK = 16;          // 32/2
constexpr uint32_t INT8_CNT_PER_BLK = 32;           // 32/2
constexpr uint32_t INT32_CNT_PER_BIG_BLK = 64;      // 256Bytes Per  Big Block  256/4
constexpr uint32_t INT16_CNT_PER_BIG_BLK = 128;     // 256/2

constexpr half MAX_HALF = 65504;

constexpr uint32_t INT8_BIT = 8;
constexpr uint32_t INT16_BIT = 16;
constexpr uint32_t INT32_BIT = 32;
constexpr uint32_t INT64_BIT = 64;

constexpr uint32_t UINT8_BIT = 8;
constexpr uint32_t UINT16_BIT = 16;
constexpr uint32_t UINT32_BIT = 32;
constexpr uint32_t UINT64_BIT = 64;

constexpr uint32_t FLOAT16_BIT = 16;
constexpr uint32_t FLOAT32_BIT = 32;
constexpr uint32_t FLOAT64_BIT = 64;


template<typename T1, typename T2> 
__aicore__ inline T1 div_ceil(const T1 &a, const T2 &b){
    return (a+b-1) / b;
}

template<typename T1, typename T2> 
__aicore__ inline T1 ceil(const T1 &a, const T2 &b){
    return (a+b-1) / b * b;
}

template<typename T1, typename T2> 
__aicore__ inline T1 floor(const T1 &a, const T2 &b){
    return a / b * b;
}

#endif