/**
 * @file matmul_reduce_scatter_demo_def.h
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef MATMUL_REDUCE_SCATTER_DEMO_DEF_H
#define MATMUL_REDUCE_SCATTER_DEMO_DEF_H

constexpr uint32_t RANK_DIM = 8;
constexpr uint32_t RANK_M = 16384;
constexpr uint32_t RANK_K = 640;
constexpr uint32_t RANK_N = 5120;
constexpr bool IS_TRANS_A = false;
constexpr bool IS_TRANS_B = false;
constexpr int64_t COMM_TURN = 0;
constexpr char REDUCE_OP[] = "sum";

#endif  // MATMUL_REDUCE_SCATTER_DEMO_DEF_H
