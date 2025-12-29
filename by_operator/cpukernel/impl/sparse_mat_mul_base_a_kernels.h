
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: api of SparseMatMulBaseA
 */

#ifndef _SPARSE_MAT_MUL_BASE_A_KERNELS_H_
#define _SPARSE_MAT_MUL_BASE_A_KERNELS_H_

#include "cpu_kernel.h"
#include "eigen3/Eigen/Dense"

namespace aicpu {
class SparseMatMulBaseACpuKernel : public CpuKernel {
public:
    ~SparseMatMulBaseACpuKernel() = default;
    virtual uint32_t Compute(CpuKernelContext &ctx) override;
    template<typename T>
      uint32_t SparseMatMulComputeA(CpuKernelContext &ctx);
    template<typename T>
    uint32_t SparseMatMulComputeWithBlockBaseA(CpuKernelContext &ctx,
                                                uint32_t blockX_id, uint32_t blockX_dim,
												uint32_t blockY_id, uint32_t blockY_dim,
												uint32_t block_size);
};
} // namespace aicpu
#endif
