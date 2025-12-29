
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: api of SparseMatMulBaseB
 */

#ifndef _SPARSE_MAT_MUL_BASE_B_KERNELS_H_
#define _SPARSE_MAT_MUL_BASE_B_KERNELS_H_

#include "cpu_kernel.h"
#include "eigen3/Eigen/Dense"

namespace aicpu {
class SparseMatMulBaseBCpuKernel : public CpuKernel {
public:
    ~SparseMatMulBaseBCpuKernel() = default;
    virtual uint32_t Compute(CpuKernelContext &ctx) override;
    template<typename T>
      uint32_t SparseMatMulComputeB(CpuKernelContext &ctx);
	template<typename T>
	uint32_t SparseMatMulComputeWithBlockBaseB(CpuKernelContext &ctx,
                                                uint32_t blockX_id, uint32_t blockX_dim,
												uint32_t blockY_id, uint32_t blockY_dim,
												uint32_t block_size);
};
} // namespace aicpu
#endif
