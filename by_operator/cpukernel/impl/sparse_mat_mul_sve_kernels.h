
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: api of SparseMatMulSVE
 */

#ifndef _SPARSE_MAT_MUL_SVE_KERNELS_H_
#define _SPARSE_MAT_MUL_SVE_KERNELS_H_

#include "cpu_kernel.h"

namespace aicpu {
class SparseMatMulSVECpuKernel : public CpuKernel {
public:
    ~SparseMatMulSVECpuKernel() = default;
    virtual uint32_t Compute(CpuKernelContext &ctx) override;
    template<typename T>
    uint32_t SparseMatMulCompute(CpuKernelContext &ctx);
    template<typename T>
    uint32_t SparseMatMulComputeWithBlock(CpuKernelContext &ctx,
                                                uint32_t blockX_id, uint32_t blockX_dim,
												uint32_t blockY_id, uint32_t blockY_dim,
												uint32_t block_size);
};
} // namespace aicpu
#endif
