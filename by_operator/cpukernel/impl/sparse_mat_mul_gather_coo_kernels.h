
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: api of SparseMatMulGatherCOO
 */

#ifndef _SPARSE_MAT_MUL_GATHER_COO_KERNELS_H_
#define _SPARSE_MAT_MUL_GATHER_COO_KERNELS_H_

#include "cpu_kernel.h"
#include "eigen3/Eigen/Dense"

namespace aicpu {
class SparseMatMulGatherCOOCpuKernel : public CpuKernel {
public:
    ~SparseMatMulGatherCOOCpuKernel() = default;
    virtual uint32_t Compute(CpuKernelContext &ctx) override;

private:
    template<typename T>
    uint32_t SparseMatMulGatherCOOCompute(CpuKernelContext &ctx);
};
} // namespace aicpu
#endif
