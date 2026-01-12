
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: api of SparseMatMulTGatherCSR
 */

#ifndef _SPARSE_MAT_MUL_T_GATHER_CSR_KERNELS_H_
#define _SPARSE_MAT_MUL_T_GATHER_CSR_KERNELS_H_

#include "cpu_kernel.h"
#include "eigen3/Eigen/Dense"

namespace aicpu {
class SparseMatMulTGatherCSRCpuKernel : public CpuKernel {
public:
    ~SparseMatMulTGatherCSRCpuKernel() = default;
    virtual uint32_t Compute(CpuKernelContext &ctx) override;

private:
    template<typename T>
    uint32_t SparseMatMulTGatherCSRCompute(CpuKernelContext &ctx);
};
} // namespace aicpu
#endif
