
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: api of DenseToCSR
 */

#ifndef _DENSE_TO_CSR_KERNELS_H_
#define _DENSE_TO_CSR_KERNELS_H_

#include "cpu_kernel.h"
#include "eigen3/Eigen/Dense"

namespace aicpu {
class DenseToCSRCpuKernel : public CpuKernel {
public:
    ~DenseToCSRCpuKernel() = default;
    virtual uint32_t Compute(CpuKernelContext &ctx) override;

private:
    template<typename T>
    uint32_t DenseToCSRCompute(CpuKernelContext &ctx);
};
} // namespace aicpu
#endif
