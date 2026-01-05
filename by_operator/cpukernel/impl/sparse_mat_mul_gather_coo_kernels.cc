
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: implement of SparseMatMulGatherCOO
 */
#define __ARM_FEATURE_SVE

#include "sparse_mat_mul_gather_coo_kernels.h"
#include <type_traits>
#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif
#include "cpu_tensor.h"
#include "cpu_tensor_shape.h"
#include "cpu_types.h"
#include "cust_cpu_utils.h"

namespace  {
const char *SPARSE_MAT_MUL_GATHER_COO = "SparseMatMulGatherCOO";
const uint32_t kIndicesXInputIndex = 0;
const uint32_t kIndicesYInputIndex = 1;
const uint32_t kSparseMatrixInputIndex = 2;
const uint32_t kWeightInputIndex = 3;
const uint32_t kOutputIndex = 0;
const uint32_t SUCCESS = 0;
const uint32_t PARAM_INVAILD = 1;
const uint32_t ERROR = 2;
}

namespace aicpu  {
uint32_t SparseMatMulGatherCOOCpuKernel::Compute(CpuKernelContext &ctx)
{
    // Get input tensors
    Tensor* indicesX = ctx.Input(kIndicesXInputIndex);
    Tensor* indicesY = ctx.Input(kIndicesYInputIndex);
    Tensor* sparse_matrix = ctx.Input(kSparseMatrixInputIndex);
    Tensor* weight = ctx.Input(kWeightInputIndex);

    // Get output tensor
    Tensor* output = ctx.Output(kOutputIndex);

    // Check input & output addresses
    if (indicesX == nullptr || indicesY == nullptr || sparse_matrix == nullptr ||
        weight == nullptr || output == nullptr) {
        CUST_KERNEL_LOG_ERROR(ctx, "Input or output tensor is nullptr.");
        return PARAM_INVAILD;
    }

    // Get input tensor DataType
    DataType sparseType = sparse_matrix->GetDataType();
    DataType weightType = weight->GetDataType();

    if (sparseType != weightType) {
        CUST_KERNEL_LOG_ERROR(ctx, "DataType of sparse_matrix and weight does not match.");
        return PARAM_INVAILD;
    }

    // Dispatch based on data type
    switch (sparseType) {
        case DT_FLOAT16:
            return SparseMatMulGatherCOOCompute<Eigen::half>(ctx);
        case DT_FLOAT:
            return SparseMatMulGatherCOOCompute<float>(ctx);
        default:
            CUST_KERNEL_LOG_ERROR(ctx, "Unsupported data type: %d", static_cast<int>(sparseType));
            return PARAM_INVAILD;
    }

    return SUCCESS;
}

template<typename T>
uint32_t SparseMatMulGatherCOOCpuKernel::SparseMatMulGatherCOOCompute(CpuKernelContext &ctx)
{
    // Get input tensors
    Tensor* indicesX = ctx.Input(kIndicesXInputIndex);
    Tensor* indicesY = ctx.Input(kIndicesYInputIndex);
    Tensor* sparse_matrix = ctx.Input(kSparseMatrixInputIndex);
    Tensor* weight = ctx.Input(kWeightInputIndex);
    Tensor* output = ctx.Output(kOutputIndex);

    // Get data pointers
    int32_t* indicesX_data = reinterpret_cast<int32_t*>(indicesX->GetData());
    int32_t* indicesY_data = reinterpret_cast<int32_t*>(indicesY->GetData());
    T* sparse_data = reinterpret_cast<T*>(sparse_matrix->GetData());
    T* weight_data = reinterpret_cast<T*>(weight->GetData());
    T* output_data = reinterpret_cast<T*>(output->GetData());

    if (indicesX_data == nullptr || indicesY_data == nullptr ||
        sparse_data == nullptr || weight_data == nullptr || output_data == nullptr) {
        CUST_KERNEL_LOG_ERROR(ctx, "Failed to get tensor data.");
        return PARAM_INVAILD;
    }

    // Get tensor shapes
    auto indices_shape = indicesX->GetTensorShape();
    auto sparse_shape = sparse_matrix->GetTensorShape();
    auto weight_shape = weight->GetTensorShape();
    auto output_shape = output->GetTensorShape();

    // Get dimensions
    int64_t num_nonzeros = indices_shape->GetDimSize(0);  // Number of non-zero elements
    int64_t sparse_rows = sparse_shape->GetDimSize(0);
    int64_t sparse_cols = sparse_shape->GetDimSize(1);
    int64_t weight_rows = weight_shape->GetDimSize(0);
    int64_t weight_cols = weight_shape->GetDimSize(1);
    int64_t output_rows = output_shape->GetDimSize(0);
    int64_t output_cols = output_shape->GetDimSize(1);

    // Validate dimensions
    if (sparse_cols != weight_rows) {
        CUST_KERNEL_LOG_ERROR(ctx, "Dimension mismatch: sparse_cols (%ld) != weight_rows (%ld)",
                              sparse_cols, weight_rows);
        return PARAM_INVAILD;
    }
    if (sparse_rows != output_rows || weight_cols != output_cols) {
        CUST_KERNEL_LOG_ERROR(ctx, "Output dimension mismatch: expected [%ld, %ld], got [%ld, %ld]",
                              sparse_rows, weight_cols, output_rows, output_cols);
        return PARAM_INVAILD;
    }

    // Initialize output to zero
    int64_t output_size = output_rows * output_cols;
    for (int64_t i = 0; i < output_size; ++i) {
        output_data[i] = static_cast<T>(0);
    }

#ifdef __ARM_FEATURE_SVE
    // Use SVE optimized version with gather operations
    if (std::is_same<T, float>::value) {
        const float* sparse_f = reinterpret_cast<const float*>(sparse_data);
        const float* weight_f = reinterpret_cast<const float*>(weight_data);
        float* output_f = reinterpret_cast<float*>(output_data);

        // Process each non-zero element using its indices
        for (int64_t idx = 0; idx < num_nonzeros; ++idx) {
            int32_t row = indicesX_data[idx];
            int32_t col = indicesY_data[idx];

            if (row >= 0 && row < sparse_rows && col >= 0 && col < sparse_cols) {
                float sparse_val = sparse_f[row * sparse_cols + col];

                // Use SVE to compute sparse_val * weight[col, :]
                uint32_t n = 0;
                while (n < weight_cols) {
                    uint32_t vl = svcntw();         // vector length in #float32 lanes
                    uint32_t remaining = weight_cols - n;
                    uint32_t current_vl = (remaining < vl) ? remaining : vl;

                    svbool_t pg = svwhilelt_b32((uint32_t)0, (uint32_t)current_vl);

                    // Load weight values: weight[col, n:n+vl]
                    svfloat32_t vweight = svld1_f32(pg, &weight_f[col * weight_cols + n]);

                    // Multiply by sparse value
                    svfloat32_t vmul = svmul_f32_z(pg, svdup_f32(sparse_val), vweight);

                    // Add to output: output[row, n:n+vl] += result
                    svfloat32_t voutput = svld1_f32(pg, &output_f[row * output_cols + n]);
                    svfloat32_t vresult = svadd_f32_z(pg, voutput, vmul);
                    svst1_f32(pg, &output_f[row * output_cols + n], vresult);

                    n += current_vl;
                }
            }
        }
    } else if (std::is_same<T, Eigen::half>::value) {
        const Eigen::half* sparse_h = reinterpret_cast<const Eigen::half*>(sparse_data);
        const Eigen::half* weight_h = reinterpret_cast<const Eigen::half*>(weight_data);
        Eigen::half* output_h = reinterpret_cast<Eigen::half*>(output_data);

        // Process each non-zero element using its indices
        for (int64_t idx = 0; idx < num_nonzeros; ++idx) {
            int32_t row = indicesX_data[idx];
            int32_t col = indicesY_data[idx];

            if (row >= 0 && row < sparse_rows && col >= 0 && col < sparse_cols) {
                float sparse_val = static_cast<float>(sparse_h[row * sparse_cols + col]);

                // Use SVE to compute sparse_val * weight[col, :]
                uint32_t n = 0;
                while (n < weight_cols) {
                    uint32_t vl = svcnth();         // vector length in #float16 lanes
                    uint32_t remaining = weight_cols - n;
                    uint32_t current_vl = (remaining < vl) ? remaining : vl;

                    svbool_t pg = svwhilelt_b16((uint32_t)0, (uint32_t)current_vl);

                    // Load weight values: weight[col, n:n+vl]
                    const __fp16* weight_ptr = reinterpret_cast<const __fp16*>(&weight_h[col * weight_cols + n]);
                    svfloat16_t vweight = svld1_f16(pg, weight_ptr);

                    // Multiply by sparse value
                    svfloat16_t vmul = svmul_f16_z(pg, svdup_f16(static_cast<Eigen::half>(sparse_val)), vweight);

                    // Add to output: output[row, n:n+vl] += result
                    const __fp16* output_ptr = reinterpret_cast<const __fp16*>(&output_h[row * output_cols + n]);
                    svfloat16_t voutput = svld1_f16(pg, output_ptr);
                    svfloat16_t vresult = svadd_f16_z(pg, voutput, vmul);
                    svst1_f16(pg, reinterpret_cast<__fp16*>(&output_h[row * output_cols + n]), vresult);

                    n += current_vl;
                }
            }
        }
    } else {
        // Fallback implementation for other types
        for (int64_t idx = 0; idx < num_nonzeros; ++idx) {
            int32_t row = indicesX_data[idx];
            int32_t col = indicesY_data[idx];

            if (row >= 0 && row < sparse_rows && col >= 0 && col < sparse_cols) {
                T sparse_val = sparse_data[row * sparse_cols + col];

                // Compute sparse_val * weight[col, :]
                for (int64_t n = 0; n < weight_cols; ++n) {
                    output_data[row * output_cols + n] += sparse_val * weight_data[col * weight_cols + n];
                }
            }
        }
    }
#else
    // Fallback implementation without SVE
    for (int64_t idx = 0; idx < num_nonzeros; ++idx) {
        int32_t row = indicesX_data[idx];
        int32_t col = indicesY_data[idx];

        if (row >= 0 && row < sparse_rows && col >= 0 && col < sparse_cols) {
            T sparse_val = sparse_data[row * sparse_cols + col];

            // Compute sparse_val * weight[col, :]
            for (int64_t n = 0; n < weight_cols; ++n) {
                output_data[row * output_cols + n] += sparse_val * weight_data[col * weight_cols + n];
            }
        }
    }
#endif

    return SUCCESS;
}

REGISTER_CPU_KERNEL(SPARSE_MAT_MUL_GATHER_COO, SparseMatMulGatherCOOCpuKernel);
} // namespace aicpu
