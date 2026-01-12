
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: implement of SparseMatMulTBaseCSR
 */
#include "sparse_mat_mul_t_base_csr_kernels.h"
#include <iostream>
#include <algorithm>
#include <type_traits>
#include "cpu_tensor.h"
#include "cpu_tensor_shape.h"
#include "cpu_types.h"
#include "cust_cpu_utils.h"

// 但是感觉CSR下直接根据SVE SIMD就足够优化的了。。。比如直接从CSR当中一个一个取出来，然后从weight矩阵中取出对应的行，用SVE做乘加累加操作。
// 如果说是base weight矩阵的话，也就是说每行只加载一遍，那么实际上CSC更有用？
// 因为压缩过的数据结构本身就已经减少了内存访问次数，剩下的就是如何高效使用weight矩阵的数据了。
// 尽量减少重复的出入...
// Include SVE headers for ARM SVE optimization
#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif
namespace  {
const char *SPARSE_MAT_MUL_T_BASE_CSR = "SparseMatMulTBaseCSR";
const uint32_t kMatrixShapeInputIndex = 0;
const uint32_t kRowPtrInputIndex = 1;
const uint32_t kColIndicesInputIndex = 2;
const uint32_t kValuesInputIndex = 3;
const uint32_t kWeightInputIndex = 4;
const uint32_t kOutputIndex = 0;
const uint32_t SUCCESS = 0;
const uint32_t PARAM_INVALID = 1;
const uint32_t ERROR = 2;
}

namespace aicpu  {
uint32_t SparseMatMulTBaseCSRCpuKernel::Compute(CpuKernelContext &ctx)
{
    // // Get input tensors
    // Tensor* matrix_shape = ctx.Input(kMatrixShapeInputIndex);
    // Tensor* row_ptr = ctx.Input(kRowPtrInputIndex);
    // Tensor* col_indices = ctx.Input(kColIndicesInputIndex);
    // Tensor* values = ctx.Input(kValuesInputIndex);
    // Tensor* weight = ctx.Input(kWeightInputIndex);

    // Tensor* output = ctx.Output(kOutputIndex);

    // // Check input & output address
    // if (matrix_shape == nullptr || row_ptr == nullptr || col_indices == nullptr || 
    //     values == nullptr || weight == nullptr || output == nullptr) {
    //     CUST_KERNEL_LOG_ERROR(ctx, "Input or output tensor is nullptr.");
    //     return PARAM_INVALID;
    // }

    // // Get input tensor shapes
    // auto matrixShape = matrix_shape->GetTensorShape();
    // auto rowPtrShape = row_ptr->GetTensorShape();
    // auto colIndicesShape = col_indices->GetTensorShape();
    // auto valuesShape = values->GetTensorShape();
    // auto weightShape = weight->GetTensorShape();
    // auto outputShape = output->GetTensorShape();

    // // Get matrix dimensions from matrix_shape tensor
    // if (matrixShape->GetDimSize(0) < 2) {
    //     CUST_KERNEL_LOG_ERROR(ctx, "Matrix shape tensor should have at least 2 elements (M, N).");
    //     return PARAM_INVALID;
    // }

    // int32_t* matrix_shape_data = reinterpret_cast<int32_t*>(matrix_shape->GetData());
    // int32_t M = matrix_shape_data[0];  // Number of rows in sparse matrix
    // int32_t N = matrix_shape_data[1];  // Number of columns in sparse matrix

    // // Get number of non-zero elements
    // int32_t nnz = valuesShape->GetDimSize(0);
    // int32_t weight_rows = weightShape->GetDimSize(0);
    // int32_t weight_cols = weightShape->GetDimSize(1);

    // // Validate dimensions
    // if (rowPtrShape->GetDimSize(0) != M + 1) {
    //     CUST_KERNEL_LOG_ERROR(ctx, "Row pointer size should be M+1 (%d), but got %d", M + 1, 
    //                           rowPtrShape->GetDimSize(0));
    //     return PARAM_INVALID;
    // }

    // if (colIndicesShape->GetDimSize(0) != nnz) {
    //     CUST_KERNEL_LOG_ERROR(ctx, "Column indices size should match values size (%d), but got %d", 
    //                           nnz, colIndicesShape->GetDimSize(0));
    //     return PARAM_INVALID;
    // }

    // if (N != weight_rows) {
    //     CUST_KERNEL_LOG_ERROR(ctx, "Sparse matrix columns (%d) should match weight rows (%d)", 
    //                           N, weight_rows);
    //     return PARAM_INVALID;
    // }

    // // Get input tensor DataType
    // DataType valuesType = values->GetDataType();
    // DataType weightType = weight->GetDataType();
    // DataType outputType = output->GetDataType();

    // if (valuesType != weightType || valuesType != outputType) {
    //     CUST_KERNEL_LOG_ERROR(ctx, "Data types of values, weight, and output should match.");
    //     return PARAM_INVALID;
    // }

    // // Dispatch based on data type
    // switch (valuesType) {
    //     case DT_FLOAT16:
    //         return SparseMatMulTBaseCSRCompute<Eigen::half>(ctx);
    //     case DT_FLOAT:
    //         return SparseMatMulTBaseCSRCompute<float>(ctx);
    //     default:
    //         CUST_KERNEL_LOG_ERROR(ctx, "Unsupported data type: %d", static_cast<int>(valuesType));
    //         return PARAM_INVALID;
    // }

    return SUCCESS;
}


template<typename T>
uint32_t SparseMatMulTBaseCSRCpuKernel::SparseMatMulTBaseCSRCompute(CpuKernelContext &ctx) {
//     // Get input tensors
//     Tensor* matrix_shape = ctx.Input(kMatrixShapeInputIndex);
//     Tensor* row_ptr = ctx.Input(kRowPtrInputIndex);
//     Tensor* col_indices = ctx.Input(kColIndicesInputIndex);
//     Tensor* values = ctx.Input(kValuesInputIndex);
//     Tensor* weight = ctx.Input(kWeightInputIndex);
//     Tensor* output = ctx.Output(kOutputIndex);

//     // Get data pointers
//     int32_t* matrix_shape_data = reinterpret_cast<int32_t*>(matrix_shape->GetData());
//     int32_t* row_ptr_data = reinterpret_cast<int32_t*>(row_ptr->GetData());
//     int32_t* col_indices_data = reinterpret_cast<int32_t*>(col_indices->GetData());
//     T* values_data = reinterpret_cast<T*>(values->GetData());
//     T* weight_data = reinterpret_cast<T*>(weight->GetData());
//     T* output_data = reinterpret_cast<T*>(output->GetData());

//     // Get matrix dimensions
//     int32_t M = matrix_shape_data[0];  // Number of rows in sparse matrix
//     int32_t N = matrix_shape_data[1];  // Number of columns in sparse matrix
//     int32_t K = ctx.Input(kWeightInputIndex)->GetTensorShape()->GetDimSize(1);  // Number of columns in weight matrix

//     // Initialize output to zero
//     int64_t output_size = static_cast<int64_t>(M) * K;
//     for (int64_t i = 0; i < output_size; ++i) {
//         output_data[i] = static_cast<T>(0.0f);
//     }

//     // Perform CSR sparse matrix multiplication: output = sparse_matrix * weight
//     // For each row in the sparse matrix
//     for (int32_t i = 0; i < M; ++i) {
//         int32_t row_start = row_ptr_data[i];
//         int32_t row_end = row_ptr_data[i + 1];

//         // For each non-zero element in the current row
//         for (int32_t j = row_start; j < row_end; ++j) {
//             int32_t col_idx = col_indices_data[j];
//             T val = values_data[j];

//             // Multiply the non-zero value with the corresponding weight row
//             // and accumulate in the output
// #ifdef __ARM_FEATURE_SVE
//             // SVE optimized version for the innermost loop
//             if (std::is_same<T, float>::value) {
//                 // For float type, use SVE vectorization
//                 const float* weight_row = reinterpret_cast<const float*>(&weight_data[static_cast<int64_t>(col_idx) * K]);
//                 float* output_row = reinterpret_cast<float*>(&output_data[static_cast<int64_t>(i) * K]);
//                 const float val_f = static_cast<float>(val);

//                 uint32_t k = 0;
//                 while (k < K) {
//                     uint32_t vl = svcntw();         // vector length in #float32 lanes
//                     uint32_t remaining = K - k;
//                     uint32_t current_vl = (remaining < vl) ? remaining : vl;

//                     // Create predicate for the current vector length
//                     svbool_t pg = svwhilelt_b32((uint32_t)k, (uint32_t)std::min((uint32_t)K, k + vl));

//                     // Load weight values
//                     svfloat32_t v_weight = svld1_f32(pg, &weight_row[k]);

//                     // Multiply by the scalar value
//                     svfloat32_t v_result = svmul_f32_x(pg, v_weight, val_f);

//                     // Add to the output (fused multiply-add equivalent)
//                     svfloat32_t v_output = svld1_f32(pg, &output_row[k]);
//                     svfloat32_t v_new_output = svadd_f32_x(pg, v_output, v_result);

//                     // Store back to output
//                     svst1_f32(pg, &output_row[k], v_new_output);

//                     k += current_vl;
//                 }
//             } else if (std::is_same<T, Eigen::half>::value) {
//                 // For half precision, use SVE vectorization
//                 const Eigen::half* weight_row = &weight_data[static_cast<int64_t>(col_idx) * K];
//                 Eigen::half* output_row = &output_data[static_cast<int64_t>(i) * K];
//                 const float val_f = static_cast<float>(val);

//                 uint32_t k = 0;
//                 while (k < K) {
//                     uint32_t vl = svcnth();         // vector length in #float16 lanes
//                     uint32_t remaining = K - k;
//                     uint32_t current_vl = (remaining < vl) ? remaining : vl;

//                     // Create predicate for the current vector length
//                     svbool_t pg = svwhilelt_b16((uint32_t)k, (uint32_t)std::min((uint32_t)K, k + vl));

//                     // Load weight values (convert Eigen::half to __fp16 for SVE operations)
//                     const __fp16* weight_ptr = reinterpret_cast<const __fp16*>(&weight_row[k]);
//                     svfloat16_t v_weight = svld1_f16(pg, weight_ptr);

//                     // Convert to float for computation
//                     svfloat32_t v_weight_f32 = svcvt_f32_f16_x(pg, v_weight);

//                     // Multiply by the scalar value
//                     svfloat32_t v_result = svmul_f32_x(pg, v_weight_f32, val_f);

//                     // Load current output values and convert to float
//                     const __fp16* output_ptr = reinterpret_cast<const __fp16*>(&output_row[k]);
//                     svfloat16_t v_output_f16 = svld1_f16(pg, output_ptr);
//                     svfloat32_t v_output_f32 = svcvt_f32_f16_x(pg, v_output_f16);

//                     // Add to the output
//                     svfloat32_t v_new_output_f32 = svadd_f32_x(pg, v_output_f32, v_result);

//                     // Convert back to float16 and store
//                     svfloat16_t v_new_output_f16 = svcvt_f16_f32_x(pg, v_new_output_f32);
//                     svst1_f16(pg, reinterpret_cast<__fp16*>(&output_row[k]), v_new_output_f16);

//                     k += current_vl;
//                 }
//             } else {
//                 // For other types, use the original implementation
//                 for (int32_t k = 0; k < K; ++k) {
//                     int64_t output_idx = static_cast<int64_t>(i) * K + k;
//                     int64_t weight_idx = static_cast<int64_t>(col_idx) * K + k;
//                     output_data[output_idx] += val * weight_data[weight_idx];
//                 }
//             }
// #else
//             // Original implementation without SVE
//             for (int32_t k = 0; k < K; ++k) {
//                 int64_t output_idx = static_cast<int64_t>(i) * K + k;
//                 int64_t weight_idx = static_cast<int64_t>(col_idx) * K + k;
//                 output_data[output_idx] += val * weight_data[weight_idx];
//             }
// #endif
//         }
//     }

    return SUCCESS;
}

REGISTER_CPU_KERNEL(SPARSE_MAT_MUL_T_BASE_CSR, SparseMatMulTBaseCSRCpuKernel);
} // namespace aicpu
