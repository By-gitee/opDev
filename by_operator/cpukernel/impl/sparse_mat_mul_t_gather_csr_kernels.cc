
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: implement of SparseMatMulTGatherCSR
 */
#include "sparse_mat_mul_t_gather_csr_kernels.h"
#include <iostream>
#include <algorithm>
#include <type_traits>
#include "cpu_tensor.h"
#include "cpu_tensor_shape.h"
#include "cpu_types.h"
#include "cust_cpu_utils.h"

// Include SVE headers for ARM SVE optimization
#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif
namespace  {
const char *SPARSE_MAT_MUL_T_GATHER_CSR = "SparseMatMulTGatherCSR";
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
uint32_t SparseMatMulTGatherCSRCpuKernel::Compute(CpuKernelContext &ctx)
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
    // int32_t K = matrix_shape_data[1];  // Number of columns in sparse matrix

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

    // if (K != weight_rows) {
    //     CUST_KERNEL_LOG_ERROR(ctx, "Sparse matrix columns (%d) should match weight rows (%d)",
    //                           K, weight_rows);
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
uint32_t SparseMatMulTGatherCSRCpuKernel::SparseMatMulTGatherCSRCompute(CpuKernelContext &ctx) {
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
//     int32_t M = matrix_shape_data[0];  // Number of rows in sparse matrix (MxK)
//     int32_t K = matrix_shape_data[1];  // Number of columns in sparse matrix (MxK)
//     int32_t N = ctx.Input(kWeightInputIndex)->GetTensorShape()->GetDimSize(0);  // Number of rows in weight matrix (KxN)

//     // Initialize output to zero
//     int64_t output_size = static_cast<int64_t>(M) * N;
//     for (int64_t i = 0; i < output_size; ++i) {
//         output_data[i] = static_cast<T>(0.0f);
//     }

//     // Perform CSR sparse matrix multiplication: output = sparse_matrix * weight
//     // Load multiple values at once and perform gather operations for corresponding weight elements
// #ifdef __ARM_FEATURE_SVE
//     if (std::is_same<T, float>::value) {
//         // For float type, use SVE gather operations
//         const float* weight_base = reinterpret_cast<const float*>(weight_data);

//         // For each row in the sparse matrix
//         for (int32_t i = 0; i < M; ++i) {
//             int32_t row_start = row_ptr_data[i];
//             int32_t row_end = row_ptr_data[i + 1];

//             // Process multiple non-zero elements in the row simultaneously
//             int32_t j = row_start;
//             while (j < row_end) {
//                 // Determine how many values we can process in this iteration based on SVE vector length
//                 uint32_t vl = svcntw();  // Get the current vector length for float32
//                 uint32_t remaining_elements = row_end - j;
//                 uint32_t elements_to_process = std::min(static_cast<uint32_t>(remaining_elements), vl);

//                 // Create predicate for the number of elements we're actually processing
//                 svbool_t pg = svwhilelt_b32(j, j + elements_to_process);

//                 // Create vector of column indices for gather operations
//                 svuint32_t v_col_indices = svld1_u32(pg, reinterpret_cast<const uint32_t*>(&col_indices_data[j]));

//                 // Load multiple values at once
//                 svfloat32_t v_values = svld1_f32(pg, reinterpret_cast<const float*>(&values_data[j]));

//                 // For each of the N output elements in the row
//                 for (int32_t n = 0; n < N; n++) {
//                     // Calculate addresses for gather operation: weight[col_idx * N + n] for all col_idx in current batch
//                     svuint32_t v_weight_indices = svmul_n_u32_x(pg, v_col_indices, static_cast<uint32_t>(N));
//                     v_weight_indices = svadd_n_u32_x(pg, v_weight_indices, static_cast<uint32_t>(n));

//                     // Gather weight values from calculated addresses
//                     svfloat32_t v_weights = svld1_gather_s32index_f32(pg, weight_base, v_weight_indices);

//                     // Multiply weights by corresponding values
//                     svfloat32_t v_results = svmul_f32_x(pg, v_weights, v_values);

//                     // Sum up the results across all active elements in the vector
//                     float final_result = svaddv_f32(pg, v_results);

//                     // Add to output
//                     output_data[static_cast<int64_t>(i) * N + n] += final_result;
//                 }

//                 // Move to next batch of non-zero elements
//                 j += elements_to_process;
//             }
//         }
//     } else if (std::is_same<T, Eigen::half>::value) {
//         // For half precision, use SVE gather operations
//         const __fp16* weight_base = reinterpret_cast<const __fp16*>(weight_data);

//         // For each row in the sparse matrix
//         for (int32_t i = 0; i < M; ++i) {
//             int32_t row_start = row_ptr_data[i];
//             int32_t row_end = row_ptr_data[i + 1];

//             // Process multiple non-zero elements in the row simultaneously
//             int32_t j = row_start;
//             while (j < row_end) {
//                 // Determine how many values we can process in this iteration based on SVE vector length
//                 uint32_t vl = svcnth();  // Get the current vector length for float16
//                 uint32_t remaining_elements = row_end - j;
//                 uint32_t elements_to_process = std::min(static_cast<uint32_t>(remaining_elements), vl);

//                 // Create predicate for the number of elements we're actually processing
//                 svbool_t pg = svwhilelt_b16(j, j + elements_to_process);

//                 // Create vector of column indices for gather operations
//                 svuint32_t v_col_indices = svld1_u32(pg, reinterpret_cast<const uint32_t*>(&col_indices_data[j]));

//                 // Load multiple values at once
//                 svfloat16_t v_values = svld1_f16(pg, reinterpret_cast<const __fp16*>(&values_data[j]));

//                 // For each of the N output elements in the row
//                 for (int32_t n = 0; n < N; n++) {
//                     // Calculate addresses for gather operation: weight[col_idx * N + n] for all col_idx in current batch
//                     svuint32_t v_weight_indices = svmul_n_u32_x(pg, v_col_indices, static_cast<uint32_t>(N));
//                     v_weight_indices = svadd_n_u32_x(pg, v_weight_indices, static_cast<uint32_t>(n));

//                     // Gather weight values from calculated addresses
//                     svfloat16_t v_weights = svld1uh_gather_s32index_f16(pg, weight_base, v_weight_indices);

//                     // Multiply weights by corresponding values
//                     svfloat16_t v_results = svmul_f16_x(pg, v_weights, v_values);

//                     // Sum up the results across all active elements in the vector
//                     __fp16 final_result = svaddv_f16(pg, v_results);

//                     // Add to output
//                     output_data[static_cast<int64_t>(i) * N + n] += static_cast<T>(final_result);
//                 }

//                 // Move to next batch of non-zero elements
//                 j += elements_to_process;
//             }
//         }
//     } else {
//         // For other types, use the original implementation
//         for (int32_t i = 0; i < M; ++i) {
//             int32_t row_start = row_ptr_data[i];
//             int32_t row_end = row_ptr_data[i + 1];

//             for (int32_t j = row_start; j < row_end; ++j) {
//                 int32_t col_idx = col_indices_data[j];
//                 T val = values_data[j];

//                 for (int32_t n = 0; n < N; ++n) {
//                     int64_t output_idx = static_cast<int64_t>(i) * N + n;
//                     int64_t weight_idx = static_cast<int64_t>(col_idx) * N + n;
//                     output_data[output_idx] += val * weight_data[weight_idx];
//                 }
//             }
//         }
//     }
// #else
//     // Original implementation without SVE
//     for (int32_t i = 0; i < M; ++i) {
//         int32_t row_start = row_ptr_data[i];
//         int32_t row_end = row_ptr_data[i + 1];

//         for (int32_t j = row_start; j < row_end; ++j) {
//             int32_t col_idx = col_indices_data[j];
//             T val = values_data[j];

//             for (int32_t n = 0; n < N; ++n) {
//                 int64_t output_idx = static_cast<int64_t>(i) * N + n;
//                 int64_t weight_idx = static_cast<int64_t>(col_idx) * N + n;
//                 output_data[output_idx] += val * weight_data[weight_idx];
//             }
//         }
//     }
// #endif

    return SUCCESS;
}

REGISTER_CPU_KERNEL(SPARSE_MAT_MUL_T_GATHER_CSR, SparseMatMulTGatherCSRCpuKernel);
} // namespace aicpu
