
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
    // Get input tensors
    Tensor* matrix_shape = ctx.Input(kMatrixShapeInputIndex);
    Tensor* row_ptr = ctx.Input(kRowPtrInputIndex);
    Tensor* col_indices = ctx.Input(kColIndicesInputIndex);
    Tensor* values = ctx.Input(kValuesInputIndex);
    Tensor* weight = ctx.Input(kWeightInputIndex);

    Tensor* output = ctx.Output(kOutputIndex);

    // Check input & output address
    if (matrix_shape == nullptr || row_ptr == nullptr || col_indices == nullptr || 
        values == nullptr || weight == nullptr || output == nullptr) {
        CUST_KERNEL_LOG_ERROR(ctx, "Input or output tensor is nullptr.");
        return PARAM_INVALID;
    }

    // Get input tensor shapes
    auto matrixShape = matrix_shape->GetTensorShape();
    auto rowPtrShape = row_ptr->GetTensorShape();
    auto colIndicesShape = col_indices->GetTensorShape();
    auto valuesShape = values->GetTensorShape();
    auto weightShape = weight->GetTensorShape();
    auto outputShape = output->GetTensorShape();

    // Get matrix dimensions from matrix_shape tensor
    if (matrixShape->GetDimSize(0) < 2) {
        CUST_KERNEL_LOG_ERROR(ctx, "Matrix shape tensor should have at least 2 elements (M, K).");
        return PARAM_INVALID;
    }

    int32_t* matrix_shape_data = reinterpret_cast<int32_t*>(matrix_shape->GetData());
    int32_t M = matrix_shape_data[0];  // Number of rows in sparse matrix
    int32_t K = matrix_shape_data[1];  // Number of columns in sparse matrix

    // Get number of non-zero elements
    int32_t nnz = valuesShape->GetDimSize(0);
    int32_t weight_rows = weightShape->GetDimSize(0);
    int32_t weight_cols = weightShape->GetDimSize(1);

    // Validate dimensions
    if (rowPtrShape->GetDimSize(0) != M + 1) {
        CUST_KERNEL_LOG_ERROR(ctx, "Row pointer size should be M+1 (%d), but got %d", M + 1,
                              rowPtrShape->GetDimSize(0));
        return PARAM_INVALID;
    }

    if (colIndicesShape->GetDimSize(0) != nnz) {
        CUST_KERNEL_LOG_ERROR(ctx, "Column indices size should match values size (%d), but got %d",
                              nnz, colIndicesShape->GetDimSize(0));
        return PARAM_INVALID;
    }

    if (K != weight_cols) {
        CUST_KERNEL_LOG_ERROR(ctx, "Sparse matrix columns (%d) should match weight columns (%d)",
                              K, weight_cols);
        return PARAM_INVALID;
    }

    // Get input tensor DataType
    DataType valuesType = values->GetDataType();
    DataType weightType = weight->GetDataType();
    DataType outputType = output->GetDataType();

    if (valuesType != weightType || valuesType != outputType) {
        CUST_KERNEL_LOG_ERROR(ctx, "Data types of values, weight, and output should match.");
        return PARAM_INVALID;
    }

    // Dispatch based on data type
    switch (valuesType) {
        case DT_FLOAT16:
            return SparseMatMulTBaseCSRCompute<Eigen::half>(ctx);
        case DT_FLOAT:
            return SparseMatMulTBaseCSRCompute<float>(ctx);
        default:
            CUST_KERNEL_LOG_ERROR(ctx, "Unsupported data type: %d", static_cast<int>(valuesType));
            return PARAM_INVALID;
    }

    return SUCCESS;
}


template<typename T>
uint32_t SparseMatMulTBaseCSRCpuKernel::SparseMatMulTBaseCSRCompute(CpuKernelContext &ctx) {
  Tensor* matrix_shape = ctx.Input(kMatrixShapeInputIndex);
  Tensor* row_ptr      = ctx.Input(kRowPtrInputIndex);
  Tensor* col_indices  = ctx.Input(kColIndicesInputIndex);
  Tensor* values       = ctx.Input(kValuesInputIndex);
  Tensor* weight_kn    = ctx.Input(kWeightInputIndex);   // expect [K, N] row-major
  Tensor* output       = ctx.Output(kOutputIndex);

  if (!matrix_shape || !row_ptr || !col_indices || !values || !weight_kn || !output) {
    CUST_KERNEL_LOG_ERROR(ctx, "Null input/output tensor.");
    return PARAM_INVALID;
  }

  // ----------------------------
  // 1) Data pointers
  // ----------------------------
  auto* matrix_shape_data = reinterpret_cast<int32_t*>(matrix_shape->GetData());
  auto* row_ptr_data      = reinterpret_cast<int32_t*>(row_ptr->GetData());
  auto* col_indices_data  = reinterpret_cast<int32_t*>(col_indices->GetData());
  auto* values_data       = reinterpret_cast<float*>(values->GetData());
  auto* weight_data       = reinterpret_cast<float*>(weight_kn->GetData());
  auto* output_data       = reinterpret_cast<float*>(output->GetData());

  if (!matrix_shape_data || !row_ptr_data || !col_indices_data ||
      !values_data || !weight_data || !output_data) {
    CUST_KERNEL_LOG_ERROR(ctx, "Null tensor data pointer.");
    return PARAM_INVALID;
  }

  // ----------------------------
  // 2) Validate matrix_shape: >=2 int32
  // ----------------------------
  const size_t ms_elems = matrix_shape->GetDataSize() / sizeof(int32_t);
  if (ms_elems < 2) {
    CUST_KERNEL_LOG_ERROR(ctx, "matrix_shape must contain >=2 int32, got %zu.", ms_elems);
    return PARAM_INVALID;
  }

  const int32_t M = matrix_shape_data[0];
  const int32_t K = matrix_shape_data[1];

  // ----------------------------
  // 3) Validate row_ptr length == M+1 and monotonic
  // ----------------------------
  const size_t rp_elems = row_ptr->GetDataSize() / sizeof(int32_t);
  if (rp_elems != static_cast<size_t>(M) + 1) {
    CUST_KERNEL_LOG_ERROR(ctx, "row_ptr length mismatch: expect %d, got %zu.", M + 1, rp_elems);
    return PARAM_INVALID;
  }
  if (row_ptr_data[0] != 0) {
    CUST_KERNEL_LOG_ERROR(ctx, "row_ptr[0] must be 0, got %d.", row_ptr_data[0]);
    return PARAM_INVALID;
  }
  for (int32_t i = 0; i < M; ++i) {
    if (row_ptr_data[i] > row_ptr_data[i + 1]) {
      CUST_KERNEL_LOG_ERROR(ctx, "row_ptr must be non-decreasing: row_ptr[%d]=%d > row_ptr[%d]=%d.",
                            i, row_ptr_data[i], i + 1, row_ptr_data[i + 1]);
      return PARAM_INVALID;
    }
  }
  const int32_t nnz = row_ptr_data[M];
  if (nnz < 0) {
    CUST_KERNEL_LOG_ERROR(ctx, "nnz is negative: %d.", nnz);
    return PARAM_INVALID;
  }

  // ----------------------------
  // 4) Validate values/col_indices length == nnz
  // ----------------------------
  const size_t values_elems = values->GetDataSize() / sizeof(float);
  const size_t cols_elems   = col_indices->GetDataSize() / sizeof(int32_t);
//   if (static_cast<int64_t>(values_elems) != static_cast<int64_t>(nnz) ||
//       static_cast<int64_t>(cols_elems)   != static_cast<int64_t>(nnz)) {
//     CUST_KERNEL_LOG_ERROR(ctx,
//       "CSR nnz mismatch: nnz=%d, values_elems=%zu, col_indices_elems=%zu.",
//       nnz, values_elems, cols_elems);
//     return PARAM_INVALID;
//   }

  // ----------------------------
  // 5) Validate weight_kn shape: must be [K, N]
  // ----------------------------
  auto wshape = weight_kn->GetTensorShape();
  if (!wshape) {
    CUST_KERNEL_LOG_ERROR(ctx, "weight shape is null.");
    return PARAM_INVALID;
  }
  if (wshape->GetDims() != 2) {
    CUST_KERNEL_LOG_ERROR(ctx, "weight_kn must be 2D [K,N], got dims=%d.", wshape->GetDims());
    return PARAM_INVALID;
  }

  const int32_t wK = static_cast<int32_t>(wshape->GetDimSize(0));
  const int32_t N  = static_cast<int32_t>(wshape->GetDimSize(1));
  if (wK != K) {
    CUST_KERNEL_LOG_ERROR(ctx, "weight_kn dim0 must equal K: weightK=%d, K=%d.", wK, K);
    return PARAM_INVALID;
  }
  if (N <= 0) {
    CUST_KERNEL_LOG_ERROR(ctx, "Invalid N=%d from weight_kn shape.", N);
    return PARAM_INVALID;
  }

  const size_t weight_elems = weight_kn->GetDataSize() / sizeof(float);
  if (weight_elems != static_cast<size_t>(static_cast<int64_t>(K) * N)) {
    CUST_KERNEL_LOG_ERROR(ctx, "weight_kn buffer size mismatch: expect %lld elems, got %zu.",
                          static_cast<long long>(static_cast<int64_t>(K) * N), weight_elems);
    return PARAM_INVALID;
  }

  // ----------------------------
  // 6) Validate output capacity >= M*N
  // ----------------------------
  const int64_t out_elems_need = static_cast<int64_t>(M) * N;
  const size_t  out_elems_have = output->GetDataSize() / sizeof(float);
  if (static_cast<int64_t>(out_elems_have) < out_elems_need) {
    CUST_KERNEL_LOG_ERROR(ctx, "output too small: need %lld elems, got %zu.",
                          static_cast<long long>(out_elems_need), out_elems_have);
    return PARAM_INVALID;
  }

  // ----------------------------
  // 7) Zero init output
  // ----------------------------
  std::fill(output_data, output_data + out_elems_need, 0.0f);

  // ----------------------------
  // 8) Compute: C(i,n) += A(i,k) * weight_kn(k,n)
  // weight_kn row for fixed k is contiguous across n -> SIMD friendly.
  // ----------------------------
  for (int32_t i = 0; i < M; ++i) {
    const int32_t row_start = row_ptr_data[i];
    const int32_t row_end   = row_ptr_data[i + 1];

    // Per-row range sanity (row_ptr already monotonic, but still guard against corruption)
    if (row_start < 0 || row_end < 0 || row_start > row_end || row_end > nnz) {
      CUST_KERNEL_LOG_ERROR(ctx, "Invalid row_ptr range at row %d: start=%d, end=%d, nnz=%d.",
                            i, row_start, row_end, nnz);
      return PARAM_INVALID;
    }

    float* out_row = output_data + static_cast<int64_t>(i) * N;

    for (int32_t j = row_start; j < row_end; ++j) {
      const int32_t k_idx = col_indices_data[j];
      if (k_idx < 0 || k_idx >= K) {
        CUST_KERNEL_LOG_ERROR(ctx, "k index out of bounds: k=%d (valid [0,%d)) at nnz_id=%d. Skip.",
                             k_idx, K, j);
        continue;
      }

      const float a = values_data[j];
      if (a == 0.0f) continue;

      const float* w_row = weight_data + static_cast<int64_t>(k_idx) * N; // contiguous [N]

#ifdef __ARM_FEATURE_SVE
      int32_t n = 0;
      while (n < N) {
        svbool_t pg = svwhilelt_b32(n, N);

        svfloat32_t vout = svld1_f32(pg, out_row + n);
        svfloat32_t vw   = svld1_f32(pg, w_row   + n);

        // vout += vw * a
        vout = svmla_n_f32_x(pg, vout, vw, a);

        svst1_f32(pg, out_row + n, vout);

        n += static_cast<int32_t>(svcntw());
      }
#else
      for (int32_t n = 0; n < N; ++n) {
        out_row[n] += a * w_row[n];
      }
#endif
    }
  }

  return SUCCESS;
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
//     int32_t K = matrix_shape_data[1];  // Number of columns in sparse matrix
//     int32_t N = ctx.Input(kWeightInputIndex)->GetTensorShape()->GetDimSize(0);  // Number of rows in weight matrix

//     // Initialize output to zero
//     int64_t output_size = static_cast<int64_t>(M) * N;
//     for (int64_t i = 0; i < output_size; ++i) {
//         output_data[i] = static_cast<T>(0.0f);
//     }

//     // Perform CSR sparse matrix multiplication: output[M, N] = sparse_matrix[M, K] * transpose(weight[N, K])
//     // For each row in the sparse matrix
//     for (int32_t i = 0; i < M; ++i) {
//         int32_t row_start = row_ptr_data[i];
//         int32_t row_end = row_ptr_data[i + 1];

//         // For each non-zero element in the current row
//         for (int32_t j = row_start; j < row_end; ++j) {
//             // Validate index bounds
//             if (j < 0) continue; // Skip invalid indices

//             int32_t col_idx = col_indices_data[j];

//             // Validate column index bounds
//             if (col_idx < 0 || col_idx >= K) {
//                 CUST_KERNEL_LOG_WARN(ctx, "Column index %d out of bounds [0, %d) for element at position %d",
//                                    col_idx, K, j);
//                 continue;
//             }

//             T val = values_data[j];

//             // Multiply the non-zero value with the corresponding weight row
//             // and accumulate in the output
// #ifdef __ARM_FEATURE_SVE
//             // SVE optimized version for the innermost loop
//             if (std::is_same<T, float>::value) {
//                 // For each row in the weight matrix (which becomes column after transpose)
//                 for (int32_t n_iter = 0; n_iter < N; ++n_iter) {
//                     int64_t output_idx = static_cast<int64_t>(i) * N + n_iter;
//                     int64_t weight_idx = static_cast<int64_t>(n_iter) * K + col_idx;

//                     // Validate indices before access
//                     if (output_idx >= 0 && output_idx < output_size &&
//                         weight_idx >= 0 && weight_idx < static_cast<int64_t>(N) * K) {
//                         output_data[output_idx] += val * weight_data[weight_idx];
//                     }
//                 }
//             } else {
//                 // For other types, use the original implementation with bounds checking
//                 for (int32_t n_iter = 0; n_iter < N; ++n_iter) {
//                     int64_t output_idx = static_cast<int64_t>(i) * N + n_iter;
//                     int64_t weight_idx = static_cast<int64_t>(n_iter) * K + col_idx;

//                     // Validate indices before access
//                     if (output_idx >= 0 && output_idx < output_size &&
//                         weight_idx >= 0 && weight_idx < static_cast<int64_t>(N) * K) {
//                         output_data[output_idx] += val * weight_data[weight_idx];
//                     }
//                 }
//             }
// #else
//             // Original implementation without SVE - with bounds checking
//             for (int32_t n_iter = 0; n_iter < N; ++n_iter) {
//                 int64_t output_idx = static_cast<int64_t>(i) * N + n_iter;
//                 int64_t weight_idx = static_cast<int64_t>(n_iter) * K + col_idx;

//                 // Validate indices before access
//                 if (output_idx >= 0 && output_idx < output_size &&
//                     weight_idx >= 0 && weight_idx < static_cast<int64_t>(N) * K) {
//                     output_data[output_idx] += val * weight_data[weight_idx];
//                 }
//             }
// #endif
//         }
//     }

//     return SUCCESS;
}

REGISTER_CPU_KERNEL(SPARSE_MAT_MUL_T_BASE_CSR, SparseMatMulTBaseCSRCpuKernel);
} // namespace aicpu
