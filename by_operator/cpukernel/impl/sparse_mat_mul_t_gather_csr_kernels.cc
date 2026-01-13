
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
        CUST_KERNEL_LOG_ERROR(ctx, "Matrix shape tensor should have at least 2 elements (M, N).");
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

    if (K != weight_rows) {
        CUST_KERNEL_LOG_ERROR(ctx, "Sparse matrix columns (%d) should match weight rows (%d)",
                              K, weight_rows);
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
        case DT_FLOAT:
            return SparseMatMulTGatherCSRCompute<float>(ctx);
        default:
            CUST_KERNEL_LOG_ERROR(ctx, "Unsupported data type: %d", static_cast<int>(valuesType));
            return PARAM_INVALID;
    }

    return SUCCESS;
}
template<typename T>
uint32_t SparseMatMulTGatherCSRCpuKernel::SparseMatMulTGatherCSRCompute(CpuKernelContext &ctx) {
Tensor* matrix_shape = ctx.Input(kMatrixShapeInputIndex);
  Tensor* row_ptr      = ctx.Input(kRowPtrInputIndex);
  Tensor* col_indices  = ctx.Input(kColIndicesInputIndex);
  Tensor* values       = ctx.Input(kValuesInputIndex);
  Tensor* weight       = ctx.Input(kWeightInputIndex);   // weight is [N, K] row-major
  Tensor* output       = ctx.Output(kOutputIndex);        // output is [M, N]

  // ----------------------------
  // 2) Get data pointers
  // ----------------------------
  auto matrix_shape_data = reinterpret_cast<int32_t*>(matrix_shape->GetData());
  auto row_ptr_data      = reinterpret_cast<int32_t*>(row_ptr->GetData());
  auto col_indices_data  = reinterpret_cast<int32_t*>(col_indices->GetData());
  auto values_data       = reinterpret_cast<T*>(values->GetData());
  auto weight_data       = reinterpret_cast<T*>(weight->GetData());
  auto output_data       = reinterpret_cast<T*>(output->GetData());

  if (!matrix_shape_data || !row_ptr_data || !col_indices_data ||
      !values_data || !weight_data || !output_data) {
    CUST_KERNEL_LOG_ERROR(ctx, "Null tensor data pointer.");
    return PARAM_INVALID;
  }

  // ----------------------------
  // 3) Validate matrix_shape ([M, K])
  // ----------------------------
  const size_t ms_elems = matrix_shape->GetDataSize() / sizeof(int32_t);
  if (ms_elems < 2) {
    CUST_KERNEL_LOG_ERROR(ctx, "matrix_shape must have >=2 int32, got %zu.", ms_elems);
    return PARAM_INVALID;
  }

  const int32_t M = matrix_shape_data[0];
  const int32_t K = matrix_shape_data[1];
  if (M <= 0 || K <= 0) {
    CUST_KERNEL_LOG_ERROR(ctx, "Invalid matrix_shape: M=%d, K=%d.", M, K);
    return PARAM_INVALID;
  }

  // ----------------------------
  // 4) Validate CSR row_ptr
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
      CUST_KERNEL_LOG_ERROR(ctx, "row_ptr must be non-decreasing at i=%d: %d > %d.",
                            i, row_ptr_data[i], row_ptr_data[i + 1]);
      return PARAM_INVALID;
    }
  }
  const int32_t nnz = row_ptr_data[M];
  if (nnz < 0) {
    CUST_KERNEL_LOG_ERROR(ctx, "nnz is negative: %d.", nnz);
    return PARAM_INVALID;
  }

//   // values/col_indices length == nnz
//   const size_t values_elems = values->GetDataSize() / sizeof(T);
//   const size_t cols_elems   = col_indices->GetDataSize() / sizeof(int32_t);
//   if (static_cast<int64_t>(values_elems) != static_cast<int64_t>(nnz) ||
//       static_cast<int64_t>(cols_elems)   != static_cast<int64_t>(nnz)) {
//     CUST_KERNEL_LOG_ERROR(ctx, "CSR nnz mismatch: nnz=%d, values=%zu, cols=%zu.",
//                           nnz, values_elems, cols_elems);
//     return PARAM_INVALID;
//   }

  // ----------------------------
  // 5) Validate weight shape: [N, K]
  // ----------------------------
  auto wshape = weight->GetTensorShape();
  if (!wshape || wshape->GetDims() != 2) {
    CUST_KERNEL_LOG_ERROR(ctx, "weight must be 2D [N,K].");
    return PARAM_INVALID;
  }

  const int32_t N  = static_cast<int32_t>(wshape->GetDimSize(0));
  const int32_t wK = static_cast<int32_t>(wshape->GetDimSize(1));
  if (N <= 0 || wK != K) {
    CUST_KERNEL_LOG_ERROR(ctx, "weight dims mismatch: weight=[%d,%d], expected [N,%d].",
                          N, wK, K);
    return PARAM_INVALID;
  }

  const size_t weight_elems = weight->GetDataSize() / sizeof(T);
  if (weight_elems != static_cast<size_t>(static_cast<int64_t>(N) * K)) {
    CUST_KERNEL_LOG_ERROR(ctx, "weight buffer size mismatch: expect %lld elems, got %zu.",
                          static_cast<long long>(static_cast<int64_t>(N) * K), weight_elems);
    return PARAM_INVALID;
  }

  // output capacity >= M*N
  const int64_t out_elems_need = static_cast<int64_t>(M) * N;
  const size_t  out_elems_have = output->GetDataSize() / sizeof(T);
  if (static_cast<int64_t>(out_elems_have) < out_elems_need) {
    CUST_KERNEL_LOG_ERROR(ctx, "output too small: need %lld elems, got %zu.",
                          static_cast<long long>(out_elems_need), out_elems_have);
    return PARAM_INVALID;
  }

  // ----------------------------
  // 6) Zero init output
  // ----------------------------
  std::fill(output_data, output_data + out_elems_need, static_cast<T>(0));

  // ----------------------------
  // 7) Compute (keep your original batching logic):
  // For each row i:
  //   process nnz in chunks (<=vl)
  //   for each n:
  //     gather weight[n, k_lane] for lane-wise k indices
  //     dot = sum_lane(values_lane * weight_lane)
  //     output[i,n] += dot
  // weight layout: [N,K] row-major => weight(n,k) = weight_data[n*K + k]
  // ----------------------------
#ifdef __ARM_FEATURE_SVE
  if (std::is_same<T, float>::value) {
    const float* typed_values_data = reinterpret_cast<const float*>(values_data);
    const float* typed_weight_data = reinterpret_cast<const float*>(weight_data);
    float* typed_output_data = reinterpret_cast<float*>(output_data);

    const uint32_t vl = svcntw();  // number of float lanes

    for (int32_t i = 0; i < M; ++i) {
      const int32_t row_start = row_ptr_data[i];
      const int32_t row_end   = row_ptr_data[i + 1];

      if (row_start < 0 || row_end < 0 || row_start > row_end || row_end > nnz) {
        CUST_KERNEL_LOG_ERROR(ctx, "Invalid row range at i=%d: [%d,%d), nnz=%d.",
                              i, row_start, row_end, nnz);
        return PARAM_INVALID;
      }

      // Calculate number of complete SVE vectors and remainder
      const int32_t total_elements = row_end - row_start;
      const int32_t num_complete_vectors = total_elements / vl;
      const int32_t remainder = total_elements % vl;

      // Process complete SVE vectors
      int32_t j = row_start;
      for (int32_t vec_idx = 0; vec_idx < num_complete_vectors; ++vec_idx) {
        // predicate for all lanes [0, vl)
        svbool_t pg = svwhilelt_b32((uint32_t)0, vl);

        // load k indices (int32) and values (float)
        svint32_t  vk     = svld1_s32(pg, col_indices_data + j);
        svfloat32_t vvals = svld1_f32(pg, typed_values_data + j);

        // mask out invalid k lanes: (0 <= k < K)
        svbool_t pg_ge0 = svcmpge_s32(pg, vk, svdup_n_s32(0));
        svbool_t pg_ltK = svcmplt_s32(pg, vk, svdup_n_s32(K));
        svbool_t pg_kok = svand_b_z(pg, pg_ge0, pg_ltK);

        // If any lanes valid, process
        if (svptest_any(pg, pg_kok)) {
          // For each output column n
          for (int32_t n = 0; n < N; ++n) {
            const float* wrow = typed_weight_data + static_cast<int64_t>(n) * K; // weight(n, :)
            // gather weight(n, k_lane) using element indices vk
            svfloat32_t vw = svld1_gather_s32index_f32(pg_kok, wrow, vk);

            // multiply and horizontal add
            svfloat32_t vprod = svmul_f32_x(pg_kok, vw, vvals);
            float dot = svaddv_f32(pg_kok, vprod);

            typed_output_data[static_cast<int64_t>(i) * N + n] += dot;
          }
        }

        j += vl;
      }

      // Process remaining elements if any
      if (remainder > 0) {
        const uint32_t elems = static_cast<uint32_t>(remainder);
        svbool_t pg = svwhilelt_b32((uint32_t)0, elems);

        // load k indices (int32) and values (float)
        svint32_t  vk     = svld1_s32(pg, col_indices_data + j);
        svfloat32_t vvals = svld1_f32(pg, typed_values_data + j);

        // mask out invalid k lanes: (0 <= k < K)
        svbool_t pg_ge0 = svcmpge_s32(pg, vk, svdup_n_s32(0));
        svbool_t pg_ltK = svcmplt_s32(pg, vk, svdup_n_s32(K));
        svbool_t pg_kok = svand_b_z(pg, pg_ge0, pg_ltK);

        // If any lanes valid, process
        if (svptest_any(pg, pg_kok)) {
          // For each output column n
          for (int32_t n = 0; n < N; ++n) {
            const float* wrow = typed_weight_data + static_cast<int64_t>(n) * K; // weight(n, :)
            // gather weight(n, k_lane) using element indices vk
            svfloat32_t vw = svld1_gather_s32index_f32(pg_kok, wrow, vk);

            // multiply and horizontal add
            svfloat32_t vprod = svmul_f32_x(pg_kok, vw, vvals);
            float dot = svaddv_f32(pg_kok, vprod);

            typed_output_data[static_cast<int64_t>(i) * N + n] += dot;
          }
        }
      }
    }
  } else {
    // For other types, use scalar implementation
    for (int32_t i = 0; i < M; ++i) {
      const int32_t row_start = row_ptr_data[i];
      const int32_t row_end   = row_ptr_data[i + 1];

      if (row_start < 0 || row_end < 0 || row_start > row_end || row_end > nnz) {
        CUST_KERNEL_LOG_ERROR(ctx, "Invalid row range at i=%d: [%d,%d), nnz=%d.",
                              i, row_start, row_end, nnz);
        return PARAM_INVALID;
      }

      for (int32_t j = row_start; j < row_end; ++j) {
        const int32_t k = col_indices_data[j];
        if (k < 0 || k >= K) continue;

        const T a = values_data[j];
        if (a == static_cast<T>(0)) continue;

        for (int32_t n = 0; n < N; ++n) {
          output_data[static_cast<int64_t>(i) * N + n] += a * weight_data[static_cast<int64_t>(n) * K + k];
        }
      }
    }
  }

#else
  // Scalar fallback (still correct for weight [N,K])
  for (int32_t i = 0; i < M; ++i) {
    const int32_t row_start = row_ptr_data[i];
    const int32_t row_end   = row_ptr_data[i + 1];

    if (row_start < 0 || row_end < 0 || row_start > row_end || row_end > nnz) {
      CUST_KERNEL_LOG_ERROR(ctx, "Invalid row range at i=%d: [%d,%d), nnz=%d.",
                            i, row_start, row_end, nnz);
      return PARAM_INVALID;
    }

    for (int32_t j = row_start; j < row_end; ++j) {
      const int32_t k = col_indices_data[j];
      if (k < 0 || k >= K) continue;

      const T a = values_data[j];
      if (a == static_cast<T>(0)) continue;

      for (int32_t n = 0; n < N; ++n) {
        output_data[static_cast<int64_t>(i) * N + n] += a * weight_data[static_cast<int64_t>(n) * K + k];
      }
    }
  }
#endif
    return SUCCESS;
}

REGISTER_CPU_KERNEL(SPARSE_MAT_MUL_T_GATHER_CSR, SparseMatMulTGatherCSRCpuKernel);
} // namespace aicpu
