
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: implement of SparseMatMulTBaseSVE
 */
#define __ARM_FEATURE_SVE

#include "sparse_mat_mul_sve_kernels.h"
#include <type_traits>
#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif
#include "cpu_tensor.h"
#include "cpu_tensor_shape.h"
#include "cpu_types.h"
#include "cust_cpu_utils.h"
#include <iostream>
namespace  {
const char *SPARSE_MAT_MUL_SVE = "SparseMatMulTBaseSVE";
const uint32_t kFirstInputIndex = 0;
const uint32_t kSecondInputIndex = 1;
const uint32_t kFirstOutputIndex = 0;
const uint32_t SUCCESS = 0;
const uint32_t PARAM_INVAILD = 1;
const uint32_t ERROR = 2;
}

namespace aicpu  {
uint32_t SparseMatMulTBaseSVECpuKernel::Compute(CpuKernelContext &ctx)
{

    Tensor* input0 = ctx.Input(kFirstInputIndex);
    Tensor* input1 = ctx.Input(kSecondInputIndex);
    
    Tensor* output = ctx.Output(kFirstOutputIndex);
    // Check input&output address
    if (input0 == nullptr || input1 == nullptr || output == nullptr) {
        return PARAM_INVAILD;
    }

    auto inputShape0 = input0->GetTensorShape();
    auto inputShape1 = input1->GetTensorShape();
    for (int32_t i = 0; i < inputShape0->GetDims(); ++i) {
        CUST_KERNEL_LOG_DEBUG(ctx, "SparseMatrix dim[%d] size:%ld.", i, inputShape0->GetDimSize(i));
    }
    for (int32_t i = 0; i < inputShape1->GetDims(); ++i) {
        CUST_KERNEL_LOG_DEBUG(ctx, "Weight dim[%d] size:%ld.", i, inputShape1->GetDimSize(i));
    }
	if(inputShape0->GetDimSize(1) != inputShape1->GetDimSize(1)) {
        CUST_KERNEL_LOG_DEBUG(ctx, "DataShape does not match.");
        return PARAM_INVAILD;
	}

    // Get input tensor DataType
    DataType inputType0 = input0->GetDataType();
    DataType inputType1 = input1->GetDataType();
    if (inputType0 != inputType1) {
        CUST_KERNEL_LOG_DEBUG(ctx, "DataType does not match.");
        return PARAM_INVAILD;
    }


    switch (inputType0) {
        case DT_FLOAT16:
            return SparseMatMulCompute<Eigen::half>(ctx);
        case DT_FLOAT:
            return SparseMatMulCompute<float>(ctx);
            default:
            return PARAM_INVAILD;
    }

    return SUCCESS;
}



template<typename T>
uint32_t SparseMatMulTBaseSVECpuKernel::SparseMatMulCompute(CpuKernelContext &ctx)
{
  //Get blockid and blockdim
  uint32_t blockX_id;
  uint32_t blockY_id;
  uint32_t blockX_dim;
  uint32_t blockY_dim;
  uint32_t block_size;

  AttrValue *blockX_id_ptr = ctx.GetAttr("blockX_id");
  AttrValue *blockY_id_ptr = ctx.GetAttr("blockY_id");
  AttrValue *blockX_dim_ptr = ctx.GetAttr("blockX_dim");
  AttrValue *blockY_dim_ptr = ctx.GetAttr("blockY_dim");
  AttrValue *block_size_ptr = ctx.GetAttr("block_size");
  //check block_id and block_num
  if (blockX_id_ptr == nullptr || blockX_dim_ptr == nullptr) {
    blockX_id = 0;
    blockX_dim = 1;
  } else {
    blockX_id = blockX_id_ptr->GetInt();
    blockX_dim = blockX_dim_ptr->GetInt();
  }
  if (blockY_id_ptr == nullptr || blockY_dim_ptr == nullptr) {
    blockY_id = 0;
    blockY_dim = 1;
  } else {
    blockY_id = blockY_id_ptr->GetInt();
    blockY_dim = blockY_dim_ptr->GetInt();
  }

  if (blockX_id >= blockX_dim || blockX_id < 0) {
    blockX_id = 0;
    blockX_dim = 1;
  }
  if (blockY_id >= blockY_dim || blockY_id < 0) {
    blockY_id = 0;
    blockY_dim = 1;
  }
  if (block_size_ptr == nullptr) {
    block_size = 0;
  } else {
    block_size = block_size_ptr->GetInt();
  }

  return SparseMatMulComputeWithBlock<T>(ctx, blockX_id, blockX_dim, blockY_id, blockY_dim,block_size);
}

// Optimized version with ARM SVE
template<typename T>
uint32_t SparseMatMulTBaseSVECpuKernel::SparseMatMulComputeWithBlock(CpuKernelContext &ctx,
                                                uint32_t blockX_id, uint32_t blockX_dim,
												uint32_t blockY_id, uint32_t blockY_dim,
												uint32_t block_size)
{
  Tensor *input0 = ctx.Input(kFirstInputIndex);
  Tensor *input1 = ctx.Input(kSecondInputIndex);
  Tensor *output = ctx.Output(kFirstOutputIndex);

  T *A = reinterpret_cast<T *>(input0->GetData());
  if (A == nullptr) {
    return PARAM_INVAILD;
  }

  // NOTE:
  // input1 的 B 是“已经转置后的矩阵 B_T”
  // 原始 B: [K, N] row-major
  // 现在 B_T: [N, K] row-major, 满足 B_T[n*K + k] = B_orig[k*N + n]
  T *B = reinterpret_cast<T *>(input1->GetData());
  if (B == nullptr) {
    return PARAM_INVAILD;
  }

  T *C = reinterpret_cast<T *>(output->GetData());
  if (C == nullptr) {
    return PARAM_INVAILD;
  }

  auto inputShape0 = input0->GetTensorShape();
  auto inputShape1 = input1->GetTensorShape();

  uint32_t M = inputShape0->GetDimSize(0);
  uint32_t K = inputShape0->GetDimSize(1);

  // B 是 B_T: [N, K]
  uint32_t N = inputShape1->GetDimSize(0);

  if (block_size == 0) {
    block_size = M;
  }

  // 当前 block 对应的输出子块：行范围 [m0, m1), 列范围 [n0, n1)
  uint32_t m0 = blockX_id * block_size;
  uint32_t n0 = blockY_id * block_size;

  uint32_t m1 = (m0 + block_size < M) ? (m0 + block_size) : M;
  uint32_t n1 = (n0 + block_size < N) ? (n0 + block_size) : N;

  uint32_t actual_m = m1 - m0;
  uint32_t actual_n = n1 - n0;

#ifdef __ARM_FEATURE_SVE

  // ARM SVE optimized version for float type
  if (std::is_same<T, float>::value) {
    const float *A_f = reinterpret_cast<const float*>(A);
    const float *B_f = reinterpret_cast<const float*>(B);  // B_T
    float *C_f = reinterpret_cast<float*>(C);

    for (uint32_t ii = 0; ii < actual_m; ++ii) {
      uint32_t m = m0 + ii;
      const float* A_row = A_f + m * K;

      for (uint32_t jj = 0; jj < actual_n; ++jj) {
        uint32_t n = n0 + jj;

        // 对应 B_T 的第 n 行（长度 K）
        const float* B_T_row = B_f + n * K;

        float sum = 0.0f;
        uint32_t k = 0;

        while (k < K) {
          uint32_t vl = svcntw();         // vector length in #float32 lanes
          uint32_t remaining = K - k;
          uint32_t current_vl = (remaining < vl) ? remaining : vl;

          // 解决重载歧义：显式用 uint32_t
          svbool_t pg = svwhilelt_b32((uint32_t)0, (uint32_t)current_vl);

          // Load A[m, k:k+vl]
          svfloat32_t va = svld1_f32(pg, &A_row[k]);

          // Load B_T[n, k:k+vl]  (因为 B 已经转置)
          svfloat32_t vb = svld1_f32(pg, &B_T_row[k]);

          // Multiply and reduce
          // 注意：SVE ACLE 用 svmul_f32_z/_x/_m，不是 svmul_f32(pg,...)
          svfloat32_t vmul = svmul_f32_z(pg, va, vb);
          sum += svaddv_f32(pg, vmul);

          k += current_vl;
        }

        // 写回 C[m, n]
        C_f[m * N + n] = sum;
      }
    }

  } else if (std::is_same<T, Eigen::half>::value) {
    // For half precision, use SVE optimization
    const Eigen::half *A_h = reinterpret_cast<const Eigen::half*>(A);
    const Eigen::half *B_h = reinterpret_cast<const Eigen::half*>(B);  // B_T
    Eigen::half *C_h = reinterpret_cast<Eigen::half*>(C);

    for (uint32_t ii = 0; ii < actual_m; ++ii) {
      uint32_t m = m0 + ii;
      const Eigen::half* A_row = A_h + m * K;

      for (uint32_t jj = 0; jj < actual_n; ++jj) {
        uint32_t n = n0 + jj;
        const Eigen::half* B_T_row = B_h + n * K;

        float sum = 0.0f;
        uint32_t k = 0;

        while (k < K) {
          uint32_t vl = svcnth();         // vector length in #float16 lanes
          uint32_t remaining = K - k;
          uint32_t current_vl = (remaining < vl) ? remaining : vl;

          svbool_t pg = svwhilelt_b16((uint32_t)0, (uint32_t)current_vl);

          // svld1_f16 的指针类型必须是 __fp16*（或等价 half 标量指针）
          // Eigen::half 通常与 __fp16 二进制兼容，但类型不同，需要转换成 __fp16*
          const __fp16* Ap = reinterpret_cast<const __fp16*>(&A_row[k]);
          const __fp16* Bp = reinterpret_cast<const __fp16*>(&B_T_row[k]);

          svfloat16_t va = svld1_f16(pg, Ap);
          svfloat16_t vb = svld1_f16(pg, Bp);

          // half mul 也用 _z/_x/_m 变体
          svfloat16_t vmul = svmul_f16_z(pg, va, vb);

          // GCC arm_sve.h 通常提供 svaddv_f16，但有的环境需要先扩到 f32 再 reduction
          // 这里保持你的写法（如果你环境 svaddv_f16 不支持，再告诉我我给你换成 f32 accumulate）
          sum += svaddv_f16(pg, vmul);

          k += current_vl;
        }

        C_h[m * N + n] = static_cast<Eigen::half>(sum);
      }
    }

  } else {
    // For other types or fallback implementation (B is transposed: B_T[n*K + k])
    for (uint32_t ii = 0; ii < actual_m; ++ii) {
      uint32_t m = m0 + ii;
      for (uint32_t jj = 0; jj < actual_n; ++jj) {
        uint32_t n = n0 + jj;
        T sum = (T)0.0f;
        for (uint32_t k = 0; k < K; ++k) {
          sum += A[m * K + k] * B[n * K + k];   // <-- B_T
        }
        C[m * N + n] = sum;
      }
    }
  }

#else
  // Naive version without SVE (B is transposed: B_T[n*K + k])
  for (uint32_t ii = 0; ii < actual_m; ++ii) {
    uint32_t m = m0 + ii;
    for (uint32_t jj = 0; jj < actual_n; ++jj) {
      uint32_t n = n0 + jj;
      T sum = (T)0.0f;
      for (uint32_t k = 0; k < K; ++k) {
        sum += A[m * K + k] * B[n * K + k];     // <-- B_T
      }
      C[m * N + n] = sum;
    }
  }
#endif

  return SUCCESS;
}

REGISTER_CPU_KERNEL(SPARSE_MAT_MUL_SVE, SparseMatMulTBaseSVECpuKernel);
} // namespace aicpu
