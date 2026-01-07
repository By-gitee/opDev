
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: implement of SparseMatMulGatherCOO
 */
#define __ARM_FEATURE_SVE


#include "sparse_mat_mul_gather_coo_kernels.h"
#include <type_traits>
#include <climits>
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
    // Inputs:
    // indicesX: [row0, nnz0, row1, nnz1, ...]  (int32)
    // indicesY: concatenated column indices for each row in the same order (int32)
    // sparse_matrix (A): [M, K] (dense storage, but many zeros)
    // weight (B_T): [N, K] (already transposed)
    // output (C): [M, N]

    Tensor* indicesX = ctx.Input(kIndicesXInputIndex);
    Tensor* indicesY = ctx.Input(kIndicesYInputIndex);
    Tensor* sparse_matrix = ctx.Input(kSparseMatrixInputIndex);
    Tensor* weight = ctx.Input(kWeightInputIndex);
    Tensor* output = ctx.Output(kOutputIndex);

    int32_t* ix = reinterpret_cast<int32_t*>(indicesX->GetData());
    int32_t* iy = reinterpret_cast<int32_t*>(indicesY->GetData());
    T* A_data = reinterpret_cast<T*>(sparse_matrix->GetData());
    T* B_data = reinterpret_cast<T*>(weight->GetData());
    T* C_data = reinterpret_cast<T*>(output->GetData());

    if (ix == nullptr || iy == nullptr || A_data == nullptr || B_data == nullptr || C_data == nullptr) {
        CUST_KERNEL_LOG_ERROR(ctx, "Input or output data pointer is nullptr.");
        return PARAM_INVAILD;
    }

    auto ix_shape = indicesX->GetTensorShape();
    auto iy_shape = indicesY->GetTensorShape();
    auto A_shape  = sparse_matrix->GetTensorShape();
    auto B_shape  = weight->GetTensorShape();
    auto C_shape  = output->GetTensorShape();

    if (ix_shape == nullptr || iy_shape == nullptr || A_shape == nullptr || B_shape == nullptr || C_shape == nullptr) {
        CUST_KERNEL_LOG_ERROR(ctx, "TensorShape is nullptr.");
        return PARAM_INVAILD;
    }

    // indicesX length must be even (pairs)
    const int64_t ix_len = ix_shape->GetDimSize(0);
    if ((ix_len & 1) != 0) {
        CUST_KERNEL_LOG_ERROR(ctx, "indicesX length must be even, got %ld.", ix_len);
        return PARAM_INVAILD;
    }
    const int64_t row_block_num = ix_len / 2;

    // total nnz is indicesY length
    const int64_t nnz = iy_shape->GetDimSize(0);

    // A: [M, K]
    const int64_t M = A_shape->GetDimSize(0);
    const int64_t K = A_shape->GetDimSize(1);

    // B_T: [N, K]
    const int64_t N  = B_shape->GetDimSize(0);
    const int64_t Kb = B_shape->GetDimSize(1);

    // C: [M, N]
    const int64_t outM = C_shape->GetDimSize(0);
    const int64_t outN = C_shape->GetDimSize(1);

    if (K != Kb) {
        CUST_KERNEL_LOG_ERROR(ctx, "Dimension mismatch: A.cols(K)=%ld != weight.cols(K)=%ld", K, Kb);
        return PARAM_INVAILD;
    }
    if (outM != M || outN != N) {
        CUST_KERNEL_LOG_ERROR(ctx, "Output dimension mismatch: expected [%ld, %ld], got [%ld, %ld]",
                              M, N, outM, outN);
        return PARAM_INVAILD;
    }

    // Initialize output to zero
    const int64_t out_size = outM * outN;
    for (int64_t i = 0; i < out_size; ++i) {
        C_data[i] = static_cast<T>(0);
    }

    // Validate that counts in indicesX match indicesY length (best-effort)
    int64_t nnz_check = 0;
    for (int64_t r = 0; r < row_block_num; ++r) {
        int32_t cnt = ix[r * 2 + 1];
        if (cnt > 0) nnz_check += (int64_t)cnt;
    }
    if (nnz_check != nnz) {
        CUST_KERNEL_LOG_ERROR(ctx, "indicesY length (%ld) != sum(indicesX counts) (%ld).", nnz, nnz_check);
        return PARAM_INVAILD;
    }

#ifdef __ARM_FEATURE_SVE
    // -------------------------------------------------------------------------
    // SVE gather implementation:
    // For each row m and its k-list (length nnz_m):
    //   C[m, n] = sum_t A[m, k_t] * B_T[n, k_t]  (B_T is [N, K])
    // We gather-load A and B_T using element indices (not byte offsets).
    // -------------------------------------------------------------------------

    if (std::is_same<T, float>::value) {
        const float* A = reinterpret_cast<const float*>(A_data);
        const float* B = reinterpret_cast<const float*>(B_data);
        float* C = reinterpret_cast<float*>(C_data);

        int64_t y_off = 0; // offset into indicesY
        const uint32_t vl = svcntw();
        const svbool_t pg_full = svptrue_b32();

        for (int64_t r = 0; r < row_block_num; ++r) {
            const int32_t m_i32 = ix[r * 2 + 0];
            const int32_t cnt_i32 = ix[r * 2 + 1];

            if (cnt_i32 <= 0) continue;
            if ((int64_t)m_i32 < 0 || (int64_t)m_i32 >= M) {
                y_off += (int64_t)cnt_i32;
                continue;
            }

            const int64_t m = (int64_t)m_i32;
            const int64_t cnt = (int64_t)cnt_i32;

            // base element index for A row
            const int64_t baseA64 = m * K;
            if (baseA64 > INT32_MAX) {
                // Fallback scalar if index cannot fit in s32 gather
                for (int64_t n = 0; n < N; ++n) {
                    float sum = 0.0f;
                    for (int64_t t = 0; t < cnt; ++t) {
                        int32_t k = iy[y_off + t];
                        if ((uint32_t)k >= (uint32_t)K) continue;
                        float a = A[baseA64 + (int64_t)k];
                        float b = B[n * K + (int64_t)k];
                        sum += a * b;
                    }
                    C[m * N + n] = sum;
                }
                y_off += cnt;
                continue;
            }
            const int32_t baseA = (int32_t)baseA64;

            for (int64_t n64 = 0; n64 < N; ++n64) {
                const int64_t baseB64 = n64 * K;
                if (baseB64 > INT32_MAX) {
                    // scalar fallback for this n
                    float sum = 0.0f;
                    for (int64_t t = 0; t < cnt; ++t) {
                        int32_t k = iy[y_off + t];
                        if ((uint32_t)k >= (uint32_t)K) continue;
                        float a = A[baseA64 + (int64_t)k];
                        float b = B[baseB64 + (int64_t)k];
                        sum += a * b;
                    }
                    C[m * N + n64] = sum;
                    continue;
                }
                const int32_t baseB = (int32_t)baseB64;

                float acc = 0.0f;

                int64_t t = 0;
                const int64_t vec_num = cnt / (int64_t)vl;
                const int64_t mod = cnt % (int64_t)vl;

                // full vectors
                for (int64_t v = 0; v < vec_num; ++v) {
                    svint32_t vk = svld1_s32(pg_full, &iy[y_off + t]); // k indices

                    // idxA = baseA + k; idxB = baseB + k
                    svint32_t iA = svadd_n_s32_z(pg_full, vk, baseA);
                    svint32_t iB = svadd_n_s32_z(pg_full, vk, baseB);

                    svfloat32_t vA = svld1_gather_s32index_f32(pg_full, A, iA);
                    svfloat32_t vB = svld1_gather_s32index_f32(pg_full, B, iB);

                    svfloat32_t vP = svmul_f32_z(pg_full, vA, vB);
                    acc += svaddv_f32(pg_full, vP);

                    t += (int64_t)vl;
                }

                // tail
                if (mod) {
                    svbool_t pg = svwhilelt_b32((uint32_t)0, (uint32_t)mod);
                    svint32_t vk = svld1_s32(pg, &iy[y_off + t]);

                    svint32_t iA = svadd_n_s32_z(pg, vk, baseA);
                    svint32_t iB = svadd_n_s32_z(pg, vk, baseB);

                    svfloat32_t vA = svld1_gather_s32index_f32(pg, A, iA);
                    svfloat32_t vB = svld1_gather_s32index_f32(pg, B, iB);

                    svfloat32_t vP = svmul_f32_z(pg, vA, vB);
                    acc += svaddv_f32(pg, vP);
                }

                C[m * N + n64] = acc;
            }

            y_off += cnt;
        }

        return SUCCESS;
    }

    // For Eigen::half, we can also use SVE optimization
    if (std::is_same<T, Eigen::half>::value) {
        const Eigen::half* A_h = reinterpret_cast<const Eigen::half*>(A_data);
        const Eigen::half* B_h = reinterpret_cast<const Eigen::half*>(B_data);
        Eigen::half* C_h = reinterpret_cast<Eigen::half*>(C_data);

        int64_t y_off = 0; // offset into indicesY
        const uint32_t vl = svcnth();  // vector length for float16
        const svbool_t pg_full = svptrue_b16();  // predicate for float16

        for (int64_t r = 0; r < row_block_num; ++r) {
            const int32_t m_i32 = ix[r * 2 + 0];
            const int32_t cnt_i32 = ix[r * 2 + 1];

            if (cnt_i32 <= 0) continue;
            if ((int64_t)m_i32 < 0 || (int64_t)m_i32 >= M) {
                y_off += (int64_t)cnt_i32;
                continue;
            }

            const int64_t m = (int64_t)m_i32;
            const int64_t cnt = (int64_t)cnt_i32;

            // base element index for A row
            const int64_t baseA64 = m * K;
            if (baseA64 > INT32_MAX) {
                // Fallback scalar if index cannot fit in s32 gather
                for (int64_t n = 0; n < N; ++n) {
                    float sum = 0.0f;
                    for (int64_t t = 0; t < cnt; ++t) {
                        int32_t k = iy[y_off + t];
                        if ((uint32_t)k >= (uint32_t)K) continue;
                        float a = static_cast<float>(A_h[baseA64 + (int64_t)k]);
                        float b = static_cast<float>(B_h[n * K + (int64_t)k]);
                        sum += a * b;
                    }
                    C_h[m * N + n] = static_cast<Eigen::half>(sum);
                }
                y_off += cnt;
                continue;
            }
            const int32_t baseA = (int32_t)baseA64;

            for (int64_t n64 = 0; n64 < N; ++n64) {
                const int64_t baseB64 = n64 * K;
                if (baseB64 > INT32_MAX) {
                    // scalar fallback for this n
                    float sum = 0.0f;
                    for (int64_t t = 0; t < cnt; ++t) {
                        int32_t k = iy[y_off + t];
                        if ((uint32_t)k >= (uint32_t)K) continue;
                        float a = static_cast<float>(A_h[baseA64 + (int64_t)k]);
                        float b = static_cast<float>(B_h[baseB64 + (int64_t)k]);
                        sum += a * b;
                    }
                    C_h[m * N + n64] = static_cast<Eigen::half>(sum);
                    continue;
                }
                const int32_t baseB = (int32_t)baseB64;

                float acc = 0.0f;

                int64_t t = 0;
                const int64_t vec_num = cnt / (int64_t)vl;
                const int64_t mod = cnt % (int64_t)vl;

                // full vectors
                for (int64_t v = 0; v < vec_num; ++v) {
                    svint32_t vk = svld1_s32(pg_full, &iy[y_off + t]); // k indices

                    // idxA = baseA + k; idxB = baseB + k
                    svint32_t iA = svadd_n_s32_z(pg_full, vk, baseA);
                    svint32_t iB = svadd_n_s32_z(pg_full, vk, baseB);

                    // Convert Eigen::half pointers to __fp16 for SVE operations
                    const __fp16* A_ptr = reinterpret_cast<const __fp16*>(A_h);
                    const __fp16* B_ptr = reinterpret_cast<const __fp16*>(B_h);

                    svuint32_t uA = svreinterpret_u32_s32(iA);
                    svuint32_t uB = svreinterpret_u32_s32(iB);
                    svfloat16_t vA = svld1_gather_u32index_f16(pg_full, (const float16_t*)A_ptr, uA);
                    svfloat16_t vB = svld1_gather_u32index_f16(pg_full, (const float16_t*)B_ptr, uB);

                    svfloat16_t vP = svmul_f16_z(pg_full, vA, vB);
                    acc += svaddv_f16(pg_full, vP);

                    t += (int64_t)vl;
                }

                // tail
                if (mod) {
                    svbool_t pg = svwhilelt_b16((uint32_t)0, (uint32_t)mod);
                    svint32_t vk = svld1_s32(pg, &iy[y_off + t]);

                    svint32_t iA = svadd_n_s32_z(pg, vk, baseA);
                    svint32_t iB = svadd_n_s32_z(pg, vk, baseB);

                    const __fp16* A_ptr = reinterpret_cast<const __fp16*>(A_h);
                    const __fp16* B_ptr = reinterpret_cast<const __fp16*>(B_h);

                    svfloat16_t vA = svld1_gather_s32index_f16(pg, A_ptr, iA);
                    svfloat16_t vB = svld1_gather_s32index_f16(pg, B_ptr, iB);

                    svfloat16_t vP = svmul_f16_z(pg, vA, vB);
                    acc += svaddv_f16(pg, vP);
                }

                C_h[m * N + n64] = static_cast<Eigen::half>(acc);
            }

            y_off += cnt;
        }

        return SUCCESS;
    }

//     // NOTE: For other types, we keep the scalar fallback below.
    #endif

    // -------------------------------------------------------------------------
    // Scalar fallback (still using indicesX pairs + indicesY concatenation)
    // -------------------------------------------------------------------------
    int64_t y_off = 0;
    for (int64_t r = 0; r < row_block_num; ++r) {
        int32_t m_i32 = ix[r * 2 + 0];
        int32_t cnt_i32 = ix[r * 2 + 1];

        if (cnt_i32 <= 0) continue;
        if ((int64_t)m_i32 < 0 || (int64_t)m_i32 >= M) {
            y_off += (int64_t)cnt_i32;
            continue;
        }

        const int64_t m = (int64_t)m_i32;
        const int64_t cnt = (int64_t)cnt_i32;

        for (int64_t n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (int64_t t = 0; t < cnt; ++t) {
                int32_t k = iy[y_off + t];
                if ((uint32_t)k >= (uint32_t)K) continue;

                float a = static_cast<float>(A_data[m * K + (int64_t)k]);
                float b = static_cast<float>(B_data[n * K + (int64_t)k]); // B_T[n,k]
                acc += a * b;
            }
            C_data[m * N + n] = static_cast<T>(acc);
        }

        y_off += cnt;
    }

    return SUCCESS;
}

REGISTER_CPU_KERNEL(SPARSE_MAT_MUL_GATHER_COO, SparseMatMulGatherCOOCpuKernel);
} // namespace aicpu
