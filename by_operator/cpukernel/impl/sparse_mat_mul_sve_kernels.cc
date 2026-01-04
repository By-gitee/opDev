
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: implement of SparseMatMulSVE
 */
#define __ARM_FEATURE_SVE

#include "sparse_mat_mul_sve_kernels.h"
#include <type_traits>
#ifdef __ARM_FEATURE_SVE
#include <arm_sve.h>
#endif

namespace  {
const char *SPARSE_MAT_MUL_SVE = "SparseMatMulSVE";
const uint32_t kFirstInputIndex = 0;
const uint32_t kSecondInputIndex = 1;
const uint32_t kFirstOutputIndex = 0;
const uint32_t SUCCESS = 0;
const uint32_t PARAM_INVAILD = 1;
const uint32_t ERROR = 2;
}

namespace aicpu  {
uint32_t SparseMatMulSVECpuKernel::Compute(CpuKernelContext &ctx)
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
	if(inputShape0->GetDimSize(1) != inputShape1->GetDimSize(0)) {
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

    // Maybe useful
	// auto outputShape = output->GetTensorShape();
//   auto outputData = output->GetData();

    // Save output
    // outputData[0] = inputData[0];
    return SUCCESS;
}



template<typename T>
uint32_t SparseMatMulSVECpuKernel::SparseMatMulCompute(CpuKernelContext &ctx)
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
uint32_t SparseMatMulSVECpuKernel::SparseMatMulComputeWithBlock(CpuKernelContext &ctx,
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
	uint32_t N = inputShape1->GetDimSize(1);
  if(block_size==0){
    block_size = M;
  }
	uint32_t block_elem = block_size*block_size;

  T* A_base_ptr = A + blockX_id*blockY_dim*block_elem;
  T* B_base_ptr = B + blockY_id*block_size;
  T* C_base_ptr = C + blockX_id*blockY_dim*block_elem + blockY_id*block_size;

#ifdef __ARM_FEATURE_SVE
  // ARM SVE optimized version for float type
  if (std::is_same<T, float>::value) {
    float *A_f = reinterpret_cast<float*>(A_base_ptr);
    float *B_f = reinterpret_cast<float*>(B_base_ptr);
    float *C_f = reinterpret_cast<float*>(C_base_ptr);

    // Process K dimension in vector chunks using SVE for better performance
    for(uint32_t i=0; i<block_size; ++i) {
      for(uint32_t j=0; j<block_size; ++j) {
        float sum = 0.0f;
        uint32_t k = 0;

        // Process K dimension in vector chunks using SVE
        while(k < K) {
          uint32_t vl = svcntw(); // Get the vector length
          uint32_t remaining = K - k;
          uint32_t current_vl = (remaining < vl) ? remaining : vl;

          svbool_t pg = svwhilelt_b32(0, current_vl);

          // Load elements from A[i,k:k+vl] and B[k:k+vl,j] for computation
          svfloat32_t va = svld1_f32(pg, &A_f[i*K + k]);
          // Load B values for the same k range but fixed j column
          svfloat32_t vb = svld1_f32(pg, &B_f[k*N + j]);

          // Multiply and accumulate
          svfloat32_t vmul = svmul_f32(pg, va, vb);
          sum += svaddv_f32(pg, vmul);

          k += current_vl;
        }

        C_f[i*N + j] = sum;
      }
    }
  } else if (std::is_same<T, Eigen::half>::value) {
    // For half precision, use SVE optimization
    Eigen::half *A_h = reinterpret_cast<Eigen::half*>(A_base_ptr);
    Eigen::half *B_h = reinterpret_cast<Eigen::half*>(B_base_ptr);
    Eigen::half *C_h = reinterpret_cast<Eigen::half*>(C_base_ptr);

    for(uint32_t i=0; i<block_size; ++i) {
      for(uint32_t j=0; j<block_size; ++j) {
        float sum = 0.0f;
        uint32_t k = 0;

        // Process K dimension in vector chunks using SVE for float16
        while(k < K) {
          uint32_t vl = svcnth(); // Get the vector length for float16
          uint32_t remaining = K - k;
          uint32_t current_vl = (remaining < vl) ? remaining : vl;

          svbool_t pg = svwhilelt_b16(0, current_vl);

          // Load elements from A[i,k:k+vl] and B[k:k+vl,j] for computation
          svfloat16_t va = svld1_f16(pg, reinterpret_cast<const svfloat16_t*>(&A_h[i*K + k]));
          // Load B values for the same k range but fixed j column
          svfloat16_t vb = svld1_f16(pg, reinterpret_cast<const svfloat16_t*>(&B_h[k*N + j]));

          // Multiply and accumulate
          svfloat16_t vmul = svmul_f16(pg, va, vb);
          sum += svaddv_f16(pg, vmul); // Use the same predicate as used for loading

          k += current_vl;
        }

        C_h[i*N+j] = static_cast<Eigen::half>(sum);
      }
    }
  } else {
    // For other types or fallback implementation
    for(uint32_t i=0;i<block_size;++i) {
      for(uint32_t j=0;j<block_size;++j) {
        T sum = (T)0.0f;
        for(uint32_t k=0;k<K;++k) {
          sum += A_base_ptr[i*K+k] * B_base_ptr[k*N+j];
        }
        C_base_ptr[i*N+j] = sum;
      }
    }
  }
#else
  // Naive version without SVE
  for(uint32_t i=0;i<block_size;++i) {
    for(uint32_t j=0;j<block_size;++j) {
      T sum = (T)0.0f;
      for(uint32_t k=0;k<K;++k) {
        sum += A_base_ptr[i*K+k] * B_base_ptr[k*N+j];
      }
      C_base_ptr[i*N+j] = sum;
    }
  }
#endif
  return SUCCESS;
}

REGISTER_CPU_KERNEL(SPARSE_MAT_MUL_SVE, SparseMatMulSVECpuKernel);
} // namespace aicpu
