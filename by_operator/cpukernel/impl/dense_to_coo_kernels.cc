
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: implement of DenseToCOO
 */
#include "dense_to_coo_kernels.h"
#include <vector>
#include "cpu_tensor.h"
#include "cpu_tensor_shape.h"
#include "cpu_types.h"
#include "cust_cpu_utils.h"

namespace  {
const char *DENSE_TO_COO = "DenseToCOO";
const uint32_t kDenseMatrixInputIndex = 0;
const uint32_t kThresholdInputIndex = 1;
const uint32_t kIndicesXOutputIndex = 0;
const uint32_t kIndicesYOutputIndex = 1;
const uint32_t SUCCESS = 0;
const uint32_t PARAM_INVAILD = 1;
const uint32_t ERROR = 2;
}

namespace aicpu  {
uint32_t DenseToCOOCpuKernel::Compute(CpuKernelContext &ctx)
{
    // Get input tensors
    Tensor* dense_matrix = ctx.Input(kDenseMatrixInputIndex);
    Tensor* threshold_tensor = ctx.Input(kThresholdInputIndex);

    // Get output tensors
    Tensor* indicesX = ctx.Output(kIndicesXOutputIndex);
    Tensor* indicesY = ctx.Output(kIndicesYOutputIndex);

    // Check input & output addresses
    if (dense_matrix == nullptr || threshold_tensor == nullptr ||
        indicesX == nullptr || indicesY == nullptr) {
        CUST_KERNEL_LOG_ERROR(ctx, "Input or output tensor is nullptr.");
        return PARAM_INVAILD;
    }

    // Get input tensor DataType
    DataType inputType = dense_matrix->GetDataType();
    DataType thresholdType = threshold_tensor->GetDataType();

    if (inputType != thresholdType) {
        CUST_KERNEL_LOG_ERROR(ctx, "DataType of dense_matrix and threshold does not match.");
        return PARAM_INVAILD;
    }

    // Dispatch based on data type
    switch (inputType) {
        case DT_FLOAT16:
            return DenseToCOOCompute<Eigen::half>(ctx);
        case DT_FLOAT:
            return DenseToCOOCompute<float>(ctx);
        default:
            CUST_KERNEL_LOG_ERROR(ctx, "Unsupported data type: %d", static_cast<int>(inputType));
            return PARAM_INVAILD;
    }

    return SUCCESS;
}


template<typename T>
uint32_t DenseToCOOCpuKernel::DenseToCOOCompute(CpuKernelContext &ctx)
{
    // Get threshold value
    Tensor* threshold_tensor = ctx.Input(kThresholdInputIndex);
    T* threshold_data = reinterpret_cast<T*>(threshold_tensor->GetData());
    T threshold = threshold_data[0];  // Assuming threshold is a scalar

    // Get input tensor
    Tensor* dense_matrix = ctx.Input(kDenseMatrixInputIndex);
    auto input_shape = dense_matrix->GetTensorShape();

    int64_t rows = input_shape->GetDimSize(0);
    int64_t cols = input_shape->GetDimSize(1);

    T* dense_data = reinterpret_cast<T*>(dense_matrix->GetData());

    // Use temporary vectors to collect row counts and column indices
    std::vector<int32_t> temp_indicesX;  // Will store [row_num, count] pairs
    std::vector<int32_t> temp_indicesY;  // Will store column indices

    for (int64_t i = 0; i < rows; ++i) {
        int32_t row_count = 0;
        for (int64_t j = 0; j < cols; ++j) {
            T value = dense_data[i * cols + j];
            // Compare absolute value with threshold for both positive and negative values
            if ((value >= threshold) || (value <= -threshold)) {
                row_count++;
                temp_indicesY.push_back(static_cast<int32_t>(j));  // Store column index
            }
        }
        if (row_count > 0) {
            temp_indicesX.push_back(static_cast<int32_t>(i));      // Row number
            temp_indicesX.push_back(row_count);                    // Count of non-zeros in this row
        }
    }

    // Set output tensor shapes based on the new format
    Tensor* indicesX = ctx.Output(kIndicesXOutputIndex);
    Tensor* indicesY = ctx.Output(kIndicesYOutputIndex);

    uint32_t non_zero_count = static_cast<uint32_t>(temp_indicesY.size());  // Total count of non-zero elements

    // Create new tensor shapes for outputs
    std::vector<int64_t> output_shape_indicesX = {static_cast<int64_t>(temp_indicesX.size())};  // Size of [row_num, count] pairs
    std::vector<int64_t> output_shape_indicesY = {static_cast<int64_t>(non_zero_count)};  // Size of column indices

    indicesX->GetTensorShape()->SetDimSizes(output_shape_indicesX);
    indicesY->GetTensorShape()->SetDimSizes(output_shape_indicesY);
    // indicesX->SetTensorShape(indicesXShape);
    // indicesY->SetTensorShape(indicesYShape);

    // Allocate memory for output tensors if needed
    indicesX->SetDataSize(temp_indicesX.size() * sizeof(int32_t));
    indicesY->SetDataSize(non_zero_count * sizeof(int32_t));

    // Copy temporary data to output tensors
    if (!temp_indicesX.empty() || !temp_indicesY.empty()) {
        int32_t* indicesX_data = reinterpret_cast<int32_t*>(indicesX->GetData());
        int32_t* indicesY_data = reinterpret_cast<int32_t*>(indicesY->GetData());

        std::copy(temp_indicesX.begin(), temp_indicesX.end(), indicesX_data);
        std::copy(temp_indicesY.begin(), temp_indicesY.end(), indicesY_data);
    }

    return SUCCESS;
}

REGISTER_CPU_KERNEL(DENSE_TO_COO, DenseToCOOCpuKernel);
} // namespace aicpu
