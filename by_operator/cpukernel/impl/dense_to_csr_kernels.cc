
/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2021. All rights reserved.
 * Description: implement of DenseToCSR
 */
#include "dense_to_csr_kernels.h"
#include <vector>
#include "cpu_tensor.h"
#include "cpu_tensor_shape.h"
#include "cpu_types.h"
#include "cust_cpu_utils.h"
#include "eigen3/Eigen/Dense"




namespace  {
const char *DENSE_TO_CSR = "DenseToCSR";
const uint32_t kDenseMatrixInputIndex = 0;
const uint32_t kThresholdInputIndex = 1;
const uint32_t kMatrixShapeOutputIndex = 0;
const uint32_t kRowPtrOutputIndex = 1;
const uint32_t kColIndicesOutputIndex = 2;
const uint32_t kValuesOutputIndex = 3;
const uint32_t SUCCESS = 0;
const uint32_t PARAM_INVALID = 1;
const uint32_t ERROR = 2;
}

namespace aicpu  {
uint32_t DenseToCSRCpuKernel::Compute(CpuKernelContext &ctx)
{
    // Get input tensors
    Tensor* dense_matrix = ctx.Input(kDenseMatrixInputIndex);
    Tensor* threshold_tensor = ctx.Input(kThresholdInputIndex);

    // Get output tensors
    Tensor* matrix_shape = ctx.Output(kMatrixShapeOutputIndex);
    Tensor* row_ptr = ctx.Output(kRowPtrOutputIndex);
    Tensor* col_indices = ctx.Output(kColIndicesOutputIndex);
    Tensor* values = ctx.Output(kValuesOutputIndex);

    // Check input & output addresses
    if (dense_matrix == nullptr || threshold_tensor == nullptr ||
        matrix_shape == nullptr || row_ptr == nullptr ||
        col_indices == nullptr || values == nullptr) {
        CUST_KERNEL_LOG_ERROR(ctx, "Input or output tensor is nullptr.");
        return PARAM_INVALID;
    }

    // Get input tensor DataType
    DataType inputType = dense_matrix->GetDataType();
    DataType thresholdType = threshold_tensor->GetDataType();

    if (inputType != thresholdType) {
        CUST_KERNEL_LOG_ERROR(ctx, "DataType of dense_matrix and threshold does not match.");
        return PARAM_INVALID;
    }

    // Dispatch based on data type
    switch (inputType) {
        case DT_FLOAT16:
            return DenseToCSRCompute<Eigen::half>(ctx);
        case DT_FLOAT:
            return DenseToCSRCompute<float>(ctx);
        default:
            CUST_KERNEL_LOG_ERROR(ctx, "Unsupported data type: %d", static_cast<int>(inputType));
            return PARAM_INVALID;
    }

    return SUCCESS;
}


template<typename T>
uint32_t DenseToCSRCpuKernel::DenseToCSRCompute(CpuKernelContext &ctx)
{
    // Get threshold value
    Tensor* threshold_tensor = ctx.Input(kThresholdInputIndex);
    T* threshold_data = reinterpret_cast<T*>(threshold_tensor->GetData());
    T threshold = threshold_data[0];  // Assuming threshold is a scalar

    // Get input tensor
    Tensor* dense_matrix = ctx.Input(kDenseMatrixInputIndex);
    auto input_shape = dense_matrix->GetTensorShape();

    // Validate input shape - must be 2D
    if (input_shape->GetDims() != 2) {
        CUST_KERNEL_LOG_ERROR(ctx, "Input dense_matrix must be 2D, but got %d dimensions",
                              input_shape->GetDims());
        return PARAM_INVALID;
    }

    int64_t rows = input_shape->GetDimSize(0);
    int64_t cols = input_shape->GetDimSize(1);

    T* dense_data = reinterpret_cast<T*>(dense_matrix->GetData());

    // Pre-count non-zero elements to avoid dynamic expansion
    uint32_t nnz_count = 0;
    std::vector<int32_t> row_nnz_counts(rows, 0);  // Count non-zeros in each row

    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < cols; ++j) {
            T value = dense_data[i * cols + j];
            // Only keep values that have absolute value greater than threshold
            if ((value >= threshold) || (value <= -threshold)) {
                row_nnz_counts[i]++;
                nnz_count++;
            }
        }
    }

    // Pre-allocate vectors with exact sizes needed
    std::vector<int32_t> temp_col_indices;
    std::vector<T> temp_values;
    std::vector<int32_t> temp_row_ptr;

    if (nnz_count > 0) {
        temp_col_indices.reserve(nnz_count);
        temp_values.reserve(nnz_count);
    }

    temp_row_ptr.resize(rows + 1, 0);  // Initialize row_ptr with zeros

    // Calculate cumulative counts for row_ptr
    temp_row_ptr[0] = 0;
    for (int64_t i = 1; i <= rows; ++i) {
        temp_row_ptr[i] = temp_row_ptr[i-1] + row_nnz_counts[i-1];
    }

    // Resize vectors to exact size needed and fill with data
    temp_col_indices.resize(nnz_count);
    temp_values.resize(nnz_count);

    // Fill the arrays with known sizes in one pass
    for (int64_t i = 0; i < rows; ++i) {
        int32_t row_start = temp_row_ptr[i];
        int32_t pos_in_row = 0;

        for (int64_t j = 0; j < cols; ++j) {
            T value = dense_data[i * cols + j];
            // Only keep values that have absolute value greater than threshold
            if ((value >= threshold) || (value <= -threshold)) {
                int32_t global_pos = row_start + pos_in_row;
                temp_col_indices[global_pos] = static_cast<int32_t>(j);
                temp_values[global_pos] = value;
                pos_in_row++;
            }
        }
    }

    // Set output tensor shapes
    Tensor* matrix_shape = ctx.Output(kMatrixShapeOutputIndex);
    Tensor* row_ptr = ctx.Output(kRowPtrOutputIndex);
    Tensor* col_indices = ctx.Output(kColIndicesOutputIndex);
    Tensor* values = ctx.Output(kValuesOutputIndex);

    // Set matrix_shape output (2 elements: [rows, cols])
    std::vector<int64_t> matrix_shape_dims = {2};
    matrix_shape->GetTensorShape()->SetDimSizes(matrix_shape_dims);
    matrix_shape->SetDataSize(2 * sizeof(int32_t));

    // Set row_ptr output (rows + 1 elements)
    std::vector<int64_t> row_ptr_dims = {rows + 1};
    row_ptr->GetTensorShape()->SetDimSizes(row_ptr_dims);
    row_ptr->SetDataSize((rows + 1) * sizeof(int32_t));

    // Set col_indices output (number of non-zero elements)
    uint32_t nnz = static_cast<uint32_t>(temp_col_indices.size());
    std::vector<int64_t> col_indices_dims = {static_cast<int64_t>(nnz)};
    col_indices->GetTensorShape()->SetDimSizes(col_indices_dims);
    col_indices->SetDataSize(nnz * sizeof(int32_t));

    // Set values output (number of non-zero elements, same type as input)
    std::vector<int64_t> values_dims = {static_cast<int64_t>(nnz)};
    values->GetTensorShape()->SetDimSizes(values_dims);
    values->SetDataSize(nnz * sizeof(T));

    // Copy data to output tensors
    if (nnz > 0 || rows > 0) {
        // Copy matrix shape
        int32_t* matrix_shape_data = reinterpret_cast<int32_t*>(matrix_shape->GetData());
        matrix_shape_data[0] = static_cast<int32_t>(rows);
        matrix_shape_data[1] = static_cast<int32_t>(cols);

        // Copy row_ptr
        int32_t* row_ptr_data = reinterpret_cast<int32_t*>(row_ptr->GetData());
        std::copy(temp_row_ptr.begin(), temp_row_ptr.end(), row_ptr_data);

        // Copy col_indices
        if (nnz > 0) {
            int32_t* col_indices_data = reinterpret_cast<int32_t*>(col_indices->GetData());
            std::copy(temp_col_indices.begin(), temp_col_indices.end(), col_indices_data);

            // Copy values
            T* values_data = reinterpret_cast<T*>(values->GetData());
            std::copy(temp_values.begin(), temp_values.end(), values_data);
        }
    } else {
        // If no non-zero elements, still set the matrix shape
        int32_t* matrix_shape_data = reinterpret_cast<int32_t*>(matrix_shape->GetData());
        matrix_shape_data[0] = static_cast<int32_t>(rows);
        matrix_shape_data[1] = static_cast<int32_t>(cols);

        // Set row_ptr to all zeros if no non-zeros
        int32_t* row_ptr_data = reinterpret_cast<int32_t*>(row_ptr->GetData());
        for (int64_t i = 0; i <= rows; ++i) {
            row_ptr_data[i] = 0;
        }
    }

    return SUCCESS;
}

REGISTER_CPU_KERNEL(DENSE_TO_CSR, DenseToCSRCpuKernel);
} // namespace aicpu
