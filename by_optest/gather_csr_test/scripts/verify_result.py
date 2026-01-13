"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
CREATED:  2023-02-20 10:12:13
MODIFIED: 2025-12-29 Updated to perform SparseMatMulTGatherCSR verification with colored output
"""
import sys
import math
import numpy as np

# Define color codes for terminal output
class Colors:
    GREEN = '\033[92m'  # Green
    RED = '\033[91m'    # Red
    ENDC = '\033[0m'    # End color


def data_contrast_with_sparse_matmul_gather_csr(matrix_shape_file, row_ptr_file, col_indices_file, values_file, weight_file, output_file, tolerance=1e-5):
    """
    Verify the SparseMatMulTGatherCSR result: sparse_matrix (CSR format) * weight_matrix = output_matrix
    Note: For SparseMatMulTGatherCSR, weight matrix has shape [N, K] instead of [K, N]
    """
    # Read CSR format sparse matrix components
    matrix_shape_data = np.fromfile(matrix_shape_file, dtype=np.int32)
    row_ptr_data = np.fromfile(row_ptr_file, dtype=np.int32)
    col_indices_data = np.fromfile(col_indices_file, dtype=np.int32)
    values_data = np.fromfile(values_file, dtype=np.float32)

    # Read weight matrix
    weight_data = np.fromfile(weight_file, dtype=np.float32)

    # Validate matrix shape data
    if len(matrix_shape_data) < 2:
        print(f"Error: Matrix shape should contain at least 2 elements (M, K), but got {len(matrix_shape_data)}.")
        return 1

    # Extract matrix dimensions
    M = int(matrix_shape_data[0])  # Number of rows in sparse matrix
    K = int(matrix_shape_data[1])  # Number of columns in sparse matrix

    # Validate CSR structure
    if len(row_ptr_data) != M + 1:
        print(f"Error: Row pointer size should be M+1 ({M + 1}), but got {len(row_ptr_data)}.")
        return 1

    nnz = len(values_data)  # Number of non-zero elements
    if len(col_indices_data) != nnz:
        print(f"Error: Column indices size ({len(col_indices_data)}) should match values size ({nnz}).")
        return 1

    # For SparseMatMulTGatherCSR, weight matrix has shape [N, K] instead of [K, N]
    N = len(weight_data) // K  # Number of rows in weight matrix

    # Reshape weight matrix to [N, K] - this is the key difference from BaseCSR
    try:
        weight_matrix = weight_data.reshape(N, K)
    except ValueError:
        print(f"Error: Cannot reshape weight matrix data to ({N}, {K}). Expected total size: {N * K}, got: {len(weight_data)}")
        return 1

    # Initialize output matrix [M, N]
    result_computed = np.zeros((M, N), dtype=np.float32)

    # Perform CSR sparse matrix multiplication for SparseMatMulTGatherCSR: output = sparse_matrix * weight_matrix
    # For each row in the sparse matrix
    for i in range(M):
        row_start = row_ptr_data[i]
        row_end = row_ptr_data[i + 1]

        # For each non-zero element in the current row
        for j in range(row_start, row_end):
            col_idx = int(col_indices_data[j])
            val = values_data[j]

            # Validate column index bounds
            if col_idx < 0 or col_idx >= K:
                print(f"Warning: Column index {col_idx} out of bounds [0, {K}) for element at position {j}. Skipping.")
                continue

            # For SparseMatMulTGatherCSR, we compute output[i, n] += val * weight[n, col_idx]
            # This means for each output column n, we get the corresponding weight value at [n, col_idx]
            for n in range(N):
                result_computed[i, n] += val * weight_matrix[n, col_idx]

    # Read expected output
    expected_output = np.fromfile(output_file, dtype=np.float32)
    try:
        expected_output = expected_output.reshape(M, N)
    except ValueError:
        print(f"Error: Cannot reshape expected output file to ({M}, {N}) matrix.")
        print(f"Expected output size: {expected_output.size}")
        return 1

    # Compare the computed result with the expected result
    if np.allclose(result_computed, expected_output, rtol=tolerance, atol=tolerance):
        print(f"{Colors.GREEN}Verification passed: computed SparseMatMulTGatherCSR result matches expected output within tolerance.{Colors.ENDC}")
        print(f"Matrix dimensions: sparse({M}x{K}), weight({N}x{K}), output({M}x{N})")
        print(f"Number of non-zero elements: {nnz}")
        return 0
    else:
        # Calculate the maximum absolute difference
        abs_diff = np.abs(result_computed - expected_output)
        max_diff = np.max(abs_diff)
        mean_diff = np.mean(abs_diff)

        print(f"{Colors.RED}Verification failed: computed result does not match expected output.{Colors.ENDC}")
        print(f"Matrix dimensions: sparse({M}x{K}), weight({N}x{K}), output({M}x{N})")
        print(f"Number of non-zero elements: {nnz}")
        print(f"Maximum absolute difference: {max_diff}")
        print(f"Mean absolute difference: {mean_diff}")
        print(f"Tolerance used: {tolerance}")

        # Print some sample values for debugging
        print(f"Sample computed result [0,0]: {result_computed[0,0]}")
        print(f"Sample expected result [0,0]: {expected_output[0,0]}")
        if M > 1 and N > 1:
            print(f"Sample computed result [1,1]: {result_computed[1,1]}")
            print(f"Sample expected result [1,1]: {expected_output[1,1]}")

        return 1


def data_contrast_legacy(file1, file2):
    """
    Legacy verification function that checks if data are the same
    """
    data1 = np.fromfile(file1, dtype=np.float32)
    data2 = np.fromfile(file2, dtype=np.float32)

    if (str(data1) == str(data2)):
        return 0
    else:
        return 1


if __name__ == '__main__':
    # if len(sys.argv) < 6:
    #     print("Usage: python verify_result.py <matrix_shape_file> <row_ptr_file> <col_indices_file> <values_file> <weight_file> <output_file> [tolerance]")
    #     print("Example: python verify_result.py test_data/data/input_0.bin test_data/data/input_1.bin test_data/data/input_2.bin test_data/data/input_3.bin test_data/data/input_4.bin result_files/output_0.bin 1e-5")
    #     sys.exit(1)

    matrix_shape_file = "test_data/data/input_0.bin"  # Matrix shape [M, K]
    row_ptr_file = "test_data/data/input_1.bin"       # Row pointers [M+1]
    col_indices_file = "test_data/data/input_2.bin"   # Column indices [nnz]
    values_file = "test_data/data/input_3.bin"        # Values [nnz]
    weight_file = "test_data/data/input_4.bin"        # Weight matrix [N, K] for SparseMatMulTGatherCSR
    output_file = "result_files/output_0.bin"         # Output matrix [M, N]

    # Use provided tolerance or default to 1e-5
    tolerance = float(sys.argv[7]) if len(sys.argv) > 7 else 1e-5

    # Perform SparseMatMulTGatherCSR verification
    result = data_contrast_with_sparse_matmul_gather_csr(matrix_shape_file, row_ptr_file, col_indices_file, values_file, weight_file, output_file, tolerance)

    # Exit with the result
    sys.exit(result)