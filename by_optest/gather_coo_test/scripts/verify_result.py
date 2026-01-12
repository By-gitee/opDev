"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
CREATED:  2023-02-20 10:12:13
MODIFIED: 2025-12-29 Updated to perform matrix multiplication verification with colored output
"""
import sys
import math
import numpy as np

# Define color codes for terminal output
class Colors:
    GREEN = '\033[92m'  # Green
    RED = '\033[91m'    # Red
    ENDC = '\033[0m'    # End color


def data_contrast_with_sparse_matmul(file1, file2, indices_x_file, indices_y_file, result_output, tolerance=1e-5):
    """
    Verify the sparse matrix multiplication result with COO format inputs
    """
    # Read input matrices
    sparse_matrix = np.fromfile(file1, dtype=np.float32)  # Matrix A: [M, K]
    weight_matrix = np.fromfile(file2, dtype=np.float32)  # Matrix B_T: [N, K] (transposed)

    # Read indices
    indices_x = np.fromfile(indices_x_file, dtype=np.int32)  # [row0, nnz0, row1, nnz1, ...]
    indices_y = np.fromfile(indices_y_file, dtype=np.int32)  # concatenated column indices

    # The indicesX length must be even (pairs of [row, count])
    if len(indices_x) % 2 != 0:
        print(f"Error: indicesX length must be even, got {len(indices_x)}.")
        return 1

    row_block_num = len(indices_x) // 2

    # Determine matrix dimensions
    # From the C++ kernel, we know that:
    # - sparse_matrix is [M, K]
    # - weight_matrix is [N, K] (already transposed)
    # - output is [M, N]
    # We can infer K from the total number of elements in sparse_matrix divided by number of rows
    # Get the maximum row index to determine M
    if len(indices_x) > 0:
        max_row_idx = max(indices_x[0::2])  # Get every other element starting from 0 (row indices)
        M = max_row_idx + 1
    else:
        print("Error: indicesX is empty.")
        return 1

    # Infer K from the sparse matrix size and M
    if len(sparse_matrix) % M == 0:
        K = len(sparse_matrix) // M
    else:
        print(f"Error: Cannot infer K dimension. Sparse matrix size {len(sparse_matrix)} is not divisible by M={M}.")
        return 1

    # Infer N from the weight matrix size and K
    if len(weight_matrix) % K == 0:
        N = len(weight_matrix) // K
    else:
        print(f"Error: Cannot infer N dimension. Weight matrix size {len(weight_matrix)} is not divisible by K={K}.")
        return 1

    # Reshape matrices
    try:
        matrix_A = sparse_matrix.reshape(M, K)
        matrix_B_T = weight_matrix.reshape(N, K)  # B is already transposed
    except ValueError:
        print(f"Error: Cannot reshape matrices with inferred dimensions M={M}, K={K}, N={N}.")
        print(f"Sparse matrix size: {len(sparse_matrix)}, Weight matrix size: {len(weight_matrix)}")
        return 1

    # Initialize output matrix
    result_computed = np.zeros((M, N), dtype=np.float32)

    # Process according to the kernel algorithm
    y_offset = 0
    for r in range(row_block_num):
        m_idx = indices_x[r * 2]      # row index
        cnt = indices_x[r * 2 + 1]    # count of non-zeros for this row

        if cnt <= 0:
            continue
        if m_idx < 0 or m_idx >= M:
            y_offset += cnt
            continue

        # For each column n in the output
        for n in range(N):
            acc = 0.0
            # Sum over the non-zero elements in row m_idx
            for t in range(cnt):
                k_idx = indices_y[y_offset + t]
                if k_idx >= 0 and k_idx < K:
                    acc += matrix_A[m_idx, k_idx] * matrix_B_T[n, k_idx]  # B_T[n, k] = B[k, n]

            result_computed[m_idx, n] = acc

        y_offset += cnt

    # Read expected output
    result = np.fromfile(result_output, dtype=np.float32)
    try:
        result = result.reshape(M, N)
    except ValueError:
        print(f"Error: Cannot reshape expected output file to ({M},{N}) matrix.")
        print(f"Expected output size: {result.size}")
        return 1

    # Compare the computed result with the expected result
    # Using allclose to check if arrays are element-wise equal within tolerance
    if np.allclose(result_computed, result, rtol=tolerance, atol=tolerance):
        print(f"{Colors.GREEN}Verification passed: computed sparse matmul result matches expected output within tolerance.{Colors.ENDC}")
        print(f"Matrix dimensions: A({M}x{K}), B_T({N}x{K}), Output({M}x{N})")
        print(f"Processed {row_block_num} row blocks with {len(indices_y)} total non-zero elements.")
        return 0
    else:
        # Calculate the maximum absolute difference
        abs_diff = np.abs(result_computed - result)
        max_diff = np.max(abs_diff)
        mean_diff = np.mean(abs_diff)

        print(f"{Colors.RED}Verification failed: computed result does not match expected output.{Colors.ENDC}")
        print(f"Matrix dimensions: A({M}x{K}), B_T({N}x{K}), Output({M}x{N})")
        print(f"Maximum absolute difference: {max_diff}")
        print(f"Mean absolute difference: {mean_diff}")
        print(f"Tolerance used: {tolerance}")

        # Print some sample values for debugging
        print(f"Sample computed result [0,0]: {result_computed[0,0]}")
        print(f"Sample expected result [0,0]: {result[0,0]}")
        if M > 10 and N > 10:
            print(f"Sample computed result [10,10]: {result_computed[10,10]}")
            print(f"Sample expected result [10,10]: {result[10,10]}")

        return 1




if __name__ == '__main__':
    # if len(sys.argv) < 5:
    #     print("Usage: python verify_result.py <sparse_matrix_file> <weight_matrix_file> <indices_x_file> <indices_y_file> [output_file] [tolerance]")
    #     print("Default files: input_0.bin, input_1.bin, indices_x.bin, indices_y.bin, output_0.bin")
    #     sys.exit(1)

    # Parse arguments
    indices_x_file = "test_data/data/input_0.bin"  # indices X
    indices_y_file = "test_data/data/input_1.bin"  # indices Y
    input1_file = "test_data/data/input_2.bin"  # sparse matrix
    input2_file = "test_data/data/input_3.bin"  # weight matrix (transposed)

    output_file = "result_files/output_0.bin"
    tolerance = float(sys.argv[6]) if len(sys.argv) > 6 else 1e-5

    # Perform sparse matrix multiplication verification
    result = data_contrast_with_sparse_matmul(input1_file, input2_file, indices_x_file, indices_y_file, output_file, tolerance)

    # Exit with the result
    sys.exit(result)