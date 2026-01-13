"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
CREATED:  2023-02-20 10:12:13
MODIFIED: 2025-12-29 Updated to perform DenseToCSR verification with colored output
"""
import sys
import math
import numpy as np

# Define color codes for terminal output
class Colors:
    GREEN = '\033[92m'  # Green
    RED = '\033[91m'    # Red
    ENDC = '\033[0m'    # End color


def data_contrast_with_dense_to_csr(dense_matrix_file, threshold_file, matrix_shape_file, row_ptr_file, col_indices_file, values_file, tolerance=1e-5):
    """
    Verify that the DenseToCSR result matches the expected output
    """
    # Read dense matrix
    dense_data = np.fromfile(dense_matrix_file, dtype=np.float32)

    # Read threshold value (should be a scalar)
    threshold_data = np.fromfile(threshold_file, dtype=np.float32)
    if threshold_data.size != 1:
        print(f"Error: Threshold file should contain exactly one value, but got {threshold_data.size} values.")
        return 1

    threshold = abs(threshold_data[0])  # Use absolute value of threshold

    # Read expected outputs
    expected_matrix_shape = np.fromfile(matrix_shape_file, dtype=np.int32)
    expected_row_ptr = np.fromfile(row_ptr_file, dtype=np.int32)
    expected_col_indices = np.fromfile(col_indices_file, dtype=np.int32)
    expected_values = np.fromfile(values_file, dtype=np.float32)

    # Only consider non-zero parts based on the threshold
    nonzero_mask = np.abs(expected_values) > threshold
    expected_col_indices = expected_col_indices[:len(expected_values)][nonzero_mask]  # Apply mask to col_indices
    expected_values = expected_values[nonzero_mask]  # Keep only values above threshold

    # Determine matrix dimensions from dense data
    # Assuming it's a square matrix, we need to figure out the dimensions
    total_elements = dense_data.size
    # Try to determine if it's a square matrix
    dim = int(math.sqrt(total_elements))
    if dim * dim != total_elements:
        # If not square, we need to get dimensions from command line or assume a common size
        # For now, let's assume 64x64 as per the main.cpp
        print(f"Warning: Dense matrix is not square ({total_elements} elements). Assuming 64x64 matrix for test.")
        dim = 64
        if total_elements % dim != 0:
            print(f"Error: Cannot determine matrix dimensions from {total_elements} elements.")
            return 1
        rows = dim
        cols = total_elements // dim
    else:
        rows = cols = dim

    # Reshape dense matrix
    try:
        dense_matrix = dense_data.reshape(rows, cols)
    except ValueError:
        print(f"Error: Cannot reshape dense matrix data to {rows}x{cols} matrix.")
        print(f"Dense data size: {dense_data.size}")
        return 1

    # Compute expected DenseToCSR result manually
    computed_matrix_shape = np.array([rows, cols], dtype=np.int32)

    # Build CSR representation
    computed_col_indices = []
    computed_values = []
    row_ptr = [0]  # Start with 0

    for i in range(rows):
        row_start = len(computed_col_indices)  # Current position in col_indices/values arrays
        for j in range(cols):
            value = dense_matrix[i, j]
            # Compare absolute value with threshold for both positive and negative values
            if abs(value) > threshold:
                computed_col_indices.append(j)  # Store column index
                computed_values.append(value)   # Store value
        row_end = len(computed_col_indices)    # End position after processing this row
        row_ptr.append(row_end)                # Record end position for this row

    computed_row_ptr = np.array(row_ptr, dtype=np.int32)
    computed_col_indices = np.array(computed_col_indices, dtype=np.int32)
    computed_values = np.array(computed_values, dtype=np.float32)

    # Only compare non-zero elements that are above the threshold
    # Find indices where both computed and expected values are non-zero
    computed_nonzero_mask = np.abs(computed_values) > tolerance
    expected_nonzero_mask = np.abs(expected_values) > tolerance

    # Check if the lengths match for CSR components
    col_idx_len_match = len(computed_col_indices) == len(expected_col_indices)
    values_len_match = len(computed_values) == len(expected_values)

    # If lengths don't match, the CSR structure is different
    if not (col_idx_len_match and values_len_match):
        print(f"{Colors.RED}Verification failed: CSR structure mismatch.{Colors.ENDC}")
        print(f"Expected col_indices length: {len(expected_col_indices)}, computed: {len(computed_col_indices)}")
        print(f"Expected values length: {len(expected_values)}, computed: {len(computed_values)}")
        return 1

    # Compare the CSR components
    shape_match = np.array_equal(computed_matrix_shape, expected_matrix_shape)
    row_ptr_match = np.array_equal(computed_row_ptr, expected_row_ptr)
    col_indices_match = np.array_equal(computed_col_indices, expected_col_indices)
    values_match = np.allclose(computed_values, expected_values, atol=tolerance)

    if shape_match and row_ptr_match and col_indices_match and values_match:
        print(f"{Colors.GREEN}Verification passed: computed DenseToCSR result matches expected output.{Colors.ENDC}")
        print(f"Matrix shape - Computed: {computed_matrix_shape}, Expected: {expected_matrix_shape}")
        print(f"Row ptr - Match: {row_ptr_match}")
        print(f"Col indices - Match: {col_indices_match}")
        print(f"Values - Match: {values_match}")
        return 0
    else:
        print(f"{Colors.RED}Verification failed: computed DenseToCSR result does not match expected output.{Colors.ENDC}")

        print(f"Input dense matrix ({rows}x{cols}):")
        print(dense_matrix)
        print(f"Threshold: {threshold}")

        print(f"Computed matrix shape: {computed_matrix_shape}")
        print(f"Expected matrix shape: {expected_matrix_shape}")
        print(f"Shape match: {shape_match}")

        print(f"Computed row_ptr: {computed_row_ptr}")
        print(f"Expected row_ptr: {expected_row_ptr}")
        print(f"Row ptr match: {row_ptr_match}")

        print(f"Computed col_indices: {computed_col_indices}")
        print(f"Expected col_indices: {expected_col_indices}")
        print(f"Col indices match: {col_indices_match}")

        print(f"Computed values: {computed_values}")
        print(f"Expected values: {expected_values}")
        print(f"Values match: {values_match}")

        # Show differences if any
        if not shape_match:
            print("Differences in matrix shape:")
            min_len = min(len(computed_matrix_shape), len(expected_matrix_shape))
            for i in range(min_len):
                if computed_matrix_shape[i] != expected_matrix_shape[i]:
                    print(f"  Position {i}: computed={computed_matrix_shape[i]}, expected={expected_matrix_shape[i]}")
            if len(computed_matrix_shape) != len(expected_matrix_shape):
                print(f"  Length mismatch: computed={len(computed_matrix_shape)}, expected={len(expected_matrix_shape)}")

        if not row_ptr_match:
            print("Differences in row_ptr:")
            min_len = min(len(computed_row_ptr), len(expected_row_ptr))
            for i in range(min_len):
                if computed_row_ptr[i] != expected_row_ptr[i]:
                    print(f"  Position {i}: computed={computed_row_ptr[i]}, expected={expected_row_ptr[i]}")
            if len(computed_row_ptr) != len(expected_row_ptr):
                print(f"  Length mismatch: computed={len(computed_row_ptr)}, expected={len(expected_row_ptr)}")

        if not col_indices_match:
            print("Differences in col_indices:")
            min_len = min(len(computed_col_indices), len(expected_col_indices))
            for i in range(min_len):
                if computed_col_indices[i] != expected_col_indices[i]:
                    print(f"  Position {i}: computed={computed_col_indices[i]}, expected={expected_col_indices[i]}")
            if len(computed_col_indices) != len(expected_col_indices):
                print(f"  Length mismatch: computed={len(computed_col_indices)}, expected={len(expected_col_indices)}")

        if not values_match:
            print("Differences in values:")
            min_len = min(len(computed_values), len(expected_values))
            for i in range(min_len):
                if not np.isclose(computed_values[i], expected_values[i], atol=tolerance):
                    print(f"  Position {i}: computed={computed_values[i]}, expected={expected_values[i]}")
            if len(computed_values) != len(expected_values):
                print(f"  Length mismatch: computed={len(computed_values)}, expected={len(expected_values)}")

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
    #     print("Usage: python verify_result.py <dense_matrix_file> <threshold_file> <matrix_shape_file> <row_ptr_file> <col_indices_file> <values_file> [tolerance]")
    #     print("Example: python verify_result.py test_data/data/input_0.bin test_data/data/input_1.bin result_files/output_0.bin result_files/output_1.bin result_files/output_2.bin result_files/output_3.bin 1e-5")
    #     sys.exit(1)

    dense_matrix_file = "test_data/data/input_0.bin"
    threshold_file = "test_data/data/input_1.bin"
    matrix_shape_file = "result_files/output_0.bin"
    row_ptr_file = "result_files/output_1.bin"
    col_indices_file = "result_files/output_2.bin"
    values_file = "result_files/output_3.bin"

    # Use provided tolerance or default to 1e-5
    tolerance = float(sys.argv[7]) if len(sys.argv) > 7 else 1e-5

    # Perform DenseToCSR verification
    result = data_contrast_with_dense_to_csr(dense_matrix_file, threshold_file, matrix_shape_file, row_ptr_file, col_indices_file, values_file, tolerance)

    # Exit with the result
    sys.exit(result)