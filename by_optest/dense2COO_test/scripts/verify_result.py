"""
Copyright (R) @huawei.com, all rights reserved
-*- coding:utf-8 -*-
CREATED:  2023-02-20 10:12:13
MODIFIED: 2025-12-29 Updated to perform DenseToCOO verification with colored output
"""
import sys
import math
import numpy as np

# Define color codes for terminal output
class Colors:
    GREEN = '\033[92m'  # Green
    RED = '\033[91m'    # Red
    ENDC = '\033[0m'    # End color


def data_contrast_with_dense_to_coo(dense_matrix_file, threshold_file, indices_x_file, indices_y_file, tolerance=1e-5):
    """
    Verify that the DenseToCOO result matches the expected output
    """
    # Read dense matrix
    dense_data = np.fromfile(dense_matrix_file, dtype=np.float32)

    # Read threshold value (should be a scalar)
    threshold_data = np.fromfile(threshold_file, dtype=np.float32)
    if threshold_data.size != 1:
        print(f"Error: Threshold file should contain exactly one value, but got {threshold_data.size} values.")
        return 1

    threshold = threshold_data[0]

    # Read expected outputs
    expected_indices_x = np.fromfile(indices_x_file, dtype=np.int32)
    expected_indices_y = np.fromfile(indices_y_file, dtype=np.int32)

    # Determine matrix dimensions from dense data
    # Assuming it's a square matrix, we need to figure out the dimensions
    # For now, let's assume it's a square matrix and calculate the dimension
    total_elements = dense_data.size
    # Try to determine if it's a square matrix
    dim = int(math.sqrt(total_elements))
    if dim * dim != total_elements:
        # If not square, we need to get dimensions from command line or assume a common size
        # For now, let's assume 8x8 as a common test case
        print(f"Warning: Dense matrix is not square ({total_elements} elements). Assuming 8x8 matrix for test.")
        dim = 8
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

    # Compute expected DenseToCOO result manually
    computed_indices_x = []  # Will store [row_num, count] pairs
    computed_indices_y = []  # Will store column indices

    nnz = 0  # Count of non-zero elements
    for i in range(rows):
        row_count = 0
        for j in range(cols):
            value = dense_matrix[i, j]
            # Compare absolute value with threshold for both positive and negative values
            if (value >= threshold) or (value <= -threshold):
                row_count += 1
                nnz += 1
                computed_indices_y.append(j)  # Store column index

        if row_count > 0:
            computed_indices_x.append(i)      # Row number
            computed_indices_x.append(row_count)  # Count of non-zeros in this row

    expected_indices_y = expected_indices_y[:nnz]  # Trim expected indicesY to nnz length
    # Convert to numpy arrays
    computed_indices_x = np.array(computed_indices_x, dtype=np.int32)
    computed_indices_y = np.array(computed_indices_y, dtype=np.int32)

    # Compare the computed result with the expected result
    x_match = np.array_equal(computed_indices_x, expected_indices_x)
    y_match = np.array_equal(computed_indices_y, expected_indices_y)

    if x_match and y_match:
        print(f"{Colors.GREEN}Verification passed: computed DenseToCOO result matches expected output.{Colors.ENDC}")
        print(f"Computed indicesX: {computed_indices_x}")
        print(f"Expected indicesX: {expected_indices_x}")
        print(f"Computed indicesY: {computed_indices_y}")
        print(f"Expected indicesY: {expected_indices_y}")
        return 0
    else:
        print(f"{Colors.RED}Verification failed: computed DenseToCOO result does not match expected output.{Colors.ENDC}")

        print(f"Input dense matrix ({rows}x{cols}):")
        print(dense_matrix)
        print(f"Threshold: {threshold}")

        print(f"Computed indicesX: {computed_indices_x}")
        print(f"Expected indicesX: {expected_indices_x}")
        print(f"Match: {x_match}")

        print(f"Computed indicesY: {computed_indices_y}")
        print(f"Expected indicesY: {expected_indices_y}")
        print(f"Match: {y_match}")

        # Show differences if any
        if not x_match:
            print("Differences in indicesX:")
            min_len = min(len(computed_indices_x), len(expected_indices_x))
            for i in range(min_len):
                if computed_indices_x[i] != expected_indices_x[i]:
                    print(f"  Position {i}: computed={computed_indices_x[i]}, expected={expected_indices_x[i]}")
            if len(computed_indices_x) != len(expected_indices_x):
                print(f"  Length mismatch: computed={len(computed_indices_x)}, expected={len(expected_indices_x)}")

        if not y_match:
            print("Differences in indicesY:")
            min_len = min(len(computed_indices_y), len(expected_indices_y))
            for i in range(min_len):
                if computed_indices_y[i] != expected_indices_y[i]:
                    print(f"  Position {i}: computed={computed_indices_y[i]}, expected={expected_indices_y[i]}")
            if len(computed_indices_y) != len(expected_indices_y):
                print(f"  Length mismatch: computed={len(computed_indices_y)}, expected={len(expected_indices_y)}")

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
    # if len(sys.argv) < 4:
    #     print("Usage: python verify_result.py <dense_matrix_file> <threshold_file> <indices_x_file> <indices_y_file> [tolerance]")
    #     print("Example: python verify_result.py test_data/data/input_0.bin test_data/data/input_1.bin result_files/output_0.bin result_files/output_1.bin 1e-5")
    #     sys.exit(1)

    dense_matrix_file = "test_data/data/input_0.bin"
    threshold_file = "test_data/data/input_1.bin"
    indices_x_file = "result_files/output_0.bin"
    indices_y_file = "result_files/output_1.bin"

    # Use provided tolerance or default to 1e-5
    tolerance = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-5

    # Perform DenseToCOO verification
    result = data_contrast_with_dense_to_coo(dense_matrix_file, threshold_file, indices_x_file, indices_y_file, tolerance)

    # Exit with the result
    sys.exit(result)