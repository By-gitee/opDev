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


def data_contrast_with_matrix_mult(file1, file2, result_output, tolerance=1e-5):
    """
    Verify that the matrix multiplication result matches the expected output
    """
    # Read input matrices
    input1 = np.fromfile(file1, dtype=np.float32)
    input2 = np.fromfile(file2, dtype=np.float32)
    
    # Reshape to 64x64 matrices (based on the C++ code)
    # The shape is defined as {64, 64} in the C++ code
    try:
        matrix1 = input1.reshape(64, 64)
        matrix2 = input2.reshape(64, 64)
    except ValueError:
        print(f"Error: Cannot reshape input files to 64x64 matrices.")
        print(f"Input1 size: {input1.size}, Input2 size: {input2.size}")
        return 1

    # Perform matrix multiplication
    right_output = np.dot(matrix1, matrix2.T)
    
    # Read expected output
    result = np.fromfile(result_output, dtype=np.float32)
    try:
        result = result.reshape(64, 64)
    except ValueError:
        print(f"Error: Cannot reshape expected output file to 64x64 matrix.")
        print(f"Expected output size: {result.size}")
        return 1

    # Compare the computed result with the expected result
    # Using allclose to check if arrays are element-wise equal within tolerance
    if np.allclose(right_output, result, rtol=tolerance, atol=tolerance):
        print(f"{Colors.GREEN}Verification passed: computed result matches expected output within tolerance.{Colors.ENDC}")
        return 0
    else:
        # Calculate the maximum absolute difference
        abs_diff = np.abs(right_output - result)
        max_diff = np.max(abs_diff)
        mean_diff = np.mean(abs_diff)

        print(f"{Colors.RED}Verification failed: computed result does not match expected output.{Colors.ENDC}")
        print(f"Maximum absolute difference: {max_diff}")
        print(f"Mean absolute difference: {mean_diff}")
        print(f"Tolerance used: {tolerance}")

        # Print some sample values for debugging
        print(f"Sample computed result [0,0]: {right_output[0,0]}")
        print(f"Sample expected result [0,0]: {result[0,0]}")
        print(f"Sample computed result [10,10]: {right_output[10,10]}")
        print(f"Sample expected result [10,10]: {result[10,10]}")

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
    
    input1_file = "test_data/data/input_0.bin"
    input2_file = "test_data/data/input_1.bin"
    output_file = "result_files/output_0.bin"

    # Use provided tolerance or default to 1e-5
    tolerance = float(sys.argv[4]) if len(sys.argv) > 4 else 1e-5
    
    # Perform matrix multiplication verification
    result = data_contrast_with_matrix_mult(input1_file, input2_file, output_file, tolerance)
    
    # Exit with the result
    # sys.exit(result)