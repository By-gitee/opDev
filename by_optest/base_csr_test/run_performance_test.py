  
#!/usr/bin/env python3 
import subprocess  
import re  
import sys  
import os  
  
def extract_execution_time(output):  
    \"\"\"Extract execution time from output.\"\"\"  
    pattern = r'Operator execution time:\\s*([\\d.]+)\\s*ms'  
    match = re.search(pattern, output)  
    return float(match.group(1)) if match else None  
  
def main(): 
    \"\"\"Main function to run the performance test.\"\"\"  
    # Default values  
    executable_path = \"./run/out/by_execute_sparse_matmul_base\"  
    n_iterations = 10  
  
    # Parse command line arguments  
    if len(sys.argv)  
        try:  
            n_iterations = int(sys.argv[1])  
        except ValueError:  
            print(\"Usage: python run_performance_test.py [N]\")  
            print(\"  N: Number of iterations to run (default: 10)\")  
            sys.exit(1)  
  
    print(f\"Running performance test: {n_iterations} iterations\")  
    print(f\"Executable: {executable_path}\")  
    print(\"-\" * 50)  
  
    execution_times = [] 
    for i in range(n_iterations):  
        try:  
            print(f\"Running iteration {i + 1}/{n_iterations}...\")  
            result = subprocess.run([executable_path], capture_output=True, text=True, timeout=300)  
            if result.returncode != 0:  
                print(f\"Warning: Execution {i + 1} failed with return code {result.returncode}\")  
                continue  
            execution_time = extract_execution_time(result.stdout)  
            if execution_time is not None:  
                execution_times.append(execution_time)  
                print(f\"  Execution time: {execution_time:.3f} ms\")  
            else:  
                print(f\"  Warning: Could not extract execution time from output\")  
        except subprocess.TimeoutExpired:  
            print(f\"Warning: Execution {i + 1} timed out\")  
        except Exception as e:  
            print(f\"Warning: Error running execution {i + 1}: {str(e)}\")  
  
    if execution_times:  
        avg_time = sum(execution_times) / len(execution_times)  
        print(\"-\" * 50)  
        print(\"PERFORMANCE TEST RESULTS:\")  
        print(f\"Successful runs: {len(execution_times)}/{n_iterations}\")  
        print(f\"Average execution time: {avg_time:.3f} ms\")  
        print(f\"All execution times: {[f'{t:.3f}' for t in execution_times]}\")  
    else:  
        print(\"No successful executions found.\")  
  
if __name__ == \"__main__\":  
    main() 
