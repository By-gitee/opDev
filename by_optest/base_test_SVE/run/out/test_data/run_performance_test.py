#!/usr/bin/env python3
import subprocess
import re
import sys

# ========================
# 参数检查
# ========================
if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <N>")
    sys.exit(1)

N = int(sys.argv[1])
exe_path = "./by_execute_sparse_matmul_base"

# ========================
# 正则：匹配时间
# 例：Operator execution time: 63.793 ms
# ========================
pattern = re.compile(r"Operator execution time:\s*([0-9.]+)\s*ms")

times = []

for i in range(N):
    print(f"Run {i+1}/{N} ...")

    result = subprocess.run(
        [exe_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    output = result.stdout
    print(output)

    match = pattern.search(output)
    if not match:
        print("❌ Failed to parse execution time!")
        sys.exit(1)

    time_ms = float(match.group(1))
    times.append(time_ms)

# ========================
# 计算平均值
# ========================
avg_time = sum(times) / len(times)

print("===================================")
print(f"Runs: {N}")
print(f"Times (ms): {times}")
print(f"Average execution time: {avg_time:.3f} ms")
print("===================================")
