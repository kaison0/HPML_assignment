import re
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np


log_data = """
+ CXX_FLAGS=-std=c++11
+ CUDA_FLAGS=-arch=sm_60
+ echo Compiling Q1.cpp
Compiling Q1.cpp
+ g++ -std=c++11 -o Q1 Q1.cpp
+ echo Compiling Q2.cu
Compiling Q2.cu
+ nvcc -arch=sm_60 -o Q2 Q2.cu
+ echo Compiling Q3.cu
Compiling Q3.cu
+ nvcc -arch=sm_60 -o Q3 Q3.cu
+ ./Q1 1
K: 1000000 Compute Running Time: 4.45232ms
+ ./Q1 5
K: 5000000 Compute Running Time: 20.83423ms
+ ./Q1 10
K: 10000000 Compute Running Time: 40.33692ms
+ ./Q1 50
K: 50000000 Compute Running Time: 198.44794ms
+ ./Q1 100
K: 100000000 Compute Running Time: 396.25123ms
+ ./Q2 1 1
Test passed
K: 1000000 Running Time: 559.38791ms
 Compute Running Time: 111.31553ms
+ ./Q2 1 2
Test passed
K: 1000000 Running Time: 327.86172ms
 Compute Running Time: 4.25332ms
+ ./Q2 1 3
Test passed
K: 1000000 Running Time: 315.45506ms
 Compute Running Time: 3.76245ms
+ ./Q2 5 1
Test passed
K: 5000000 Running Time: 620.51850ms
 Compute Running Time: 294.89411ms
+ ./Q2 5 2
Test passed
K: 5000000 Running Time: 320.92175ms
 Compute Running Time: 15.94565ms
+ ./Q2 5 3
Test passed
K: 5000000 Running Time: 309.66932ms
 Compute Running Time: 9.40824ms
+ ./Q2 10 1
Test passed
K: 10000000 Running Time: 884.39937ms
 Compute Running Time: 548.56388ms
+ ./Q2 10 2
Test passed
K: 10000000 Running Time: 367.82120ms
 Compute Running Time: 31.63643ms
+ ./Q2 10 3
Test passed
K: 10000000 Running Time: 352.95988ms
 Compute Running Time: 18.44045ms
+ ./Q2 50 1
Test passed
K: 50000000 Running Time: 3312.35756ms
 Compute Running Time: 2735.44679ms
+ ./Q2 50 2
Test passed
K: 50000000 Running Time: 737.39978ms
 Compute Running Time: 153.72520ms
+ ./Q2 50 3
Test passed
K: 50000000 Running Time: 656.64613ms
 Compute Running Time: 83.40918ms
+ ./Q2 100 1
Test passed
K: 100000000 Running Time: 6408.53759ms
 Compute Running Time: 5519.01943ms
+ ./Q2 100 2
Test passed
K: 100000000 Running Time: 1190.71012ms
 Compute Running Time: 308.88182ms
+ ./Q2 100 3
Test passed
K: 100000000 Running Time: 1057.37146ms
 Compute Running Time: 172.97044ms
+ ./Q3 1 1
Test passed
K: 1000000 Running Time: 367.21019ms
 Compute Running Time: 59.76108ms
+ ./Q3 1 2
Test passed
K: 1000000 Running Time: 309.73899ms
 Compute Running Time: 3.46615ms
+ ./Q3 1 3
Test passed
K: 1000000 Running Time: 300.43974ms
 Compute Running Time: 3.00311ms
+ ./Q3 5 1
Test passed
K: 5000000 Running Time: 564.16092ms
 Compute Running Time: 245.20341ms
+ ./Q3 5 2
Test passed
K: 5000000 Running Time: 310.46148ms
 Compute Running Time: 12.94587ms
+ ./Q3 5 3
Test passed
K: 5000000 Running Time: 333.40249ms
 Compute Running Time: 12.52086ms
+ ./Q3 10 1
Test passed
K: 10000000 Running Time: 768.94916ms
 Compute Running Time: 448.09063ms
+ ./Q3 10 2
Test passed
K: 10000000 Running Time: 358.31150ms
 Compute Running Time: 26.17909ms
+ ./Q3 10 3
Test passed
K: 10000000 Running Time: 354.09700ms
 Compute Running Time: 25.40167ms
+ ./Q3 50 1
Test passed
K: 50000000 Running Time: 2788.42701ms
 Compute Running Time: 2252.04401ms
+ ./Q3 50 2
Test passed
K: 50000000 Running Time: 688.48587ms
 Compute Running Time: 134.20302ms
+ ./Q3 50 3
Test passed
K: 50000000 Running Time: 650.99785ms
 Compute Running Time: 120.61392ms
+ ./Q3 100 1
Test passed
K: 100000000 Running Time: 5299.75955ms
 Compute Running Time: 4497.42943ms
+ ./Q3 100 2
Test passed
K: 100000000 Running Time: 1076.97988ms
 Compute Running Time: 262.21731ms
+ ./Q3 100 3
Test passed
K: 100000000 Running Time: 1038.47464ms
 Compute Running Time: 235.88074ms
"""



K = [1, 5, 10, 50, 100]

pattern = r'Compute Running Time:\s([0-9.]+)ms'

matches = re.findall(pattern, log_data)

program_name = ['Q1', 'Q2-1', 'Q2-2', 'Q2-3', 'Q3-1', 'Q3-2', 'Q3-3']
compute_time = {}
count = 0
for i in range(len(program_name)):
    compute_time[program_name[i]] = []
    for j in range(5):
        if (i == 0):
            compute_time[program_name[i]].append(float(matches[j]))
        else:
            compute_time[program_name[i]].append(float(matches[5+((i - 1)//3) * 15 + j * 3 + (i-1) % 3]))
    

plt.figure(figsize=(8, 6))

print(compute_time['Q1'])
print(compute_time['Q2-1'])
print(compute_time['Q2-2'])
print(compute_time['Q2-3'])
plt.loglog(K, np.array(compute_time["Q1"]), label='Q1', marker='o')
plt.loglog(K, np.array(compute_time["Q2-1"]), label='Q2_1', marker='x')
plt.loglog(K, np.array(compute_time["Q2-2"]), label='Q2_2', marker='s')
plt.loglog(K, np.array(compute_time["Q2-3"]), label='Q2_3', marker='^')

plt.xlabel("K (Input Size)", fontsize=12)
plt.ylabel("Compute Time Without Unified memory (ms)", fontsize=12)
plt.title("Log-Log Plot of Compute Time Without Unified memory vs K", fontsize=14)

plt.grid(True)

plt.legend()

plt.figure(figsize=(8, 6))

print(compute_time['Q1'])
print(compute_time['Q3-1'])
print(compute_time['Q3-2'])
print(compute_time['Q3-3'])
plt.loglog(K, np.array(compute_time["Q1"]), label='Q1', marker='o')
plt.loglog(K, np.array(compute_time["Q3-1"]), label='Q3_1', marker='x')
plt.loglog(K, np.array(compute_time["Q3-2"]), label='Q3_2', marker='s')
plt.loglog(K, np.array(compute_time["Q3-3"]), label='Q3_3', marker='^')

plt.xlabel("K (Input Size)", fontsize=12)
plt.ylabel("Compute Time With Unified memory (ms)", fontsize=12)
plt.title("Log-Log Plot of Compute Time With Unified memory vs K", fontsize=14)

plt.grid(True)

plt.legend()

plt.show()