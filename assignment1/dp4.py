import time
import numpy as np

def dp(N, A, B):
    R = 0.0
    for j in range(0, N):
        R += A[j] * B[j]
    return R

def evaluate(N, reps):   
    A = np.ones(N, dtype = np.float32)
    B = np.ones(N, dtype = np.float32)

    average_time = 0.0
    for i in range(0, reps):
        t1 = time.clock_gettime(time.CLOCK_MONOTONIC)

        dp(N, A, B)

        t2 = time.clock_gettime(time.CLOCK_MONOTONIC)

        elapsed = t2 - t1
        if (i >= reps / 2):
            average_time = average_time + elapsed / (reps / 2)
    bandwith = 8.0 * N / average_time / 1e9
    throughput = 2.0 * N / average_time
    print(f"N: {N} <T>: {average_time} sec B: {bandwith} GB/sec F: {throughput} FLOP/sec")


if __name__ == '__main__':
    evaluate(1000000, 1000)
    evaluate(300000000, 20)
   # evaluate(20, 20)
