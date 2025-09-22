import time
import numpy as np
import argparse

def dp(N, A, B):
    R = np.dot(A, B)
    return R

def evaluate(N, reps):   
    A = np.ones(N, dtype = np.float32)
    B = np.ones(N, dtype = np.float32)

    average_time = 0.0
    result = 0.0
    for i in range(0, reps):
        t1 = time.clock_gettime(time.CLOCK_MONOTONIC)

        result = dp(N, A, B)

        t2 = time.clock_gettime(time.CLOCK_MONOTONIC)

        elapsed = t2 - t1
        if (i >= reps / 2):
            average_time = average_time + elapsed / (reps / 2)

    print(f"result: {result}")        
    bandwith = 8.0 * N / average_time / 1e9
    throughput = 2.0 * N / average_time
    print(f"N: {N} <T>: {average_time} sec B: {bandwith} GB/sec F: {throughput} FLOP/sec")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("N", type=int, help="N")
    parser.add_argument("reps", type=int, help="reps")
    args = parser.parse_args()

    evaluate(args.N, args.reps)
   # evaluate(20, 20)
