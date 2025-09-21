#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void init(long N, float **pA, float **pB) {
	*pA = malloc(N * sizeof(float));
	*pB = malloc(N * sizeof(float));

	srand(time(NULL));
	for (int i = 0; i < N; i++) {
		(*pA)[i] = (float)rand();
		(*pB)[i] = (float)rand();
	}

}
float dp(long N, float *pA, float *pB) {
	float R = 0.0;
	int j;
	for (j = 0; j < N; j++) 
		R += pA[j] * pB[j];
     
	return R;
}

void evaluate(int N, int reps) {
    float *pA, *pB;

	init(N, &pA, &pB);

    float average_exe_time = 0.0f;

    struct timespec start, end;


    for (int i = 0; i < reps; i++) {
        clock_gettime(CLOCK_MONOTONIC, &start);

	    dp(N, pA, pB);

        clock_gettime(CLOCK_MONOTONIC, &end);
        
        double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        if (i >= reps / 2) {
            average_exe_time += elapsed / (reps / 2);     
        }
    }
    float bandwidth = (float)8 * N / average_exe_time / 1e9;// GB/sec
    float throughput = (float)(2  * N) / average_exe_time;// FLOP/sec

    printf("N %d <T>: %f sec B: %f GB/sec F: %f FLOP/sec \n", N, average_exe_time, bandwidth, throughput);

	free(pA); free(pB);
}

int main() {
	float *pA, *pB;
	
	evaluate(1000000, 1000);
    evaluate(300000000, 20);

}
