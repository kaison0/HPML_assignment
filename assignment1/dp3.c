#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mkl_cblas.h>

void init(long N, float **pA, float **pB) {
	*pA = malloc(N * sizeof(float));
	*pB = malloc(N * sizeof(float));

	srand(time(NULL));
	for (int i = 0; i < N; i++) {
		(*pA)[i] = 1.0f;
		(*pB)[i] = 1.0f;
	}

}
float dp(long N, float *pA, float *pB) {
	float R = 0.0;
	int j;
	for (j = 0; j < N; j++) 
		R += pA[j] * pB[j];
     
	return R;
}

float dpunroll(long N, float *pA, float *pB) {
    float R = 0.0f;
    int j;

    for (j = 0; j < N; j += 4)
        R += pA[j] * pB[j] + pA[j + 1] * pB[j + 1] 
            + pA[j + 2] * pB[j + 2] + pA[j + 3] * pB[j + 3];

    return R;
}
float bdp(long N, float *pA, float *pB) {
    float R = cblas_sdot(N, pA, 1, pB, 1);
    
    return R;
}

void evaluate(int N, int reps) {
    float *pA, *pB;

	init(N, &pA, &pB);

    float average_exe_time = 0.0f;
    float result = 0.0f;

    struct timespec start, end;


    for (int i = 0; i < reps; i++) {
        clock_gettime(CLOCK_MONOTONIC, &start);

	    //dp(N, pA, pB);
        //dpunroll(N, pA, pB);
        result = bdp(N, pA, pB);

        clock_gettime(CLOCK_MONOTONIC, &end);
        
        double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        if (i >= reps / 2) {
            average_exe_time += elapsed / (reps / 2);     
        }
    }
    float bandwidth = (float)8 * N / average_exe_time / 1e9;// GB/sec
    float throughput = (float)(2  * N) / average_exe_time;// FLOP/sec
    printf("result: %f\n", result);
    printf("N %d <T>: %f sec B: %f GB/sec F: %f FLOP/sec \n", N, average_exe_time, bandwidth, throughput);

	free(pA); free(pB);
}

int main(int argc, char *argv[]) {
	int N = atoi(argv[1]);
    int reps = atoi(argv[2]);
	
	evaluate(N, reps);

}
