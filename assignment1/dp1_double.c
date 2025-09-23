#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void init(long N, double **pA, double **pB) {
	*pA = malloc(N * sizeof(double));
	*pB = malloc(N * sizeof(double));

	srand(time(NULL));
	for (int i = 0; i < N; i++) {
		(*pA)[i] = 1.0f;
		(*pB)[i] = 1.0f;
	}

}
double dp(long N, double *pA, double *pB) {
	double R = 0.0;
	int j;
	for (j = 0; j < N; j++) 
		R += pA[j] * pB[j];
     
	return R;
}

void evaluate(int N, int reps) {
    double *pA, *pB;

	init(N, &pA, &pB);

    double average_exe_time = 0.0f;
    double res = 0.0f;

    struct timespec start, end;


    for (int i = 0; i < reps; i++) {
        clock_gettime(CLOCK_MONOTONIC, &start);

	    res = dp(N, pA, pB);

        clock_gettime(CLOCK_MONOTONIC, &end);
        
        double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        if (i >= reps / 2) {
            average_exe_time += elapsed / (reps / 2);     
        }
    }
    double bandwidth = (double)8 * N / average_exe_time / 1e9;// GB/sec
    double throughput = (double)(2  * N) / average_exe_time;// FLOP/sec
    
    printf("result: %f\n", res);
    printf("N %d <T>: %f sec B: %f GB/sec F: %f FLOP/sec \n", N, average_exe_time, bandwidth, throughput);

	free(pA); free(pB);
}

int main(int argc, char* argv[]) {
    int N = atoi(argv[1]);
    int reps = atoi(argv[2]);
	
	evaluate(N, reps);
}
