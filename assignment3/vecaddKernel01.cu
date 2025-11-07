// @ym3121 
// Add vectors with memory coalesc
__global__ void AddVectors(const float* A, const float* B, float* C, int N)
{
     int threadNum  = gridDim.x * blockDim.x;
     int threadStartId = blockIdx.x * blockDim.x + threadIdx.x; 
     int i;
     
     for( i = threadStartId; i < threadNum * N ; i += threadNum ){
         C[i] = A[i] + B[i];
     }
}
