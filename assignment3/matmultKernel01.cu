///
/// matmultKernel01.cu
/// Modified by Yankai Mao

#include "matmultKernel.h"
#include <stdio.h>

#define VALUE_NUMBER 4
// Magic number 4 means every thread computes 4 values
// Define a gpu kernel to perform matrix multiplication
// of A x B = C.

__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C){

  // matrix blocks
  float *Asub, *Bsub, *Csub;
  // Putting these into registers speeds access.
  int thread_id = threadIdx.y * BLOCK_SIZE + threadIdx.x;
  int thread_row = thread_id / (FOOTPRINT_SIZE/VALUE_NUMBER);
  int thread_col = thread_id % (FOOTPRINT_SIZE/VALUE_NUMBER);
  int block_row = blockIdx.y;
  int block_col = blockIdx.x;

  // Each THREAD BLOCK computes one sub matrix Csub of C
  // EACH THREAD creates its own matrix descriptor Csub
  Csub = &C.elements[C.stride * block_row * FOOTPRINT_SIZE + FOOTPRINT_SIZE * block_col]; // C[FOOTPRINT_SIZE * block_row][FOOTPRINT_SIZE * block_col]
  // Each thread computes one element of Csub in its copy of CValue
  float Cvalue[4] = {0.0f};

  // Loop over all sub matrices in block_row of A and block_col of B
  // required to compute Csub. Block multiply each pair of sub matrices
  // and accumulate results
  for (int m = 0;  m < (A.width / FOOTPRINT_SIZE); ++m){
    // Get Asub and Bsub descriptors
    Asub = &A.elements[A.stride * block_row * FOOTPRINT_SIZE + FOOTPRINT_SIZE * m]; // A[FOOTPRINT_SIZE * block_row][FOOTPRINT_SIZE * m]
    Bsub = &B.elements[B.stride * FOOTPRINT_SIZE * m + FOOTPRINT_SIZE * block_col]; // B[FOOTPRINT_SIZE * m][FOOTPRINT_SIZE * block_col]

    // Copy ELEMENTS OF  ASub and Bsub into shared memory
    // EACH THREAD loads ONE ELEMENT of ASub and ONE of Bsub
    // Notice: it does not need to be the element it requires to
    //         compute its Cvalue, as long as all elements are 
    //         collaboratively read. 

    // Notice: every thread declares shared_A and shared_B in shared memory
    //         even though a thread block has only one shared_A and one shared_B
    __shared__ float shared_A[FOOTPRINT_SIZE][FOOTPRINT_SIZE];
    __shared__ float shared_B[FOOTPRINT_SIZE][FOOTPRINT_SIZE];

    // Each thread copies just one element of shared_A and one element of shared_B
#pragma unroll
    for (int i = 0; i < VALUE_NUMBER; i++) {
        shared_A[thread_row][thread_col*VALUE_NUMBER+i] = Asub[thread_row * A.stride + thread_col*VALUE_NUMBER+i];
        shared_B[thread_row][thread_col*VALUE_NUMBER+i] = Bsub[thread_row * B.stride + thread_col*VALUE_NUMBER+i];
    }
    // Synchronize to ensure all elements are read
    __syncthreads();

    // Do an inproduct of one row of shared_A and one col of shared_B
    // computing one Cvalue by accumulation
#pragma unroll
    for (int i = 0; i < VALUE_NUMBER; i++) {
        #pragma unroll
        for(int e=0; e<FOOTPRINT_SIZE; ++e)
            Cvalue[i] += shared_A[thread_row][e] * shared_B[e][thread_col*VALUE_NUMBER+i];
    }
    // Synchronize to ensure all Cvalues have been incremented
    // before reading in the next shared_A AND shared_B BLOCKS
    __syncthreads();
  }

  // Write Csub to GLOBAL memory.
  // Each thread writes its own cell value.
#pragma unroll
  for (int i = 0; i < VALUE_NUMBER;i ++) {
    Csub[thread_row*C.stride + thread_col * VALUE_NUMBER + i] = Cvalue[i];
  }
}

