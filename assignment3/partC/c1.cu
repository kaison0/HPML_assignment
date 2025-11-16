#include <iostream>
#include <cstdlib>
#include <chrono>
#include <iomanip>

#define H 1024
#define W 1024
#define C 3
#define FW 3
#define FH 3
#define K 64
#define P 1
#define verbose 0

#define idx_I(c, x, y) ((c) * (H+2*P) * (W+2*P) + (x) * (W+2*P) + y)
#define idx_F(k, c, i, j) ((k) * C * FH * FW + (c) * FH * FW + (i) * FW)
#define idx_O(k, x, y) ((k) * H * W + (x) * W + y)

__global__ void compute_convolution(double* O, double* I, double* F) {
    int y = threadIdx.x + blockIdx.x * blockDim.x;
    int x = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;
//    printf("%d %d %d\n", x, y, k);
    //for (int k = 0; k < K; k++) {
        double value = 0.0;
        for (int c = 0; c < C; c++) {
            for (int i = 0; i < FH; i++)
                for (int j = 0; j < FW; j++)
                    value += F[idx_F(k, c, FW - 1 - i, FH - 1 - j)] * I[idx_I(c, x+i, y+j)];
        }

        O[idx_O(k, x, y)] = value;
    //}
}

int main(int argc, char* argv[]) {
    // one block is thread_num*thread_num
    int block_x, block_y, block_z;
    if (argc == 4) {
        block_x = std::stoi(argv[1]);
        block_y = std::stoi(argv[2]);
        block_z = std::stoi(argv[3]);
    }
    double *I, *F, *O, *O_correct;
    long I_bytesize = (long)C*(H+2*P)*(W+2*P)* sizeof(double);
    long F_bytesize = (long)K*C*FH*FW* sizeof(double);
    long O_bytesize = (long)K*H*W* sizeof(double);
    I = (double*)std::malloc(I_bytesize);
    F = (double*)std::malloc(F_bytesize);
    O = (double*)std::malloc(O_bytesize);
    O_correct = (double*)std::malloc(O_bytesize);
    for (int c = 0; c < C; c++)
        for (int h = 0; h < H; h++)
            for (int w = 0; w < W; w++)
                I[idx_I(c, h+1 ,w+1)] = (double)c * (h + w);
    for (int c = 0; c < C; c++) {
        for (int h = 0; h < H + 2*P; h++) {
            I[idx_I(c, h, 0)] = 0.0;
            I[idx_I(c, h, W + 2*P - 1)] = 0.0;
        }
        for (int w = 0; w < W + 2*P; w++) {
            I[idx_I(c, 0, w)] = 0.0;
            I[idx_I(c, H + 2*P - 1, w)] = 0.0;
        }
    }

    for (int k = 0; k < K; k++)
        for (int c = 0; c < C; c++)
            for (int i = 0; i < H; i++)
                for (int j = 0; j < W; j++)
                    F[idx_F(k, c, i, j)] = (double)(c + k) * (i + j);

    double * I_device, *F_device, *O_device;
    cudaMalloc(&I_device, I_bytesize);
    cudaMalloc(&F_device, F_bytesize);
    cudaMalloc(&O_device, O_bytesize);

    cudaMemcpy(I_device, I, I_bytesize, cudaMemcpyHostToDevice);
    cudaMemcpy(F_device, F, F_bytesize, cudaMemcpyHostToDevice);
    cudaMemcpy(O_device, O, O_bytesize, cudaMemcpyHostToDevice);

    for (int k = 0; k < K; k++)
        for (int x = 0; x < W; x++)
            for (int y = 0; y < H; y++) {
                O_correct[idx_O(k, x, y)] = 0.0;
                for (int c = 0; c < C; c++)
                    for (int j = 0; j < FH; j++)
                        for (int i = 0; i < FW; i++)
                            O_correct[idx_O(k, x, y)] += F[idx_F(k, c, FW - 1 - i, FH - 1 - j)] * I[idx_I(c, x + i, y + j)];
            }
    
    auto start = std::chrono::high_resolution_clock::now();

    dim3 dim_block(block_x, block_y, block_z);
    dim3 dim_grid(H/block_x, W/block_y, K/block_z);
    compute_convolution<<<dim_grid, dim_block>>>(O_device, I_device, F_device);
    
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Block_size: (" << block_x << ", " << block_y << ", " << block_z << ") Kernel Running Time: " << std::fixed << std::setprecision(5) << duration.count() << "ms\n";

    cudaMemcpy(O, O_device, O_bytesize, cudaMemcpyDeviceToHost);
    
    bool pass_test = true;
    for (int k = 0; k < K; k++) {
        for (int x = 0; x < W; x++)
            for (int y = 0; y < H;y++)
                if (std::fabs(O_correct[idx_O(k, x, y)] - O[idx_O(k, x, y)]) > 1e-6) {
                    std::cout << "Element (" << k << "," << x << "," << y << ") is wrong" << std::endl;
                    pass_test = false;
                }
    }
    std::cout << (pass_test ? "Test passed!" : "Test Failed (") << std::endl;
    if (verbose) {
        std::cout << "correct: " << std::endl;
        for (int x = 0; x < W; x++) {
            for (int y = 0; y < H; y++) {
                std::cout << O_correct[idx_O(0, x, y)] << " ";
            }
            std::cout << std::endl;
        }

        std::cout << "result: " << std::endl;
        for (int x = 0; x < W; x++) {
            for (int y = 0; y < H; y++) {
                std::cout << O[idx_O(0, x, y)] << " ";
            }
            std::cout << std::endl;
        }
    }

    free(I);free(F);free(O);free(O_correct);
    cudaFree(I_device);cudaFree(F_device);cudaFree(O_device);
}
