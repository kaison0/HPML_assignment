// @ym3121
// cudnn v8.0+
// Tile convolution kernel or not?
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <iomanip>
#include <cudnn.h>

#define H 1024
#define W 1024
#define C 3
#define FW 3
#define FH 3
#define K 64
#define P 1
#define verbose 0
#define block_x 16 // block_x * block_y has to be greater than K
#define block_y 16
#define block_z 1 

#define idx_I(c, x, y) ((c) * (H) * (W) + (x) * (W) + y)
#define idx_F(k, c, i, j) ((k) * C * FH * FW + (c) * FH * FW + (i) * FW + j)
#define idx_O(k, x, y) ((k) * H * W + (x) * W + y)


int main(int argc, char* argv[]) {
    // one block is thread_num*thread_num
 //   int block_x, block_y, block_z;
//    if (argc == 4) {
 //       block_x = std::stoi(argv[1]);
 //       block_y = std::stoi(argv[2]);
  //      block_z = std::stoi(argv[3]);
  //  }
    float *I, *F, *O, *O_correct;
    long I_bytesize = (long)C*H*W* sizeof(float);
    long F_bytesize = (long)K*C*FH*FW* sizeof(float);
    long O_bytesize = (long)K*H*W* sizeof(float);
    I = (float*)std::malloc(I_bytesize);
    F = (float*)std::malloc(F_bytesize);
    O = (float*)std::malloc(O_bytesize);
    O_correct = (float*)std::malloc(O_bytesize);
    for (int c = 0; c < C; c++)
        for (int h = 0; h < H; h++)
            for (int w = 0; w < W; w++)
                I[idx_I(c, h ,w)] = (float)c * (h + w);

    for (int k = 0; k < K; k++)
        for (int c = 0; c < C; c++)
            for (int i = 0; i < FH; i++)
                for (int j = 0; j < FW; j++)
                    F[idx_F(k, c, i, j)] = (float)(c + k) * (i + j);

    float * I_device, *F_device, *O_device;
    cudaMalloc(&I_device, I_bytesize);
    cudaMalloc(&F_device, F_bytesize);
    cudaMalloc(&O_device, O_bytesize);

    cudaMemcpy(I_device, I, I_bytesize, cudaMemcpyHostToDevice);
    cudaMemcpy(F_device, F, F_bytesize, cudaMemcpyHostToDevice);
    cudaMemset(O_device, 0, O_bytesize);

    for (int k = 0; k < K; k++)
        for (int x = 0; x < W; x++)
            for (int y = 0; y < H; y++) {
                O_correct[idx_O(k, x, y)] = 0.0;
                for (int c = 0; c < C; c++)
                    for (int j = 0; j < FH; j++)
                        for (int i = 0; i < FW; i++)
                        {
                            int ix = x + i - P, iy = y + j - P;
                            if (ix >= 0 && ix < W && iy >= 0 && iy < H) {
                                O_correct[idx_O(k, x, y)] += F[idx_F(k, c, FW - 1 - i, FH - 1 - j)] * I[idx_I(c, ix, iy)];
                            }
                        
                        }
            }

    cudnnHandle_t handle;
    cudnnCreate(&handle);

    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    cudnnCreateTensorDescriptor(&input_desc);
    cudnnCreateTensorDescriptor(&output_desc);
    cudnnCreateFilterDescriptor(&filter_desc);
    cudnnCreateConvolutionDescriptor(&conv_desc);

    cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, C, H, W);
    cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, K, C, FH, FW);
    cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, K, H, W);
    cudnnSetConvolution2dDescriptor(conv_desc, 1, 1, 1, 1, 1, 1, CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT);

    cudnnConvolutionFwdAlgoPerf_t perf_results[8];
    int returned_algo_count = 0;

    cudnnGetConvolutionForwardAlgorithm_v7(
        handle,
        input_desc,
        filter_desc,
        conv_desc,
        output_desc,
        8,
        &returned_algo_count,
        perf_results);

    cudnnConvolutionFwdAlgo_t algo = perf_results[0].algo;
    // ---------------------------------------------------

    size_t ws_size = 0;
    cudnnGetConvolutionForwardWorkspaceSize(
        handle, input_desc, filter_desc, conv_desc, output_desc,
        algo,
        &ws_size);
   
    void *d_workspace = nullptr;
    printf("workspace size %d\n", ws_size);
    cudaMalloc(&d_workspace, ws_size);

    float alpha = 1.0;
    float beta  = 0.0;
   // cudnnConvolutionForward(handle, &alpha, input_desc, I_device, filter_desc, F_device, conv_desc, algo, d_workspace, ws_size, &beta, output_desc, O_device);
    
    for (int i = 0; i < 50; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        cudnnConvolutionForward(handle, &alpha, input_desc, I_device, filter_desc, F_device, conv_desc, algo, d_workspace, ws_size, &beta, output_desc, O_device);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<float, std::milli> duration = end - start;
        std::cout << " Kernel Running Time: " << std::fixed << std::setprecision(5) << duration.count() << "ms\n";
    }
    cudaMemcpy(O, O_device, O_bytesize, cudaMemcpyDeviceToHost);
    
    bool pass_test = true;
    for (int k = 0; k < K; k++) {
        for (int x = 0; x < W; x++)
            for (int y = 0; y < H;y++)
                if (std::fabs(O_correct[idx_O(k, x, y)] - O[idx_O(k, x, y)]) > 1e-6) {
                    if (verbose)
                    std::cout << "Element (" << k << "," << x << "," << y << ") is wrong correct: " << O_correct[idx_O(k, x, y)] << " result: " << O[idx_O(k, x, y)] << std::endl;
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
    long check_sum = 0;
    long correct_sum = 0;
    for (int k = 0; k < K; k++)
        for (int x = 0; x < W; x++)
            for (int y = 0; y < H; y++) {
                check_sum += O[idx_O(k, x, y)];
                correct_sum += O_correct[idx_O(k, x, y)];
            }
    
    std::cout << " Check sum: " << check_sum << " Correct sum: " << correct_sum << std::endl;
    free(I);free(F);free(O);free(O_correct);
    cudaFree(d_workspace);
    cudnnDestroy(handle);
}
