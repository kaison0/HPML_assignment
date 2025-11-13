#include <iostream>
#include <string>
#include <chrono>
#include <cstdlib>
#include <iomanip>



__global__ void add_vec1(int* A, int* B, int* C, int size) {
    for (int i = 0; i < size; i++)
        C[i] = A[i] + B[i];
}


__global__ void add_vec2(int* A, int* B, int* C, int size) {
    for (int i = threadIdx.x; i < size; i += blockDim.x)
        C[i] = A[i] + B[i];
}

__global__ void add_vec3(int* A, int* B, int* C, int size) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size)
        C[id] = A[id] + B[id];
}



int main(int argc, char *argv[]) {
    auto start = std::chrono::high_resolution_clock::now();
    int k = 0;
    int method_id = 1;

    if (argc == 3) {
        k = std::stoi(argv[1]) * 1000000;
        method_id = std::stoi(argv[2]);
    }
    
    size_t byte_size = k * sizeof(int);
    int* a = (int*)std::malloc(byte_size);
    int* b = (int*)std::malloc(byte_size);
    int* c = (int*)std::malloc(byte_size);

    for (int i = 0; i < k; i++) {
        a[i] = i;
        b[i] = k - i;
    }
    int* a_device;
    int* b_device;
    int* c_device;
    cudaMalloc(&a_device, byte_size);
    cudaMalloc(&b_device, byte_size);
    cudaMalloc(&c_device, byte_size);

    cudaMemcpy(a_device, a, byte_size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_device, b, byte_size, cudaMemcpyHostToDevice);

    switch (method_id) {
        case 1: add_vec1<<1, 1>>(a_device, b_device, c_device);break;
        case 2: add_vec2<<1, 256>>(a_device, b_device, c_device);break;
        case 3: {
            int block_num = (k + thread_num - 1) / thread_num;
            add_vec<<<block_num, thread_num>>>(a_device, b_device, c_device, k);
            break;
        }
    }

    cudaMemcpy(c, c_device, byte_size, cudaMemcpyDeviceToHost);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    bool pass_test = true;
    for (int i = 0; i < k; i++)
        if (c[i] != k)
            pass_test = false;
    std::cout << (pass_test ? "Test passed" : "Test failed") << std::endl;
    std::cout << "K: " << k <<  " Running Time: " << std::fixed << std::setprecision(5) << duration.count() << "ms\n";

    free(a);
    free(b);
    free(c);
    cudaFree(a_device);
    cudaFree(b_device);
    cudaFree(c_device);
}
