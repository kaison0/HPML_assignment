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
    int *a, *b, *c;
    cudaMallocManaged(&a, byte_size);
    cudaMallocManaged(&b, byte_size);
    cudaMallocManaged(&c, byte_size);

    for (int i = 0; i < k; i++) {
        a[i] = i;
        b[i] = k - i;
    }

    auto start_compute = std::chrono::high_resolution_clock::now();
    switch (static_cast<int>(method_id)) {
        case 1: add_vec1<<<1, 1>>>(a, b, c, k);break;
        case 2: add_vec2<<<1, 256>>>(a, b, c, k);break;
        case 3: {
            int thread_num = 256;
           int block_num = (k + thread_num - 1) / thread_num;
            add_vec3<<<block_num, thread_num>>>(a, b, c, k);
            break;
        }
        default: break;
    }
    
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::chrono::duration<double, std::milli> duration_compute = end - start_compute;

    bool pass_test = true;
    for (int i = 0; i < k; i++)
        if (c[i] != k)
            pass_test = false;
    std::cout << (pass_test ? "Test passed" : "Test failed") << std::endl;
    std::cout << "K: " << k <<  " Running Time: " << std::fixed << std::setprecision(5) << duration.count() << "ms\n";
    std::cout << " Compute Running Time: " << std::fixed << std::setprecision(5) << duration_compute.count() << "ms\n";
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
}
