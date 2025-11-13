#include <iostream>
#include <string>
#include <chrono>
#include <cstdlib>
#include <iomanip>

int main(int argc, char *argv[]) {
    int k = 0;

    if (argc == 2) {
        k = std::stoi(argv[1]) * 1000000;
    }
    
    int* a = (int*)std::malloc(k * sizeof(int));
    int* b = (int*)std::malloc(k * sizeof(int));
    int* c = (int*)std::malloc(k * sizeof(int));

    for (int i = 0; i < k; i++) {
        a[i] = i;
        b[i] = k - i;
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < k; i++) {
        c[i] = a[i] + b[i];
    }

    auto end = std::chrono::high_resolution_clock::now();

    free(a);
    free(b);
    free(c);
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "K: " << k <<  " Running Time: " << std::fixed << std::setprecision(5) << duration.count() << "ms\n";
}
