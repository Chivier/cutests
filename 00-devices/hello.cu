#include <iostream>
#include <cuda_runtime.h>

__device__ void gpu_hello() {
    printf("gpu hello!\n");
}

__host__ void cpu_hello() {
    printf("cpu hello!\n");
}

__global__ void kernel() {
    gpu_hello();
}

int main() {
    // kernel<<<1, 2>>>();
    kernel();
    cudaDeviceSynchronize();
    cpu_hello();
    return 0;
}

