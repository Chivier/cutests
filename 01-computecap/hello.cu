#include <iostream>
#include <cuda_runtime.h>

__device__ void gpu_hello() {
    #ifdef __CUDA_ARCH__
        printf("%d\n", __CUDA_ARCH__);
    #endif
}

__global__ void kernel() {
    gpu_hello();
}

int main() {
    kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}

