#include <cstdio>
#include <cuda_runtime.h>
#include "helper_cuda.h"

__device__ float sum = 0;

template <class Func>
__global__ void kernel(int n, Func func) {
    for (int i = blockDim.x * blockIdx.x + threadIdx.x;
         i < n; i += blockDim.x * gridDim.x) {
        func(i);
    }
}

int main() {
    int n = 65536;
    int *arr;
    float result = 0;

    cudaMallocManaged(&arr, n * sizeof(int));

    int block_dim = 128;
    int grid_dim = (n - 1) / block_dim;
    kernel<<<grid_dim, block_dim>>>(n, [=] __device__ (int i) {
        arr[i] = i;
    });
    
    
    kernel<<<grid_dim, block_dim>>>(n, [=] __device__ (int i) {
        sum += sinf(arr[i]);
    });

    cudaMemcpyFromSymbol(&result, sum, sizeof(float), 0, cudaMemcpyDeviceToHost);
    checkCudaErrors(cudaDeviceSynchronize());
    
    printf("%f\n", result);

    // Compare
    result = 0;
    for(int index = 0; index < n; ++index) {
        result += sinf(index);
    }
    printf("%f", result);

    cudaFree(arr);
    return 0;
}
