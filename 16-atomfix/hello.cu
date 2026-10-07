// The fix for 15-atomerr: atomicAdd does the read-modify-write in one step.
// The GPU result now differs from the CPU reference only by floating-point summation order.
#include <cstdio>
#include <cmath>
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
    checkCudaErrors(cudaMallocManaged(&arr, n * sizeof(int)));

    int block_dim = 128;
    int grid_dim = (n + block_dim - 1) / block_dim;
    kernel<<<grid_dim, block_dim>>>(n, [=] __device__ (int i) {
        arr[i] = i;
    });
    kernel<<<grid_dim, block_dim>>>(n, [=] __device__ (int i) {
        atomicAdd(&sum, sinf(arr[i]));
    });
    checkCudaErrors(cudaGetLastError());

    float result = 0;
    checkCudaErrors(cudaMemcpyFromSymbol(&result, sum, sizeof(float)));
    printf("GPU %f\n", result);

    double reference = 0;
    for (int i = 0; i < n; ++i)
        reference += sinf(i);
    printf("CPU %f\n", reference);

    checkCudaErrors(cudaFree(arr));
    return 0;
}
