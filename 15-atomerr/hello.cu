// A data race: every thread does sum += x on one __device__ variable.
// sum += x is load, add, store; threads interleave and overwrite each other, so the GPU
// result is wrong and changes from run to run. 16-atomfix and 17-customatom fix it.
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
        sum += sinf(arr[i]);   // race!
    });
    checkCudaErrors(cudaGetLastError());

    // A __device__ variable has a device address: read it with cudaMemcpyFromSymbol
    // (it waits for the kernels, like cudaMemcpy).
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
