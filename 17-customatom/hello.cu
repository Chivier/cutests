#include <cstdio>
#include <cuda_runtime.h>
#include "helper_cuda.h"

__device__ float sum = 0;

__device__ float my_atom_add(float *dst, float src){
    int old = __float_as_int(*dst);
    int expect;
    do {
        expect = old;
        old = atomicCAS((int *)dst, expect,
                __float_as_int(__int_as_float(expect) + sinf(src)));
    } while(expect != old);
    return old;
}

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
        my_atom_add(&sum, arr[i]);
    });

    cudaMemcpyFromSymbol(&result, sum, sizeof(float), 0, cudaMemcpyDeviceToHost);
    checkCudaErrors(cudaDeviceSynchronize());
    
    printf("%f\n", result);

    // Compare
    result = 0;
    for(int index = 0; index < n; ++index) {
        result += sinf(index);
    }
    printf("%f\n", result);

    cudaFree(arr);
    return 0;
}
