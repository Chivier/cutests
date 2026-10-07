// Build your own atomic float add from atomicCAS (compare-and-swap).
// atomicCAS(addr, expect, desired) stores desired only if *addr == expect and always
// returns the old value. It works on integer types, so the float bits travel as an int.
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>
#include "helper_cuda.h"

__device__ float sum = 0;

// Same contract as atomicAdd(float*, float): adds src to *dst, returns the old value.
__device__ float my_atom_add(float *dst, float src) {
    int old = __float_as_int(*dst);
    int expect;
    do {
        expect = old;
        old = atomicCAS((int *)dst, expect,
                        __float_as_int(__int_as_float(expect) + src));
    } while (expect != old);   // another thread changed *dst first: retry
    return __int_as_float(old);
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
    checkCudaErrors(cudaMallocManaged(&arr, n * sizeof(int)));

    int block_dim = 128;
    int grid_dim = (n + block_dim - 1) / block_dim;
    kernel<<<grid_dim, block_dim>>>(n, [=] __device__ (int i) {
        arr[i] = i;
    });
    kernel<<<grid_dim, block_dim>>>(n, [=] __device__ (int i) {
        my_atom_add(&sum, sinf(arr[i]));
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
