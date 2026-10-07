// The fix for 06/08: copy the data to device memory, run, copy the result back. Prints 55.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "helper_cuda.h"

__global__ void kernel(int *arr) {
    arr[0] = 0;
    for (int i = 1; arr[i] != 0; ++i)
        arr[0] += arr[i];
}

int main() {
    size_t bytes = 12 * sizeof(int);
    int *a = (int *)calloc(12, sizeof(int));   // a[11] == 0 ends the kernel loop
    for (int i = 1; i <= 10; ++i)
        a[i] = i;

    int *cuda_a;
    checkCudaErrors(cudaMalloc(&cuda_a, bytes));
    checkCudaErrors(cudaMemcpy(cuda_a, a, bytes, cudaMemcpyHostToDevice));
    kernel<<<1, 1>>>(cuda_a);
    checkCudaErrors(cudaGetLastError());
    // cudaMemcpy waits for the kernel (same default stream): no cudaDeviceSynchronize needed.
    checkCudaErrors(cudaMemcpy(a, cuda_a, bytes, cudaMemcpyDeviceToHost));
    printf("%d\n", a[0]);

    free(a);
    checkCudaErrors(cudaFree(cuda_a));
    return 0;
}
