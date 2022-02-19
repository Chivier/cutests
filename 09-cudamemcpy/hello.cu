#include <iostream>
#include <cuda_runtime.h>
#include "helper_cuda.h"

__global__ void kernel(int *arr) {
    arr[0] = 0;
    int index = 1;
    while (arr[index] != 0) {
        arr[0] += arr[index];
        index++;
    }
}

int main() {
    int *a;
    a = (int *)malloc(sizeof(int) * 12);
    int index = 1;
    for (index = 1; index <= 10; ++index) {
        a[index] = index;
    }

    int *cuda_a;
    cudaMalloc(&cuda_a, sizeof(int) * 12);
    cudaMemcpy(cuda_a, a, sizeof(int) * 12, cudaMemcpyHostToDevice);
    kernel<<<1, 1>>>(cuda_a);
    cudaMemcpy(a, cuda_a, sizeof(int) * 12, cudaMemcpyDeviceToHost);

    checkCudaErrors(cudaDeviceSynchronize());
    printf("%d\n", a[0]);
    free(a);
    cudaFree(cuda_a);
    return 0;
}

