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
    checkCudaErrors(cudaMallocManaged(&a, sizeof(int) * 12));
    int index = 1;
    for (index = 1; index <= 10; ++index) {
        a[index] = index;
    }

    kernel<<<1, 1>>>(a);
    
    checkCudaErrors(cudaDeviceSynchronize());
    printf("%d\n", a[0]);
    cudaFree(a);
    return 0;
}

