// Managed (unified) memory: one pointer valid on host and device. Prints 55.
#include <cstdio>
#include <cuda_runtime.h>
#include "helper_cuda.h"

__global__ void kernel(int *arr) {
    arr[0] = 0;
    for (int i = 1; arr[i] != 0; ++i)
        arr[0] += arr[i];
}

int main() {
    int *a;
    checkCudaErrors(cudaMallocManaged(&a, sizeof(int) * 12));
    for (int i = 1; i <= 10; ++i)
        a[i] = i;
    a[11] = 0;   // cudaMallocManaged does not zero memory; the kernel loop stops at 0

    kernel<<<1, 1>>>(a);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());   // no implicit copy: wait before the host reads
    printf("%d\n", a[0]);

    checkCudaErrors(cudaFree(a));
    return 0;
}
