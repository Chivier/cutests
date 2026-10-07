// 06-memerr1 with error checks: the illegal access is now reported.
//   CUDA error at .../hello.cu:NN code=700(cudaErrorIllegalAddress) "cudaDeviceSynchronize()"
// On HMM/ATS systems the access is legal and the program prints 55 (see 06-memerr1).
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
    int *a = (int *)calloc(12, sizeof(int));
    for (int i = 1; i <= 10; ++i)
        a[i] = i;

    kernel<<<1, 1>>>(a);                       // host pointer
    checkCudaErrors(cudaGetLastError());       // launch errors (bad configuration, ...)
    checkCudaErrors(cudaDeviceSynchronize());  // errors raised while the kernel ran
    printf("%d\n", a[0]);

    free(a);
    return 0;
}
