// Template for a new example: copy this directory.
#include <cstdio>
#include <cuda_runtime.h>

__global__ void kernel() {
#ifdef __CUDA_ARCH__
    printf("__CUDA_ARCH__ = %d\n", __CUDA_ARCH__);
#endif
}

int main() {
    kernel<<<1, 1>>>();
    cudaError_t err = cudaGetLastError();   // launch errors
    if (err == cudaSuccess)
        err = cudaDeviceSynchronize();      // errors raised while the kernel ran
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}
