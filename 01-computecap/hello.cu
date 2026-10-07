// __CUDA_ARCH__ is the compile target of the device code that runs, not the GPU itself.
// Build for an older target (e.g. -DCMAKE_CUDA_ARCHITECTURES=75 on an sm_86 card) and the
// two lines differ: the driver JIT-compiles the compute_75 PTX for the newer GPU.
#include <cstdio>
#include <cuda_runtime.h>

__global__ void kernel() {
#ifdef __CUDA_ARCH__   // defined only while compiling device code
    printf("compiled for   __CUDA_ARCH__ = %d\n", __CUDA_ARCH__);
#endif
}

int main() {
    kernel<<<1, 1>>>();
    cudaError_t err = cudaGetLastError();   // launch errors, e.g. PTX newer than the driver
    if (err == cudaSuccess)
        err = cudaDeviceSynchronize();
    cudaDeviceProp prop;
    if (err == cudaSuccess)
        err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("running on     %s, compute capability %d.%d\n", prop.name, prop.major, prop.minor);
    return 0;
}
