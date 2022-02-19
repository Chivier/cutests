#include <iostream>
#include <cuda_runtime.h>

using namespace std;
__global__ void kernel() {
    printf("Thread %d of %d\n",
           blockIdx.x * blockDim.x + threadIdx.x, blockDim.x * gridDim.x);
}

int main() {
    kernel<<<4, 3>>>();
    cudaDeviceSynchronize();
    return 0;
}

