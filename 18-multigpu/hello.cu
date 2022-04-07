#include <iostream>
#include <cuda_runtime.h>

float p2p_copy (size_t size) {
    int *pointers[2];
  
    cudaSetDevice (0);
    cudaDeviceEnablePeerAccess (1, 0);
    cudaMalloc (&pointers[0], size);
  
    cudaSetDevice (1);
    cudaDeviceEnablePeerAccess (0, 0);
    cudaMalloc (&pointers[1], size);
  
    cudaEvent_t begin, end;
    cudaEventCreate (&begin);
    cudaEventCreate (&end);
  
    cudaEventRecord (begin);
    cudaMemcpyAsync (pointers[0], pointers[1], size, cudaMemcpyDeviceToDevice);
    cudaEventRecord (end);
    cudaEventSynchronize (end);
  
    float elapsed;
    cudaEventElapsedTime (&elapsed, begin, end);
    elapsed /= 1000;
  
    cudaSetDevice (0);
    cudaFree (pointers[0]);
  
    cudaSetDevice (1);
    cudaFree (pointers[1]);
  
    cudaEventDestroy (end);
    cudaEventDestroy (begin);
  
    return elapsed;
}

int main() {
    auto time = p2p_copy(1000000);
    printf("time = %f\n", time);
    return 0;
}

