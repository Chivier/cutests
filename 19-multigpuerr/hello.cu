#include <stdio.h>
#include <cuda_runtime.h>

float p2p_copy (size_t size) {
    int gpu_numbers = 100;
    int *pointers[gpu_numbers];
    
    for (int index = 0; index < gpu_numbers; ++index) {
        cudaSetDevice(index);
        cudaMalloc(&pointers[index], size);
    }

    for (int indexi = 0; indexi < gpu_numbers; ++indexi) {
        cudaSetDevice(indexi);
        for (int indexj = 0; indexj < gpu_numbers; ++indexj) {
            if (indexi == indexj)
                continue;
            cudaDeviceEnablePeerAccess(indexj, 0);
        }
    }
    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);
  
    cudaEventRecord(begin);
    for (int TTT = 100; TTT >= 0; --TTT)
    for (int index = 1; index < gpu_numbers; ++index) {
        cudaMemcpyAsync(pointers[0], pointers[index], size, cudaMemcpyDeviceToDevice);
    }
    cudaEventRecord(end);
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
    auto time = p2p_copy(1000000000);
    printf("time = %f s\n", time);
    return 0;
}

