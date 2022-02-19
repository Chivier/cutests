#include <cstdio>  
#include <cuda_runtime.h>  
#include "helper_cuda.h"  
  
template <class Func>  
__global__ void kernel(int n, Func func) {  
   for (int i = blockDim.x * blockIdx.x + threadIdx.x;  
        i < n; i += blockDim.x * gridDim.x) {  
       func(i);  
   }  
}  
  
int main() {  
   int n = 10;  
   int *arr;  
  
   cudaMallocManaged(&arr, n * sizeof(int));  
  
   int block_dim = 128;  
   int grid_dim = (n - 1) / block_dim + 1;  
   kernel<<<grid_dim, block_dim>>>(n, [=] __device__ (int i) {  
       arr[i] = i;  
   });  
      
   checkCudaErrors(cudaDeviceSynchronize());  
   kernel<<<grid_dim, block_dim>>>(n, [=] __device__ (int i) {  
       printf("%d, %f\n", i, sinf(arr[i]));  
   });  
  
   checkCudaErrors(cudaDeviceSynchronize());  
  
   // Compare  
   // for(int index = 0; index < n; ++index) {  
   //    printf("%d, %f\n", index, sinf(index));  
   //}  
  
   cudaFree(arr);  
   return 0;  
}
