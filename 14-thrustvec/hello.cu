#include <cstdio>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

template <class Func>
__global__ void kernel(int n, Func func) {
    for (int i = blockDim.x * blockIdx.x + threadIdx.x;
         i < n; i += blockDim.x * gridDim.x) {
        func(i);
    }
}

int main() {
    int n = 65536;

    int block_dim = 128;
    int grid_dim = (n - 1) / block_dim;
    
    thrust::host_vector<float> x_host(n);
    thrust::host_vector<float> y_host(n);

    thrust::generate(x_host.begin(), x_host.end(), []{return std::rand() / 3.0;});
    thrust::generate(y_host.begin(), y_host.end(), []{return std::rand() / 11.0;});

    printf("%f + %f = \n", x_host[0], y_host[0]);

    thrust::device_vector<float> x_dev(n);
    thrust::device_vector<float> y_dev(n);
    x_dev = x_host;
    y_dev = y_host;

    kernel<<<grid_dim, block_dim>>>(n, [x = x_dev.data(), y = y_dev.data()] __device__ (int index){
        x[index] = x[index] + y[index];
    });

    checkCudaErrors(cudaDeviceSynchronize());
    x_host = x_dev;

    printf("%f\n", x_host[0]);

    return 0;
}

