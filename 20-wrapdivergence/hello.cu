#include <cstdio>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <chrono>

using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::seconds;
using std::chrono::system_clock;

template <class Func>
__global__ void kernel(int n, Func func) {
    for (int i = blockDim.x * blockIdx.x + threadIdx.x;
         i < n; i += blockDim.x * gridDim.x) {
        func(i);
    }
}

template <class Func1, class Func2>
__global__ void kernel_split(int n, Func1 func1, Func2 func2) {
    if (threadIdx.x & 2 == 1) {
        for(int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
            func1(i);
        }
    } else {
        for(int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
            func2(i);
        }
    }
}

template <class Func1, class Func2>
__global__ void kernel_better(int n, Func1 func1, Func2 func2) {
    for (int i = 0; i < n; i += 2) {
        func2(i);
    }
    for (int i = 1; i < n; i += 2) {
        func1(i);
    }
}

int main() {
    int n = 1 << 26;

    int block_dim = 128;
    int grid_dim = (n - 1) / block_dim;

    cudaDeviceSynchronize(); 
    auto begin_millis = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    thrust::host_vector<float> x_host(n);
    thrust::host_vector<float> y_host(n);

    thrust::generate(x_host.begin(), x_host.end(), []{return std::rand() / 3.0;});
    thrust::generate(y_host.begin(), y_host.end(), []{return std::rand() / 11.0;});

    
    thrust::device_vector<float> x_dev(n);
    thrust::device_vector<float> y_dev(n);
    x_dev = x_host;
    y_dev = y_host;

    kernel<<<grid_dim, block_dim>>>(n, [x = x_dev.data(), y = y_dev.data()] __device__ (int index){
        if (index % 2 == 1)
            x[index] = x[index] + y[index];
        else
            x[index] = x[index] - y[index];
    });

    checkCudaErrors(cudaDeviceSynchronize());
    x_host = x_dev;

    auto end_millis = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    cudaDeviceSynchronize(); 
    printf("%ld\n", end_millis - begin_millis);
    
    cudaDeviceSynchronize(); 
    begin_millis = end_millis;
    
    thrust::generate(x_host.begin(), x_host.end(), []{return std::rand() / 3.0;});
    thrust::generate(y_host.begin(), y_host.end(), []{return std::rand() / 11.0;});
    
    x_dev = x_host;
    y_dev = y_host;
    kernel_split<<<grid_dim, block_dim>>>(n,
            [x = x_dev.data(), y = y_dev.data()] __device__ (int index) {
                x[index] = x[index] + y[index];
            },
            [x = x_dev.data(), y = y_dev.data()] __device__ (int index) {
                x[index] = x[index] - y[index];
            });

    checkCudaErrors(cudaDeviceSynchronize());
    x_host = x_dev;
    end_millis = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    cudaDeviceSynchronize(); 
    printf("%ld\n",end_millis - begin_millis);
    
    cudaDeviceSynchronize(); 
    begin_millis = end_millis;
    
    thrust::generate(x_host.begin(), x_host.end(), []{return std::rand() / 3.0;});
    thrust::generate(y_host.begin(), y_host.end(), []{return std::rand() / 11.0;});
    
    x_dev = x_host;
    y_dev = y_host;
    kernel_better<<<grid_dim, block_dim>>>(n,
            [x = x_dev.data(), y = y_dev.data()] __device__ (int index) {
                x[index] = x[index] + y[index];
            },
            [x = x_dev.data(), y = y_dev.data()] __device__ (int index) {
                x[index] = x[index] - y[index];
            });

    checkCudaErrors(cudaDeviceSynchronize());
    x_host = x_dev;
    end_millis = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    cudaDeviceSynchronize(); 
    printf("%ld\n",end_millis - begin_millis);
    
    return 0;
}

