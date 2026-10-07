// thrust containers: host_vector / device_vector; assignment copies between them.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>

// Grid-stride loop: any grid size covers all n elements.
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
    int grid_dim = (n + block_dim - 1) / block_dim;

    thrust::host_vector<float> x_host(n);
    thrust::host_vector<float> y_host(n);
    thrust::generate(x_host.begin(), x_host.end(), [] { return std::rand() / 3.0; });
    thrust::generate(y_host.begin(), y_host.end(), [] { return std::rand() / 11.0; });
    float expect = x_host[0] + y_host[0];
    printf("%f + %f = \n", x_host[0], y_host[0]);

    thrust::device_vector<float> x_dev = x_host;   // H -> D
    thrust::device_vector<float> y_dev = y_host;

    // Device lambda (needs nvcc --extended-lambda) captures two thrust::device_ptr by value;
    // thrust::raw_pointer_cast(x_dev.data()) would give a plain float*.
    kernel<<<grid_dim, block_dim>>>(n, [x = x_dev.data(), y = y_dev.data()] __device__ (int i) {
        x[i] = x[i] + y[i];
    });
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    x_host = x_dev;                                // D -> H
    printf("%f (%s)\n", x_host[0], x_host[0] == expect ? "ok" : "WRONG");
    return 0;
}
