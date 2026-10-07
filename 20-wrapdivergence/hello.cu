// Warp divergence: odd elements get x + y, even elements get x - y, three ways.
//   divergent: one thread per element, branch on i % 2 (neighbouring threads disagree)
//   split:     branch on threadIdx.x parity (still splits every warp in two)
//   pairs:     one thread per (even, odd) pair, no branch at all
// All three are checked against a CPU reference and timed with CUDA events (kernel only).
// This kernel is memory-bound, so divergence of a one-instruction branch costs little;
// divergence matters when the two paths are long.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

__global__ void divergent(int n, float *x, const float *y) {
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        if (i % 2 == 1)
            x[i] = x[i] + y[i];
        else
            x[i] = x[i] - y[i];
    }
}

__global__ void split(int n, float *x, const float *y) {
    // blockDim.x and the grid stride are even, so i has the parity of threadIdx.x.
    if ((threadIdx.x & 1) == 1) {
        for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
            x[i] = x[i] + y[i];
    } else {
        for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
            x[i] = x[i] - y[i];
    }
}

__global__ void pairs(int n, float *x, const float *y) {
    for (int k = blockDim.x * blockIdx.x + threadIdx.x; 2 * k + 1 < n; k += blockDim.x * gridDim.x) {
        x[2 * k] = x[2 * k] - y[2 * k];
        x[2 * k + 1] = x[2 * k + 1] + y[2 * k + 1];
    }
}

template <class Launch>
void run(const char *name, Launch launch, thrust::device_vector<float> &x_dev,
         const thrust::host_vector<float> &x_host, const thrust::host_vector<float> &expect) {
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    const int reps = 10;
    float total_ms = 0;
    for (int r = 0; r <= reps; ++r) {   // r == 0 is the warm-up
        x_dev = x_host;
        checkCudaErrors(cudaEventRecord(start));
        launch();
        checkCudaErrors(cudaEventRecord(stop));
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaEventSynchronize(stop));
        float ms;
        checkCudaErrors(cudaEventElapsedTime(&ms, start, stop));
        if (r > 0)
            total_ms += ms;
    }
    thrust::host_vector<float> got = x_dev;
    size_t bad = 0;
    for (size_t i = 0; i < got.size(); ++i)
        bad += got[i] != expect[i];
    printf("%-10s %8.3f ms  %s\n", name, total_ms / reps, bad ? "WRONG" : "ok");
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
}

int main() {
    int n = 1 << 26;
    int block_dim = 128;
    int grid_dim = (n + block_dim - 1) / block_dim;

    thrust::host_vector<float> x_host(n), y_host(n), expect(n);
    for (int i = 0; i < n; ++i) {
        x_host[i] = std::rand() / 3.0f;
        y_host[i] = std::rand() / 11.0f;
        expect[i] = i % 2 ? x_host[i] + y_host[i] : x_host[i] - y_host[i];
    }
    thrust::device_vector<float> x_dev(n);
    thrust::device_vector<float> y_dev = y_host;
    float *x = thrust::raw_pointer_cast(x_dev.data());
    const float *y = thrust::raw_pointer_cast(y_dev.data());

    run("divergent", [&] { divergent<<<grid_dim, block_dim>>>(n, x, y); }, x_dev, x_host, expect);
    run("split", [&] { split<<<grid_dim, block_dim>>>(n, x, y); }, x_dev, x_host, expect);
    run("pairs", [&] { pairs<<<grid_dim / 2, block_dim>>>(n, x, y); }, x_dev, x_host, expect);
    return 0;
}
