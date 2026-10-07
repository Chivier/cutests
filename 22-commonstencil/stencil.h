// Shared harness for the three stencil versions: data, CPU reference, timing, check.
// Stencil (periodic boundary): out = up + down + left + right - 4 * centre.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include "helper_cuda.h"

inline int stencil_ref(const std::vector<int> &a, int rows, int cols, int r, int c) {
    return a[(size_t)((r + rows - 1) % rows) * cols + c] + a[(size_t)((r + 1) % rows) * cols + c] +
           a[(size_t)r * cols + (c + cols - 1) % cols] + a[(size_t)r * cols + (c + 1) % cols] -
           4 * a[(size_t)r * cols + c];
}

// launch(rows, cols, d_in, d_out) enqueues the kernel. Returns 0 if the result is correct.
template <class Launch>
int run_stencil(const char *name, Launch launch) {
    const int rows = 1 << 14, cols = 1 << 14;
    const size_t count = (size_t)rows * cols, bytes = count * sizeof(int);

    std::vector<int> in(count), out(count);
    for (size_t i = 0; i < count; ++i)
        in[i] = std::rand() % 1024 - 512;

    int *d_in, *d_out;
    checkCudaErrors(cudaMalloc(&d_in, bytes));
    checkCudaErrors(cudaMalloc(&d_out, bytes));
    checkCudaErrors(cudaMemcpy(d_in, in.data(), bytes, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    launch(rows, cols, d_in, d_out);   // warm-up
    checkCudaErrors(cudaGetLastError());
    const int reps = 10;
    checkCudaErrors(cudaEventRecord(start));
    for (int r = 0; r < reps; ++r)
        launch(rows, cols, d_in, d_out);
    checkCudaErrors(cudaEventRecord(stop));
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaEventSynchronize(stop));
    float ms;
    checkCudaErrors(cudaEventElapsedTime(&ms, start, stop));

    checkCudaErrors(cudaMemcpy(out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    size_t bad = 0;
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            bad += out[(size_t)r * cols + c] != stencil_ref(in, rows, cols, r, c);

    printf("%-8s %7.3f ms per step, %6.1f GB/s  %s\n", name, ms / reps,
           2.0 * bytes / (ms / reps * 1e-3) / 1e9, bad ? "WRONG" : "ok");
    checkCudaErrors(cudaFree(d_in));
    checkCudaErrors(cudaFree(d_out));
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    return bad != 0;
}
