// Mistake: hard-code the number of GPUs. With fewer than 100 GPUs, cudaSetDevice fails
// (cudaErrorInvalidDevice). Without checks every later call silently uses the wrong
// device and the "time" printed is meaningless; here the first error is reported.
#include <cstdio>
#include <cuda_runtime.h>

#define CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(e)); return -1.0f; } } while (0)

const int gpu_numbers = 100;   // wrong: ask cudaGetDeviceCount instead

float p2p_copy(size_t size) {
    int *pointers[gpu_numbers];

    for (int index = 0; index < gpu_numbers; ++index) {
        CHECK(cudaSetDevice(index));
        CHECK(cudaMalloc(&pointers[index], size));
    }
    for (int i = 0; i < gpu_numbers; ++i) {
        CHECK(cudaSetDevice(i));
        for (int j = 0; j < gpu_numbers; ++j)
            if (i != j)
                CHECK(cudaDeviceEnablePeerAccess(j, 0));
    }

    cudaEvent_t begin, end;
    CHECK(cudaEventCreate(&begin));
    CHECK(cudaEventCreate(&end));
    CHECK(cudaEventRecord(begin));
    for (int repeat = 0; repeat <= 100; ++repeat)
        for (int index = 1; index < gpu_numbers; ++index)
            CHECK(cudaMemcpyAsync(pointers[0], pointers[index], size, cudaMemcpyDeviceToDevice));
    CHECK(cudaEventRecord(end));
    CHECK(cudaEventSynchronize(end));

    float elapsed_ms;
    CHECK(cudaEventElapsedTime(&elapsed_ms, begin, end));
    for (int index = 0; index < gpu_numbers; ++index) {
        CHECK(cudaSetDevice(index));
        CHECK(cudaFree(pointers[index]));
    }
    return elapsed_ms / 1000;
}

int main() {
    float seconds = p2p_copy(1000000000);
    if (seconds < 0)
        return 1;
    printf("time = %f s\n", seconds);
    return 0;
}
