// Peer-to-peer copy between GPU 0 and GPU 1, timed with CUDA events.
#include <cstdio>
#include <cuda_runtime.h>

#define CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(e)); return -1.0f; } } while (0)

float p2p_copy(size_t size) {
    int *pointers[2];
    int can01 = 0, can10 = 0;
    CHECK(cudaDeviceCanAccessPeer(&can01, 0, 1));
    CHECK(cudaDeviceCanAccessPeer(&can10, 1, 0));
    printf("peer access 0->1: %d, 1->0: %d%s\n", can01, can10,
           can01 && can10 ? "" : " (copy is staged through the host)");

    CHECK(cudaSetDevice(0));
    if (can01) CHECK(cudaDeviceEnablePeerAccess(1, 0));
    CHECK(cudaMalloc(&pointers[0], size));

    CHECK(cudaSetDevice(1));
    if (can10) CHECK(cudaDeviceEnablePeerAccess(0, 0));
    CHECK(cudaMalloc(&pointers[1], size));

    // Events and the copy are queued on the current device (1).
    cudaEvent_t begin, end;
    CHECK(cudaEventCreate(&begin));
    CHECK(cudaEventCreate(&end));
    CHECK(cudaEventRecord(begin));
    CHECK(cudaMemcpyAsync(pointers[0], pointers[1], size, cudaMemcpyDeviceToDevice));
    CHECK(cudaEventRecord(end));
    CHECK(cudaEventSynchronize(end));

    float elapsed_ms;
    CHECK(cudaEventElapsedTime(&elapsed_ms, begin, end));

    CHECK(cudaEventDestroy(end));
    CHECK(cudaEventDestroy(begin));
    CHECK(cudaFree(pointers[1]));
    CHECK(cudaSetDevice(0));
    CHECK(cudaFree(pointers[0]));
    return elapsed_ms / 1000;
}

int main() {
    int count = 0;
    cudaGetDeviceCount(&count);
    if (count < 2) {
        printf("needs 2 GPUs, found %d (set CUDA_VISIBLE_DEVICES to two GPUs)\n", count);
        return 1;
    }
    size_t size = 1000000;
    float seconds = p2p_copy(size);
    if (seconds < 0)
        return 1;
    printf("time = %f s, %.2f GB/s\n", seconds, size / seconds / 1e9);
    return 0;
}
