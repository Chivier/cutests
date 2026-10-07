// Stencil, version 3: each block loads a tile plus a one-element halo into shared memory,
// synchronizes, then computes from shared memory. Every input is read from global memory
// about once per block instead of five times.
// For a single 5-point step the L1/L2 caches already catch most of that reuse, so expect
// a small gain at best; tiling pays off with wider stencils or several steps per load.
#include "stencil.h"

constexpr int TX = 32, TY = 8;

__device__ __forceinline__ int wrap(int i, int n) { return i < 0 ? i + n : (i >= n ? i - n : i); }

__global__ void stencil(int rows, int cols, const int *in, int *out) {
    __shared__ int tile[TY + 2][TX + 2];
    int c = blockIdx.x * TX + threadIdx.x;
    int r = blockIdx.y * TY + threadIdx.y;
    int tx = threadIdx.x + 1, ty = threadIdx.y + 1;
    // rows and cols are multiples of the tile size (checked on the host): no bounds test,
    // and every thread reaches __syncthreads().
    tile[ty][tx] = in[(size_t)r * cols + c];
    if (threadIdx.y == 0)      tile[0][tx]      = in[(size_t)wrap(r - 1, rows) * cols + c];
    if (threadIdx.y == TY - 1) tile[TY + 1][tx] = in[(size_t)wrap(r + 1, rows) * cols + c];
    if (threadIdx.x == 0)      tile[ty][0]      = in[(size_t)r * cols + wrap(c - 1, cols)];
    if (threadIdx.x == TX - 1) tile[ty][TX + 1] = in[(size_t)r * cols + wrap(c + 1, cols)];
    __syncthreads();
    out[(size_t)r * cols + c] = tile[ty - 1][tx] + tile[ty + 1][tx]
                              + tile[ty][tx - 1] + tile[ty][tx + 1] - 4 * tile[ty][tx];
}

int main() {
    return run_stencil("shared", [](int rows, int cols, const int *in, int *out) {
        if (rows % TY || cols % TX) {
            fprintf(stderr, "rows must be a multiple of %d and cols of %d\n", TY, TX);
            exit(1);
        }
        stencil<<<dim3(cols / TX, rows / TY), dim3(TX, TY)>>>(rows, cols, in, out);
    });
}
