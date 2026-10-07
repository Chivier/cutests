// Stencil, version 1: a 1-D grid, one thread per element.
// Consecutive threads read consecutive addresses, so the loads coalesce.
#include "stencil.h"

__global__ void stencil(int rows, int cols, const int *in, int *out) {
    size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= (size_t)rows * cols)
        return;
    int r = index / cols, c = index % cols;
    out[index] = in[(size_t)((r + rows - 1) % rows) * cols + c]   // up
               + in[(size_t)((r + 1) % rows) * cols + c]          // down
               + in[(size_t)r * cols + (c + cols - 1) % cols]     // left
               + in[(size_t)r * cols + (c + 1) % cols]            // right
               - 4 * in[index];
}

int main() {
    return run_stencil("1-D", [](int rows, int cols, const int *in, int *out) {
        size_t total = (size_t)rows * cols;
        int block = 256;
        stencil<<<(unsigned)((total + block - 1) / block), block>>>(rows, cols, in, out);
    });
}
