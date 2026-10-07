// Stencil, version 2: a 2-D grid of 32 x 8 blocks, no division or modulo for r and c.
// x must map to columns: threads of a warp differ in threadIdx.x, so they then read
// neighbouring addresses of one row (coalesced). Mapping x to rows would make every
// thread of a warp touch a different row, cols * 4 bytes apart.
#include "stencil.h"

__global__ void stencil(int rows, int cols, const int *in, int *out) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= rows || c >= cols)
        return;
    int up = r == 0 ? rows - 1 : r - 1, down = r == rows - 1 ? 0 : r + 1;
    int left = c == 0 ? cols - 1 : c - 1, right = c == cols - 1 ? 0 : c + 1;
    size_t row = (size_t)r * cols;
    out[row + c] = in[(size_t)up * cols + c] + in[(size_t)down * cols + c]
                 + in[row + left] + in[row + right] - 4 * in[row + c];
}

int main() {
    return run_stencil("2-D", [](int rows, int cols, const int *in, int *out) {
        dim3 block(32, 8);
        dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);
        stencil<<<grid, block>>>(rows, cols, in, out);
    });
}
