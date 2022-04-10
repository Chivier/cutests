#include <iostream>
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <math.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/universal_vector.h>
#include <chrono>

using std::chrono::duration_cast;
using std::chrono::milliseconds;
using std::chrono::seconds;
using std::chrono::system_clock;

template <int iters, int blocksize>
__global__ void stencil_kernel(int row_num, int col_num, int *arr_data, int *result) {
    constexpr int chunksize = blocksize - iters * 2;
    int globalx = blockIdx.x * chunksize - iters + threadIdx.x;
    int globaly = blockIdx.y * chunksize - iters + threadIdx.y;
    __shared__ int mem[2][blocksize + 2][blocksize + 2];
    int boundx1 = std::min(std::max(globalx, 0), row_num - 1);
    int boundy1 = std::min(std::max(globaly, 0), col_num - 1);
    mem[0][1 + threadIdx.y][1 + threadIdx.x] = arr_data[row_num * boundy1 + boundx1];

    if (threadIdx.y == 0) {
        mem[0][0][1 + threadIdx.x] = arr_data[std::min(std::max((int)blockIdx.y * chunksize - iters - 1, 0), col_num - 1) * col_num + boundx1];
        mem[0][1 + blocksize][1 + threadIdx.x] = arr_data[std::min(std::max((int)blockIdx.y * chunksize - iters + blocksize, 0), col_num - 1) * col_num + boundx1];
    }

    if (threadIdx.x == 0) {
        mem[0][1 + threadIdx.y][0] = arr_data[std::min(std::max((int)blockIdx.x * chunksize - iters - 1, 0), row_num - 1) + row_num * boundy1];
        mem[0][1 + threadIdx.y][1 + blocksize] = arr_data[std::min(std::max((int)blockIdx.x * chunksize - iters + blocksize, 0),row_num - 1) + row_num * boundy1];
    }

    __syncthreads();
    for (int stage = 0; stage < iters; stage += 2) {
#pragma unroll
        for (int flag = 0; flag < 2; ++flag) {
            mem[1 ^ flag][1 + threadIdx.y + 1][1 + threadIdx.x] = 
                mem[flag][1 + threadIdx.y - 1][1 + threadIdx.x] +
                mem[flag][1 + threadIdx.y][1 + threadIdx.x + 1] +
                mem[flag][1 + threadIdx.y][1 + threadIdx.x - 1] +
                mem[flag][1 + threadIdx.y][1 + threadIdx.x] * 4;
            __syncthreads();
        }
    }

    result[row_num * boundy1 + boundx1] = mem[0][1 + threadIdx.y][1 + threadIdx.x];
}

template <int iters, int blocksize>
void stencil(int row_num, int col_num, int *arr_data, int *result) {
    constexpr int chunksize = blocksize - iters * 2;
    stencil_kernel<iters, blocksize><<<dim3((row_num + chunksize - 1) / chunksize, (col_num + chunksize - 1) / chunksize, 1), dim3(blocksize, blocksize, 1)>>>
        (row_num, col_num, arr_data, result);
}

int main() {
    int row_num = 1 << 14;
    int col_num = 1 << 14;

    int *arr;
    int *result;
    cudaMallocManaged(&arr, sizeof(int) * row_num * col_num);
    cudaMallocManaged(&result, sizeof(int) * row_num * col_num);

    for (int index = 0; index < row_num * col_num; ++index) {
        arr[index] = rand() % 1024 - 512;
    }

    cudaDeviceSynchronize();
    auto begin_millis = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();

    cudaDeviceSynchronize();
    int total_numbers = row_num * col_num;
    int block_size = 1024;
    stencil<4, 32>(row_num, col_num, arr, result);

    cudaDeviceSynchronize(); 
    auto end_millis = duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
    printf("%ld\n", end_millis - begin_millis);
    cudaDeviceSynchronize();
    return 0;
}

