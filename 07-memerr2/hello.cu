// Mistake 2: dereference a cudaMalloc pointer on the host.
// cudaMalloc returns a device address; the host write below crashes (segmentation fault).
#include <cstdio>
#include <cuda_runtime.h>

__global__ void kernel(int *arr) {
    arr[0] = 0;
    for (int i = 1; arr[i] != 0; ++i)
        arr[0] += arr[i];
}

int main() {
    int *a;
    cudaMalloc(&a, sizeof(int) * 12);   // a now points into GPU memory
    for (int i = 1; i <= 10; ++i)
        a[i] = i;                        // host write: crash
    kernel<<<1, 1>>>(a);
    cudaDeviceSynchronize();
    printf("%d\n", a[0]);                // host read: never reached
    cudaFree(a);
    return 0;
}
