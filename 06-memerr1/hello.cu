#include <iostream>
#include <cuda_runtime.h>

__global__ void kernel(int *arr) {
    arr[0] = 0;
    int index = 1;
    while (arr[index] != 0) {
        arr[0] += arr[index];
        index++;
    }
}

int main() {
    int *a;
    a = (int *)malloc(sizeof(int) * 12);
    int index = 1;
    for (index = 1; index <= 10; ++index) {
        a[index] = index;
    }
    kernel<<<1, 1>>>(a);
    printf("%d", a[0]);
    cudaDeviceSynchronize();
    free(a);
    return 0;
}

