// Mistake 1: hand a host (malloc) pointer to a kernel and never check for errors.
//
// On most PCIe GPUs the kernel faults on the host address, nothing reports it, and
// a[0] prints whatever malloc left there. 08-errhandle shows how to catch the error.
// On systems with HMM or ATS (cudaDevAttrPageableMemoryAccess = 1, "Addressing Mode:
// HMM/ATS" in nvidia-smi -q) the GPU can read pageable host memory and this prints 55.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void kernel(int *arr) {
    arr[0] = 0;
    for (int i = 1; arr[i] != 0; ++i)   // a[11] == 0 ends the loop
        arr[0] += arr[i];
}

int main() {
    int *a = (int *)calloc(12, sizeof(int));   // calloc: a[11] must be 0
    for (int i = 1; i <= 10; ++i)
        a[i] = i;

    kernel<<<1, 1>>>(a);       // host pointer
    cudaDeviceSynchronize();   // return value ignored: that is the mistake
    printf("%d\n", a[0]);

    free(a);
    return 0;
}
