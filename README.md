# cutests

Basics: https://blog.chivier.site/2022/02/20/2022/2202-CudaProgramming/

Optimization: https://blog.chivier.site/2022/04/11/2022/2204-GPU%E7%A8%8B%E5%BA%8F%E4%BC%98%E5%8C%96%E6%96%B9%E6%B3%95/

## Build and run

Requirements: CUDA 12 or 13, CMake 3.18+, a C++17 host compiler. Each directory is a separate CMake project:

```sh
cd 09-cudamemcpy
cmake -S . -B build
cmake --build build
./build/hello
```

GPU target: by default each example is built for the GPU in the machine. CMake 3.24+ uses `native`; older CMake reads the compute capability from `nvidia-smi`. To choose a target, pass e.g. `-DCMAKE_CUDA_ARCHITECTURES=86`. If you build for an older target, the GPU runs JIT-compiled PTX, which fails with `cudaErrorUnsupportedPtxVersion` when the driver is older than the toolkit (e.g. CUDA 13.2 with driver 580). CUDA 13 no longer compiles for sm_50 to sm_72 (Maxwell, Pascal, Volta). `01-computecap` shows the compile target next to the GPU it runs on.

`./clear.sh` removes every `build/` directory.

## Notes

- `06-memerr1` and `08-errhandle` pass a `malloc` pointer to a kernel. On most PCIe systems this fails with `cudaErrorIllegalAddress` (700). On systems with HMM (Linux 6.1.24+, open kernel modules, CUDA 12.2+) or ATS (Grace Hopper), kernels can read pageable host memory, and both programs print 55. Check `nvidia-smi -q | grep "Addressing Mode"`, or `cudaDevAttrPageableMemoryAccess`.
- The kernels in 06–10 sum until they reach a 0, so `a[11]` must be 0. `malloc`, `cudaMalloc` and `cudaMallocManaged` do not zero memory, so the examples use `calloc` or set `a[11] = 0`.
- Only `__global__` kernels cannot be recursive. `__device__` functions may recurse.
- `18-multigpu` needs two visible GPUs. `19-multigpuerr` is a deliberate mistake: it assumes 100 GPUs and reports the first failing call.
- `20-wrapdivergence` and `22-commonstencil` check every result against a CPU reference and time only the kernel, using CUDA events.
