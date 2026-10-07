# Shared settings for every example. Include before project().
#
# GPU target: pass -DCMAKE_CUDA_ARCHITECTURES=86 (or 75, 90, ...) to choose one.
# Otherwise build for the GPU in this machine: CMake >= 3.24 has "native"; older CMake
# asks nvidia-smi. Without either, nvcc's default target is used, and the GPU then runs
# JIT-compiled PTX, which fails if the driver is older than the toolkit
# (cudaErrorUnsupportedPtxVersion).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(CMAKE_CUDA_ARCHITECTURES native)
  else()
    execute_process(COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
                    OUTPUT_VARIABLE _cutests_cc RESULT_VARIABLE _cutests_rc ERROR_QUIET)
    if(_cutests_rc EQUAL 0 AND _cutests_cc MATCHES "^([0-9]+)\\.([0-9]+)")
      set(CMAKE_CUDA_ARCHITECTURES "${CMAKE_MATCH_1}${CMAKE_MATCH_2}")
      message(STATUS "cutests: CUDA architecture ${CMAKE_CUDA_ARCHITECTURES} (from nvidia-smi)")
    endif()
  endif()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
