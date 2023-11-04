cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_CUDA_ARCHITECTURES 61)

project(devices LANGUAGES CXX CUDA)

add_executable(common hello.cu)
target_include_directories(common PUBLIC ../Common)
target_compile_options(common PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_compile_options(common PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(better hello1.cu)
target_include_directories(better PUBLIC ../Common)
target_compile_options(better PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_compile_options(better PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(best hello2.cu)
target_include_directories(best PUBLIC ../Common)
target_compile_options(best PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_compile_options(best PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
