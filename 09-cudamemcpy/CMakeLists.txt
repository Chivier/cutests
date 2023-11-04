cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CUDA_ARCHITECTURES 61)

project(devices LANGUAGES CXX CUDA)

add_executable(hello hello.cu)
target_include_directories(hello PUBLIC ../Common)

