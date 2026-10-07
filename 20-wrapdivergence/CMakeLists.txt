cmake_minimum_required(VERSION 3.18)

include(${CMAKE_CURRENT_LIST_DIR}/../cmake/cutests.cmake)

project(devices LANGUAGES CXX CUDA)

add_executable(hello hello.cu)
target_include_directories(hello PUBLIC ../common)
target_compile_options(hello PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_compile_options(hello PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

