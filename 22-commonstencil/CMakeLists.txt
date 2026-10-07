cmake_minimum_required(VERSION 3.18)

include(${CMAKE_CURRENT_LIST_DIR}/../cmake/cutests.cmake)

project(devices LANGUAGES CXX CUDA)

add_executable(common hello.cu)
target_include_directories(common PUBLIC ../common)
target_compile_options(common PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_compile_options(common PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(better hello1.cu)
target_include_directories(better PUBLIC ../common)
target_compile_options(better PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_compile_options(better PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

add_executable(best hello2.cu)
target_include_directories(best PUBLIC ../common)
target_compile_options(best PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
target_compile_options(best PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
