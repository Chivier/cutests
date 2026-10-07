cmake_minimum_required(VERSION 3.18)

include(${CMAKE_CURRENT_LIST_DIR}/../cmake/cutests.cmake)

project(threads LANGUAGES CXX CUDA)

add_executable(threads threads.cu)

