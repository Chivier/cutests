cmake_minimum_required(VERSION 3.18)

include(${CMAKE_CURRENT_LIST_DIR}/../cmake/cutests.cmake)

project(devices LANGUAGES CXX CUDA)

add_executable(hello hello.cu)

