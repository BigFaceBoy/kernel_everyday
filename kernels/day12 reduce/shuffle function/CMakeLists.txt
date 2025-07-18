cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
Project(CUDA_PUZZLES CXX C CUDA)

set(CMAKE_CUDA_FLAGS "-arch=compute_80 -O3")
include_directories(./include)

add_executable(shfl shfl.cu)