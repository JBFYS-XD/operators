cmake_minimum_required(VERSION 3.28)

project(DEVELOP_ VERSION 1.0 LANGUAGES CXX CUDA C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

message("------\t loading libraries \t------")

find_package(CUDAToolkit REQUIRED)
if (CUDAToolkit_FOUND)
    message("CUDA_LIB loaded")
endif()

find_package(MPI REQUIRED)
if (MPI_FOUND)
    message("MPI_LIB loaded")
endif()

message("------\t end libraries \t------")

add_subdirectory(src)