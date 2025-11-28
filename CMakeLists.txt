cmake_minimum_required(VERSION 3.28)


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

project(DEVELOP_ VERSION 1.0 LANGUAGES CXX CUDA C)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

message("------\t loading libraries \t------")

# CUDA 配置
find_package(CUDAToolkit REQUIRED)
if (CUDAToolkit_FOUND)
    message("CUDA_LIB found")
else()
    message(FATAL_ERROR "CUDA library not find")
endif()

# CuDNN 配置
find_package(CUDNN REQUIRED)
if(CUDNN_FOUND)
    message("CUDNN_LIB found")
else()
    message(FATAL_ERROR "CUDNN library not find")
endif()

# OpenMP 配置
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
    message("OpenMP_LIB found")
else()
    message(FATAL_ERROR "OpenMP library not find")
endif()

# MPI 配置
find_package(MPI REQUIRED)
if (MPI_FOUND)
    message("MPI_LIB found")
else()
    message(FATAL_ERROR "MPI_LIB library not find")
endif()

# CUDNN_FRONTEND 配置
set(CUDNN_FRONTEND_DIR "${CMAKE_SOURCE_DIR}/libs/cudnn-frontend")
set(CUDNN_FRONTEND_INCLUDE_DIR "${CUDNN_FRONTEND_DIR}/include")
if (EXISTS "${CUDNN_FRONTEND_INCLUDE_DIR}/cudnn_frontend.h")
    message("CUDNN_FRONTEND_LIB found")
else()
    message(FATAL_ERROR "CUDNN_FRONTEND headers not found at ${CUDNN_FRONTEND_INCLUDE_DIR}")
endif()

# CUTLASS 配置
set(CUTLASS_DIR "${CMAKE_SOURCE_DIR}/libs/cutlass")
set(CUTLASS_INCLUDE_DIR "${CUTLASS_DIR}/include")
if(EXISTS "${CUTLASS_INCLUDE_DIR}/cutlass/cutlass.h")
    message("CUTLASS_LIB found")
else()
    message(FATAL_ERROR "CUTLASS headers not found at ${CUTLASS_INCLUDE_DIR}")
endif()

message("------\t end libraries \t------")

add_subdirectory(src)