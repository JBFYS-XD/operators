cmake_minimum_required(VERSION 3.28)

project(DEVELOP_ VERSION 1.0 LANGUAGES CXX CUDA C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin ${CMAKE_CXX_COMPILER}")


message("------\t loading libraries \t------")

# cuda 配置
find_package(CUDAToolkit REQUIRED)
if (CUDAToolkit_FOUND)
    message("CUDA_LIB loaded")
endif()

# MPI 配置
find_package(MPI REQUIRED)
if (MPI_FOUND)
    message("MPI_LIB loaded")
endif()

# CUDNN_FRONTEND
set(CUDNN_FRONTEND_DIR "${CMAKE_SOURCE_DIR}/libs/cudnn-frontend")
set(CUDNN_FRONTEND_INCLUDE_DIR "${CUDNN_FRONTEND_DIR}/include")
if (NOT EXISTS "${CUDNN_FRONTEND_INCLUDE_DIR}/cudnn_frontend.h")
    message(FATAL_ERROR "CUDNN_FRONTEND headers not found at ${CUDNN_FRONTEND_INCLUDE_DIR}")
else()
    INCLUDE_DIRECTORIES("${CUDNN_FRONTEND_INCLUDE_DIR}")
    message("CUDNN_FRONTEND_LIB loaded")
endif()

# CUTLASS 配置
set(CUTLASS_DIR "${CMAKE_SOURCE_DIR}/libs/cutlass")
set(CUTLASS_INCLUDE_DIR "${CUTLASS_DIR}/include")
if(NOT EXISTS "${CUTLASS_INCLUDE_DIR}/cutlass/cutlass.h")
    INCLUDE_DIRECTORIES("${CUTLASS_INCLUDE_DIR}/cutlass")
    message(FATAL_ERROR "CUTLASS headers not found at ${CUTLASS_INCLUDE_DIR}")
else()
    message("CUTLASS_LIB loaded")
endif()

message("------\t end libraries \t------")

add_subdirectory(src)