#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

// 用 Brent-Kung 实现的块内 scan

template<int BLOCKSIZE>
__global__ void psum_blockKernel(float* input, int n, size_t stride) {
    __shared__ float smem[BLOCKSIZE];
    
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    size_t now = (idx + 1) * stride - 1;
    smem[tid] = now < n ? input[now] : 0.;

    for (int i = 2; i <= BLOCKSIZE; i <<= 1) {
        __syncthreads();
        int pos = (tid + 1) * i - 1;
        if (pos < BLOCKSIZE)
            smem[pos] += smem[pos - (i >> 1)];
    }

    for (int i = BLOCKSIZE / 2; i > 1; i >>= 1) {
        __syncthreads();
        int pos = (tid + 1) * i - 1;
        if (pos + (i >> 1) < BLOCKSIZE)
            smem[pos + (i >> 1)] += smem[pos];
    }

    __syncthreads();
    if (now < n)
        input[now] = smem[tid];
}

template<int BLOCKSIZE>
__global__ void psum_Kernel(float* input, int n, float* output) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    float val = idx < n ? input[idx] : 0.;
    for (size_t i = BLOCKSIZE; i < n; i *= BLOCKSIZE) {
        int now = (idx + 1) / i;
        int pre = now * i - 1;
        if (now == 0 || ((idx + 1) % i) == 0 || (pre + 1) % (i * BLOCKSIZE) == 0) continue;
        val += input[pre];
    }
    if (idx < n)
        output[idx] = val;
}

// input, output are device pointers
void prefix_sum(int n, const float* input, float* output) {

    constexpr int BLOCKSIZE = 1024;

    float* input_other;
    cudaMalloc(&input_other, sizeof(float) * n);
    cudaMemcpy(input_other, input, sizeof(float) * n, cudaMemcpyDeviceToDevice);

    int blockSize = BLOCKSIZE;

    for (size_t i = 1; i < n; i *= blockSize) {
        int oper_size = n / i;
        int gridSize = (oper_size + blockSize - 1) / blockSize;

        psum_blockKernel<BLOCKSIZE><<<gridSize, blockSize>>>(input_other, n, i);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    int gridSize = (n + blockSize - 1) / blockSize;
    psum_Kernel<BLOCKSIZE><<<gridSize, blockSize>>>(input_other, n, output);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaFree(input_other);

} 