#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>

#define BLOCKSIZE 256
#define ELEMENTS_PER_THREAD 16
#define ELEMENTS_PER_BLOCK (BLOCKSIZE * ELEMENTS_PER_THREAD)

// spend time: 1.272 ms
// bandwidth: 98.270 GB/s
// spendup: 1.60x

__device__ void warpReduce(volatile float* smem, int tid) {
    smem[tid] += smem[tid + 32];
    smem[tid] += smem[tid + 16];
    smem[tid] += smem[tid +  8];
    smem[tid] += smem[tid +  4];
    smem[tid] += smem[tid +  2];
    smem[tid] += smem[tid +  1];
}

__global__ void reduceKernel(float* input, float* output, int n) {
    __shared__ float sdata[BLOCKSIZE];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    
    float sum = 0;
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i ++) {
        sum += idx + i * blockDim.x < n ? input[idx + i * blockDim.x] : 0.;
    }
    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32) warpReduce(sdata, tid);
    if (tid == 0) output[bid] = sdata[0];
}

float reduce(int n, float* input) {

    float output;
    int threadsPerBlock = BLOCKSIZE;
    int blocksPerGrid = (n + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;
    
    float* temp;

    cudaMalloc((void**)&temp, sizeof(float) * blocksPerGrid * 2);
    
    float* arr1 = temp;
    float* arr2 = arr1 + blocksPerGrid;

    reduceKernel<<<blocksPerGrid, threadsPerBlock>>>(input, arr1, n);
    cudaDeviceSynchronize();

    n = blocksPerGrid;

    while (n > 1) {
        blocksPerGrid = (n + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;
        reduceKernel<<<blocksPerGrid, threadsPerBlock>>>(arr1, arr2, n);
        cudaDeviceSynchronize();
        n = blocksPerGrid;
        std::swap(arr1, arr2);
    }
    
    cudaMemcpy(&output, arr1, sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(temp);

    return output;
}