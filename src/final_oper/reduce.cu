#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>

#define BLOCKSIZE 256
#define WARPSIZE 32
#define ELEMENTS_PER_THREAD 4
#define ELEMENTS_PER_BLOCK (BLOCKSIZE * ELEMENTS_PER_THREAD)

template <int blockSize>
__device__ int warpReduce(float sum) {
    #pragma unroll
    for (int i = warpSize / 2; i > 0; i >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, i);
    }
    return sum;
}

template <int blockSize>
__global__ void reduceKernel(float* input, float* output, int n) {
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = blockIdx.x * blockDim.x * ELEMENTS_PER_THREAD + threadIdx.x;
    
    float sum = 0;
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; i ++) {
        sum += idx + i * blockDim.x < n ? input[idx + i * blockDim.x] : 0.;
    }
    
    __shared__ float sdata[WARPSIZE];
    int wid = tid / WARPSIZE;
    int lid = tid % WARPSIZE;
    
    sum = warpReduce<blockSize>(sum);
    
    if (lid == 0) sdata[wid] = sum;
    __syncthreads();

    sum = tid < (blockDim.x / WARPSIZE) ? sdata[lid] : 0.;
    if (wid == 0) sum = warpReduce<blockSize>(sum);
    if (tid == 0) output[bid] = sum;
}

float reduce(int n, float* input) {

    float output;
    int threadsPerBlock = BLOCKSIZE;
    int blocksPerGrid = (n + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;
    
    float* temp;

    cudaMalloc((void**)&temp, sizeof(float) * blocksPerGrid * 2);
    
    float* arr1 = temp;
    float* arr2 = arr1 + blocksPerGrid;

    reduceKernel<BLOCKSIZE><<<blocksPerGrid, threadsPerBlock>>>(input, arr1, n);
    cudaDeviceSynchronize();

    n = blocksPerGrid;

    while (n > 1) {
        blocksPerGrid = (n + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;
        reduceKernel<BLOCKSIZE><<<blocksPerGrid, threadsPerBlock>>>(arr1, arr2, n);
        cudaDeviceSynchronize();
        n = blocksPerGrid;
        std::swap(arr1, arr2);
    }
    
    cudaMemcpy(&output, arr1, sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(temp);

    return output;
}