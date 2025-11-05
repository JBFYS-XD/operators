#include <bits/stdc++.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <sys/time.h>

#define BLOCKSIZE 256
#define ELEMENTS_PER_BLOCK BLOCKSIZE

// Top Bandwidth: 192 GB/s
// spend time: 8.119 ms
// bandwidth: 15.296 GB/s
// spendup: 1.0x

__global__ void reduceKernel(float* input, float* output, int n) {
    __shared__ float sdata[BLOCKSIZE];

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = idx < n ? input[idx] : 0.;
    __syncthreads();

    for(int s = 1; s < blockDim.x; s <<= 1) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

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