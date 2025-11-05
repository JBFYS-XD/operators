#pragma once
#include <cuda_runtime.h>

#define matrix_pos(i, j, n) ((i) * n + (j))

// 在分块基础上引入粗度

// CUBLAS: 4.397502 ms
// MyKernel: 13.094252 ms

template<int BLOCKSIZE, int K_TILL, int CFACTOR>
__global__ 
void sgemm_gKernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float smem_A[BLOCKSIZE * CFACTOR][K_TILL * CFACTOR];
    __shared__ float smem_B[K_TILL * CFACTOR][BLOCKSIZE * CFACTOR];
    
    int tid_y = threadIdx.y * CFACTOR;
    int tid_x = threadIdx.x * CFACTOR;
    int idx_y = (blockDim.y * blockIdx.y + threadIdx.y) * CFACTOR;
    int idx_x = (blockDim.x * blockIdx.x + threadIdx.x) * CFACTOR;

    float temp[CFACTOR][CFACTOR];
    for (int i = 0; i < CFACTOR; i ++) {
        for (int j = 0; j < CFACTOR; j ++) {
            temp[i][j] = 0.;
        }
    }
    for (int s = 0; s < K; s += K_TILL * CFACTOR) {
        __syncthreads();
        for (int k = 0; k < K_TILL; k += BLOCKSIZE) {
            int pos_A = s + tid_x;
            int pos_B = s + tid_y;
            for (int i = 0; i < CFACTOR; i ++) {
                for (int j = 0; j < CFACTOR; j ++) {
                    smem_A[tid_y + i][tid_x + k + j] = A[matrix_pos(idx_y + i, k + pos_A + j, K)];
                    smem_B[tid_y + k + i][tid_x + j] = B[matrix_pos(k + pos_B + i, idx_x + j, N)];
                }
            }
        }
        __syncthreads();

        for (int k = 0; k < K_TILL * CFACTOR; k ++) {
            for (int i = 0; i < CFACTOR; i ++) {
                for (int j = 0; j < CFACTOR; j ++) {
                    temp[i][j] += smem_A[tid_y + i][k] * smem_B[k][tid_x + j];
                }
            }
        }
    }
    for (int i = 0; i < CFACTOR; i ++) {
        for (int j = 0; j < CFACTOR; j ++) {
            C[matrix_pos(idx_y + i, idx_x + j, N)] = temp[i][j];
        }
    }
}

void sgemm_g(int m, int n, int k, float* matrix_A, float* matrix_B, float* matrix_C) {
    constexpr int BLOCKSIZE = 16;
    constexpr int K_TILL = 16;
    constexpr int CFACTOR = 2;
    dim3 blockSize(BLOCKSIZE, BLOCKSIZE);
    dim3 gridSize(((n + CFACTOR - 1) / CFACTOR + BLOCKSIZE - 1) / BLOCKSIZE,
                    ((m + CFACTOR - 1) / CFACTOR + BLOCKSIZE - 1) / BLOCKSIZE);

    sgemm_gKernel<BLOCKSIZE, K_TILL, CFACTOR><<<gridSize, blockSize>>>(matrix_A, matrix_B, matrix_C, m, n, k);
    cudaDeviceSynchronize();

}