#include <cuda_runtime.h>

#define matrix_pos(i, j, n) ((i) * n + (j))

// 利用 float4 优化矩阵 A 的数据读取， 重构代码

// CUBLAS: 4.556484 ms
// MyKernel: 19.072315 ms

template<int BLOCK_M, int BLOCK_N, int BLOCK_K, int CFACTOR>
__global__ 
void sgemm_gKernel(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float smem_A[BLOCK_M][BLOCK_K];
    __shared__ float smem_B[BLOCK_K][BLOCK_N];
    
    int tid_x = threadIdx.x * 4;
    int tid_y = threadIdx.y;
    int idx_x = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
    int idx_y = (blockDim.y * blockIdx.y + threadIdx.y);

    float temp[4] = {0.f};

    for (int s = 0; s < K; s += BLOCK_K) {
        __syncthreads();
        int pos_A = s + tid_x;
        int pos_B = s + tid_y;
        reinterpret_cast<float4*>(&(smem_A[tid_y][tid_x]))[0] = 
            reinterpret_cast<float4*>(&(A[matrix_pos(idx_y, pos_A, K)]))[0];
        reinterpret_cast<float4*>(&(smem_B[tid_y][tid_x]))[0] = 
            reinterpret_cast<float4*>(&(B[matrix_pos(pos_B, idx_x, N)]))[0];
        __syncthreads();
        for (int k = 0; k < BLOCK_K; k ++) {
            for (int i = 0; i < 4; i ++) {
                temp[i] += smem_A[tid_y][k] * smem_B[k][tid_x + i];
            }
        }
    }

    for (int i = 0; i < 4; i ++) {
        C[matrix_pos(idx_y, idx_x + i, N)] = temp[i];
    }
}
