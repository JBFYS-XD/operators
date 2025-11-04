#include <cuda_runtime.h>

#define matrix_pos(i, j, n) ((i) * n + (j))
#define FETCH_FLOAT4(pointer)  (reinterpret_cast<float4*>(&(pointer))[0])


// float4 对矩阵 B 的读取也进行优化, 并应用寄存器进行第二级缓存

// CUBLAS: 4.383114 ms
// MyKernel: 6.994575 ms

template<int BLOCK_M, int BLOCK_N, int BLOCK_K, int THREAD_X, int THREAD_Y>
__global__ 
void sgemm_gKernel(float* A, float* B, float* C, int M, int N, int K) {
    
    __shared__ float smem_A[BLOCK_M][BLOCK_K];
    __shared__ float smem_B[BLOCK_K][BLOCK_N];

    int store_tid_y = threadIdx.y * THREAD_Y;
    int store_tid_x = threadIdx.x * THREAD_X;
    int store_idx_y = (blockDim.y * blockIdx.y + threadIdx.y) * THREAD_Y;
    int store_idx_x = (blockDim.x * blockIdx.x + threadIdx.x) * THREAD_X;
    
    int calc_tid_y = threadIdx.y * THREAD_Y;
    int calc_tid_x = threadIdx.x * THREAD_X;
    int calc_idx_y = (blockDim.y * blockIdx.y + threadIdx.y) * THREAD_Y;
    int calc_idx_x = (blockDim.x * blockIdx.x + threadIdx.x) * THREAD_X;

    float reg_A[THREAD_Y];
    float reg_B[THREAD_X];
    float temp[THREAD_Y][THREAD_X] = {0.f};

    for (int s = 0; s < K; s += BLOCK_K) {
        __syncthreads();
        for (int i = 0; i < THREAD_Y; i ++) {
            for (int j = 0; j < THREAD_X; j += 4) {
                int pos_A = s + store_tid_x;
                int pos_B = s + store_tid_y;
                FETCH_FLOAT4(smem_A[store_tid_y + i][store_tid_x + j]) = 
                    FETCH_FLOAT4(A[matrix_pos(store_idx_y + i, pos_A + j, K)]);
                FETCH_FLOAT4(smem_B[store_tid_y + i][store_tid_x + j]) = 
                    FETCH_FLOAT4(B[matrix_pos(pos_B + i, store_idx_x + j, N)]);
            }
        }
        __syncthreads();
        for (int k = 0; k < BLOCK_K; k ++) {
            for (int i = 0; i < 4; i ++) {
                reg_A[i] = smem_A[calc_tid_y + i][k];
            }
            FETCH_FLOAT4(reg_B[0]) = FETCH_FLOAT4(smem_B[k][calc_tid_x]);
            for (int i = 0; i < THREAD_Y; i ++) {
                for (int j = 0; j < THREAD_X; j ++) {
                    temp[i][j] += reg_A[i] * reg_B[j];
                }
            }
        }
    }

    for (int i = 0; i < THREAD_Y; i ++) {
        for (int j = 0; j < THREAD_X; j += 4) {
            FETCH_FLOAT4(C[matrix_pos(calc_idx_y + i, calc_idx_x + j, N)]) = FETCH_FLOAT4(temp[i][j]);
        }
    }
}
