#pragma once
#include <cuda_runtime.h>
#include <iostream>

#define matrix_pos(i, j, n) ((i) * n + (j))
#define FETCH_FLOAT4(pointer)  (reinterpret_cast<float4*>(&(pointer))[0])

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

// 此版本与 v5 所作的优化完全一致， 但是重构了代码

// CUBLAS: 4.383114 ms
// MyKernel: 6.994575 ms

template<int BLOCK_M, int BLOCK_N, int BLOCK_K, int THREAD_X, int THREAD_Y>
__global__ 
void sgemm_gKernel(float* A, float* B, float* C, int M, int N, int K) {
    
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    
    constexpr int thread_rows = BLOCK_M / THREAD_Y;
    constexpr int thread_cols = BLOCK_N / THREAD_X;
    constexpr int threads_per_block = thread_rows * thread_cols;
    
    constexpr int thread_ld_x = 4;
    
    constexpr int sum_cols_lda = BLOCK_K / thread_ld_x;
    constexpr int sum_cols_ldb = BLOCK_N / thread_ld_x;
    constexpr int sum_rows_lda = threads_per_block / sum_cols_lda;
    constexpr int sum_rows_ldb = threads_per_block / sum_cols_ldb;
    
    constexpr int row_step_lda = BLOCK_M / sum_rows_lda;
    constexpr int row_step_ldb = BLOCK_K / sum_rows_ldb;

    int col_glb_bid = blockDim.x * blockIdx.x * THREAD_X;
    int row_glb_bid = blockDim.y * blockIdx.y * THREAD_Y;
    
    int col_tid_lda = (tid % sum_cols_lda) * thread_ld_x;
    int col_tid_ldb = (tid % sum_cols_ldb) * thread_ld_x;
    int row_tid_lda = (tid / sum_cols_lda) * row_step_lda;
    int row_tid_ldb = (tid / sum_cols_ldb) * row_step_ldb;

    __shared__ float smem_A[BLOCK_K][BLOCK_M];
    __shared__ float smem_B[BLOCK_K][BLOCK_N];

    float reg_lda[thread_ld_x] = {0.f};
    float reg_ldb[thread_ld_x] = {0.f};


    int col_tid_calc = threadIdx.x * THREAD_X;
    int row_tid_calc = threadIdx.y * THREAD_Y;
    int col_idx_calc = (blockDim.x * blockIdx.x + threadIdx.x) * THREAD_X;
    int row_idx_calc = (blockDim.y * blockIdx.y + threadIdx.y) * THREAD_Y;

    float reg_calc_a[THREAD_Y] = {0.f};
    float reg_calc_b[THREAD_X] = {0.f};
    float temp[THREAD_Y][THREAD_X] = {0.f};

    for (int till_k = 0; till_k < K; till_k += BLOCK_K) {
        __syncthreads();
        for (int i = 0; i < row_step_lda; i ++) {
            int global_col = col_tid_lda + till_k;
            int global_row = row_glb_bid + row_tid_lda + i;
            FETCH_FLOAT4(reg_lda[0]) = FETCH_FLOAT4(A[matrix_pos(global_row, global_col, K)]);
            int shared_row = col_tid_lda;
            int shared_col = row_tid_lda + i;
            smem_A[shared_row    ][shared_col] = reg_lda[0];
            smem_A[shared_row + 1][shared_col] = reg_lda[1];
            smem_A[shared_row + 2][shared_col] = reg_lda[2];
            smem_A[shared_row + 3][shared_col] = reg_lda[3];
        }
        for (int i = 0; i < row_step_ldb; i ++) {
            int global_col = col_glb_bid + col_tid_ldb;
            int global_row = row_tid_ldb + till_k + i;
            int shared_col = col_tid_ldb;
            int shared_row = row_tid_ldb + i;
            FETCH_FLOAT4(smem_B[shared_row][shared_col]) = FETCH_FLOAT4(B[matrix_pos(global_row, global_col, N)]);
        }
        __syncthreads();
        for (int k = 0; k < BLOCK_K; k ++) {
            for (int i = 0; i < THREAD_X; i += 4) {
                FETCH_FLOAT4(reg_calc_b[i]) = FETCH_FLOAT4(smem_B[k][col_tid_calc + i]);
            }
            for (int i = 0; i < THREAD_Y; i += 4) {
                FETCH_FLOAT4(reg_calc_a[i]) = FETCH_FLOAT4(smem_A[k][row_tid_calc + i]);
            }
            for (int i = 0; i < THREAD_Y; i ++) {
                for (int j = 0; j < THREAD_X; j ++) {
                    temp[i][j] += reg_calc_a[i] * reg_calc_b[j];
                }
            }
        }
    }

    for (int i = 0; i < THREAD_Y; i ++) {
        for (int j = 0; j < THREAD_X; j += 4) {
            FETCH_FLOAT4(C[matrix_pos(row_idx_calc + i, col_idx_calc + j, N)]) = FETCH_FLOAT4(temp[i][j]);
        }
    }
}

void sgemm_g(int m, int n, int k, float* matrix_A, float* matrix_B, float* matrix_C) {
    constexpr int BLOCK_M = 64 * 2;
    constexpr int BLOCK_N = 64 * 2;
    constexpr int BLOCK_K = 8;
    constexpr int THREAD_X = 4 * 2;
    constexpr int THREAD_Y = 4 * 2;
    dim3 blockSize(16, 16);
    dim3 gridSize(((n) + BLOCK_N - 1) / BLOCK_N,
                    ((m) + BLOCK_M - 1) / BLOCK_M);

    sgemm_gKernel<BLOCK_M, BLOCK_N, BLOCK_K, THREAD_X, THREAD_Y><<<gridSize, blockSize>>>
                    (matrix_A, matrix_B, matrix_C, m, n, k);
    CUDA_CHECK(cudaDeviceSynchronize());

}