#pragma once
#include <cuda_runtime.h>

#define matrix_pos(i, j, n) ((i) * n + (j))
#define FETCH_FLOAT4(pointer)  (reinterpret_cast<float4*>(&(pointer))[0])

// 对单线程处理的数据分布做出改变， 提高内存合并
// 改前： 每个线程处理 2 维连续的 8 * 8
// 改后： 离散的 (4 * 2) * 8 个元素

// CUBLAS: 5.010714 ms
// MyKernel: 4.618322 ms

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
    int row_tid_lda = (tid / sum_cols_lda);
    int row_tid_ldb = (tid / sum_cols_ldb);

    __shared__ float smem_A[2][BLOCK_K][BLOCK_M];
    __shared__ float smem_B[2][BLOCK_K][BLOCK_N];

    float reg_lda[thread_ld_x] = {0.f};
    float reg_ldb[thread_ld_x] = {0.f};

    constexpr int sum_cols_calc = BLOCK_N / THREAD_X;
    constexpr int sum_rows_calc = BLOCK_M / THREAD_Y;

    int col_tid_calc = threadIdx.x * 4;
    int row_tid_calc = threadIdx.y * 4;
    int col_idx_calc = (blockDim.x * blockIdx.x) * THREAD_X + col_tid_calc;
    int row_idx_calc = (blockDim.y * blockIdx.y) * THREAD_Y + row_tid_calc;

    float reg_calc_a[2][THREAD_Y] = {0.f};
    float reg_calc_b[2][THREAD_X] = {0.f};
    float temp[THREAD_Y][THREAD_X] = {0.f};

    #pragma unroll
    for (int i = 0; i < BLOCK_M; i += sum_rows_lda) {
        int global_col = col_tid_lda;
        int global_row = row_glb_bid + row_tid_lda + i;
        FETCH_FLOAT4(reg_lda[0]) = FETCH_FLOAT4(A[matrix_pos(global_row, global_col, K)]);
        int shared_row = col_tid_lda;
        int shared_col = row_tid_lda + i;
        smem_A[0][shared_row    ][shared_col] = reg_lda[0];
        smem_A[0][shared_row + 1][shared_col] = reg_lda[1];
        smem_A[0][shared_row + 2][shared_col] = reg_lda[2];
        smem_A[0][shared_row + 3][shared_col] = reg_lda[3];
    }
    #pragma unroll
    for (int i = 0; i < BLOCK_K; i += sum_rows_ldb) {
        int global_col = col_glb_bid + col_tid_ldb;
        int global_row = row_tid_ldb + i;
        int shared_col = col_tid_ldb;
        int shared_row = row_tid_ldb + i;
        FETCH_FLOAT4(smem_B[0][shared_row][shared_col]) = FETCH_FLOAT4(B[matrix_pos(global_row, global_col, N)]);
    }

    __syncthreads();
    
    #pragma unroll
    for (int i = 0, r_i = 0; i < BLOCK_N; i += sum_cols_calc * 4, r_i += 4) {
        FETCH_FLOAT4(reg_calc_b[0][r_i]) = FETCH_FLOAT4(smem_B[0][0][col_tid_calc + i]);
    }
    #pragma unroll
    for (int i = 0, r_i = 0; i < BLOCK_M; i += sum_rows_calc * 4, r_i += 4) {
        FETCH_FLOAT4(reg_calc_a[0][r_i]) = FETCH_FLOAT4(smem_A[0][0][row_tid_calc + i]);
    }
    
    int till_k = 0, pp_f = 0, pr_f = 0;
    do {
        till_k += BLOCK_K;
        if (till_k < K) {
            #pragma unroll
            for (int i = 0; i < BLOCK_M; i += sum_rows_lda) {
                int global_col = col_tid_lda + till_k;
                int global_row = row_glb_bid + row_tid_lda + i;
                FETCH_FLOAT4(reg_lda[0]) = FETCH_FLOAT4(A[matrix_pos(global_row, global_col, K)]);
            }
            #pragma unroll
            for (int i = 0; i < BLOCK_K; i += sum_rows_ldb) {
                int global_col = col_glb_bid + col_tid_ldb;
                int global_row = row_tid_ldb + till_k + i;
                FETCH_FLOAT4(reg_ldb[0]) = FETCH_FLOAT4(B[matrix_pos(global_row, global_col, N)]);
            }
        }
        #pragma unroll
        for (int k = 0; k < BLOCK_K - 1; k ++) {

            #pragma unroll
            for (int i = 0, r_i = 0; i < BLOCK_N; i += sum_cols_calc * 4, r_i += 4) {
                FETCH_FLOAT4(reg_calc_b[pr_f ^ 1][r_i]) = FETCH_FLOAT4(smem_B[pp_f][k + 1][col_tid_calc + i]);
            }
            #pragma unroll
            for (int i = 0, r_i = 0; i < BLOCK_M; i += sum_rows_calc * 4, r_i += 4) {
                FETCH_FLOAT4(reg_calc_a[pr_f ^ 1][r_i]) = FETCH_FLOAT4(smem_A[pp_f][k + 1][row_tid_calc + i]);
            }
            
            #pragma unroll
            for (int i = 0; i < THREAD_Y; i ++) {
                #pragma unroll
                for (int j = 0; j < THREAD_X; j ++) {
                    temp[i][j] += reg_calc_a[pr_f][i] * reg_calc_b[pr_f][j];
                }
            }
            pr_f ^= 1;
        }
        if (till_k < K) {
            for (int i = 0; i < BLOCK_M; i += sum_rows_lda) {
                int shared_row = col_tid_lda;
                int shared_col = row_tid_lda + i;
                smem_A[pp_f ^ 1][shared_row    ][shared_col] = reg_lda[0];
                smem_A[pp_f ^ 1][shared_row + 1][shared_col] = reg_lda[1];
                smem_A[pp_f ^ 1][shared_row + 2][shared_col] = reg_lda[2];
                smem_A[pp_f ^ 1][shared_row + 3][shared_col] = reg_lda[3];
            }
            for (int i = 0; i < BLOCK_K; i += sum_rows_ldb) {
                int shared_col = col_tid_ldb;
                int shared_row = row_tid_ldb + i;
                FETCH_FLOAT4(smem_B[pp_f ^ 1][shared_row][shared_col]) = FETCH_FLOAT4(reg_ldb[0]);
            }
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0, r_i = 0; i < BLOCK_N; i += sum_cols_calc * 4, r_i += 4) {
            FETCH_FLOAT4(reg_calc_b[pr_f ^ 1][r_i]) = FETCH_FLOAT4(smem_B[pp_f ^ 1][0][col_tid_calc + i]);
        }
        #pragma unroll
        for (int i = 0, r_i = 0; i < BLOCK_M; i += sum_rows_calc * 4, r_i += 4) {
            FETCH_FLOAT4(reg_calc_a[pr_f ^ 1][r_i]) = FETCH_FLOAT4(smem_A[pp_f ^ 1][0][row_tid_calc + i]);
        }
        #pragma unroll
        for (int i = 0; i < THREAD_Y; i ++) {
            #pragma unroll
            for (int j = 0; j < THREAD_X; j ++) {
                temp[i][j] += reg_calc_a[pr_f][i] * reg_calc_b[pr_f][j];
            }
        }
        pr_f ^= 1;
        pp_f ^= 1;
    } while (till_k < K);

    #pragma unroll
    for (int i = 0, r_i = 0; i < BLOCK_M; i += sum_rows_calc * 4, r_i += 4) {
        #pragma unroll
        for (int s = 0; s < 4; s ++) {
            #pragma unroll
            for (int j = 0, r_j = 0; j < BLOCK_N; j += sum_cols_calc * 4, r_j += 4) {
                FETCH_FLOAT4(C[matrix_pos(row_idx_calc + i + s, col_idx_calc + j, N)]) = FETCH_FLOAT4(temp[r_i + s][r_j]);
            }
        }
    }

}
