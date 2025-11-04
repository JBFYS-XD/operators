#include <cuda_runtime.h>

#define matrix_pos(i, j, n) ((i) * n + (j))

// 最基本的 K 方向分块

// CUBLAS: 4.920701 ms
// MyKernel: 26.043930 ms

template<int BLOCKSIZE, int K_TILL>
__global__ 
void sgemm_gKernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float smem_A[BLOCKSIZE][K_TILL];
    __shared__ float smem_B[K_TILL][BLOCKSIZE];
    
    int tid_y = threadIdx.y;
    int tid_x = threadIdx.x;
    int idx_y = blockDim.y * blockIdx.y + threadIdx.y;
    int idx_x = blockDim.x * blockIdx.x + threadIdx.x;

    float temp = 0.;
    for (int s = 0; s < K; s += K_TILL) {
        int pos_A = s + tid_x;
        int pos_B = s + tid_y;
        for (int k = 0; k < K_TILL; k += BLOCKSIZE) {
            smem_A[tid_y][tid_x + k] = A[matrix_pos(idx_y, k + pos_A, K)];
            smem_B[tid_y + k][tid_x] = B[matrix_pos(k + pos_B, idx_x, N)];
        }
        __syncthreads();

        for (int k = 0; k < K_TILL; k ++) {
            temp += smem_A[tid_y][k] * smem_B[k][tid_x];
        }
        __syncthreads();
    }
    C[matrix_pos(idx_y, idx_x, N)] = temp;
}
