#include <cuda_runtime.h>

#define matrix_pos(i, j, n) ((i) * n + (j))

// naive 版本

// CUBLAS: 4.827395 ms
// MyKernel: 79.229347 ms

__global__ 
void sgemm_gKernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int m = blockDim.y * blockIdx.y + threadIdx.y;
    int n = blockDim.x * blockIdx.x + threadIdx.x;

    if (m < M && n < N) {
        float temp = 0.;
        for (int k = 0; k < K; k ++) {
            temp += A[matrix_pos(m, k, K)] * B[matrix_pos(k, n, N)];
        }
        C[matrix_pos(m, n, N)] = temp;
    }
}
