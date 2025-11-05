#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include "../final_oper/sgemm_cuda-core.cu"

/**
 * @note 测试环境 GTX 1060
 */


#define LEN 8192        // 1 << 13
#define KLEN 8192       // 1 << 13

#define matrix_pos(i, j, n) ((i) * n + (j))

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)


void random_matrix(int m, int n, float* a) {
    for (int i = 0; i < m; i ++) {
        for (int j = 0; j < n; j ++) {
            a[matrix_pos(i, j, n)] = 2.0 * (float)drand48() - 1.0;
        }
    }
}

float compare_matrices(float* output, float* correct, int n) {
    float max_diff = 0.;
    for (int i = 0; i < n; i++) {
        float diff = abs(output[i] - correct[i]);
        max_diff = diff > max_diff ? diff : max_diff;
    }
    return max_diff;
}

int main() {
    int m = LEN;
    int n = LEN;
    int k = KLEN;
    const size_t mem_size_A = m * k * sizeof(float);
    const size_t mem_size_B = k * n * sizeof(float);
    const size_t mem_size_C = m * n * sizeof(float);
    
    int nIter = 100;

    fprintf(stdout, "---------- gemm test ----------\n");
    fprintf(stdout, "matrix A: %d x %d\n", m, k);
    fprintf(stdout, "matrix B: %d x %d\n", k, n);
    fprintf(stdout, "matrix C: %d x %d\n", m, n);
    
    
    fprintf(stdout, "---------- step one ----------\n");
    fprintf(stdout, "alloc memory and init data\n");
    
    float* matrix_A_host = (float*)malloc(mem_size_A);
    float* matrix_B_host = (float*)malloc(mem_size_B);
    
    float* matrix_C_host_cbl_calc = (float*)malloc(mem_size_C);
    float* matrix_C_host_mkl_calc = (float*)malloc(mem_size_C);
    
    random_matrix(m, k, matrix_A_host);
    random_matrix(k, n, matrix_B_host);
    memset(matrix_C_host_cbl_calc, 0, mem_size_C);
    memset(matrix_C_host_mkl_calc, 0, mem_size_C);
    
    float* matrix_A_device;
    float* matrix_B_device;
    float* matrix_C_device;
    
    cudaMalloc(&matrix_A_device, mem_size_A);
    cudaMalloc(&matrix_B_device, mem_size_B);
    cudaMalloc(&matrix_C_device, mem_size_C);
    
    cudaMemcpy(matrix_A_device, matrix_A_host, mem_size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(matrix_B_device, matrix_B_host, mem_size_B, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    fprintf(stdout, "---------- step two ----------\n");
    fprintf(stdout, "Calc SGEMM use CUBLAS\n");
    
    cublasHandle_t handle;
    cublasCreate(&handle);
    const float alpha = 1.f;
    const float beta = 0.f;
    
    cudaEventRecord(start);
    
    for (int i = 0; i < nIter; i ++) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, matrix_B_device, n, matrix_A_device, k, &beta, matrix_C_device, m);
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float cbl_ms = 0.;
    cudaEventElapsedTime(&cbl_ms, start, stop);
    fprintf(stdout, "CUBLAS spend time: %f ms\n", cbl_ms / nIter);
    
    cudaMemcpy(matrix_C_host_cbl_calc, matrix_C_device, mem_size_C, cudaMemcpyDeviceToHost);
    cublasDestroy(handle);
    
    fprintf(stdout, "---------- step three ----------\n");
    fprintf(stdout, "Calc SGEMM use MyKernel\n");
    
    
    cudaEventRecord(start);
    
    for (int i = 0; i < nIter; i ++) {
        sgemm_g(m, n, k, matrix_A_device, matrix_B_device, matrix_C_device);
    }
    // cudaDeviceSynchronize();
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float mkl_ms = 0.;
    cudaEventElapsedTime(&mkl_ms, start, stop);
    fprintf(stdout, "MyKernel spend time: %f ms\n", mkl_ms / nIter);
    
    cudaMemcpy(matrix_C_host_mkl_calc, matrix_C_device, mem_size_C, cudaMemcpyDeviceToHost);
    
    fprintf(stdout, "---------- step four ----------\n");
    fprintf(stdout, "compare answer between CUBLAS and MyKernel\n");
    
    float diff = compare_matrices(matrix_C_host_mkl_calc, matrix_C_host_cbl_calc, m * n);
    fprintf(stdout, "max of diff between CUBLAS and MyKernel: %.3f\n", diff);
    
    if (diff < 0.5f) {
        fprintf(stdout, "\033[32m---------- Accept ----------\033[37m \n");
    } else {
        fprintf(stdout, "\033[31m---------- Wrong Answer ----------\033[37m \n");
    }

    free(matrix_A_host);
    free(matrix_B_host);
    free(matrix_C_host_cbl_calc);
    free(matrix_C_host_mkl_calc);
    
    cudaFree(matrix_A_device);
    cudaFree(matrix_B_device);
    cudaFree(matrix_C_device);
}
