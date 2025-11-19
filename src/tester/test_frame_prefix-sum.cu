#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include "../prefix-sum/prefix_sum_v2.cu"

#define LEN 1024 * 1024

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

void random_element(int n, float* a) {
    for (int i = 0; i < n; i ++) {
        a[i] = 0.02 * (float)drand48() - 1.0;
        // a[i] = 1.;
    }
}

float compare_matrices(float* output, float* correct, int n) {
    float max_diff = 0.;
    for (int i = 0; i < n; i++) {
        float diff = abs(output[i] - correct[i]);
        // float diff = abs(output[i] - (i + 1.0));
        max_diff = diff > max_diff ? diff : max_diff;
    }
    return max_diff;
}

int main() {
    const int n = LEN;
    const size_t mem_size_input = n * sizeof(float);
    const size_t mem_size_output = n * sizeof(float);

    int nIter = 1;

    fprintf(stdout, "---------- prefix sum test ----------\n");
    fprintf(stdout, "element array input: %d\n", n);

    fprintf(stdout, "---------- step one ----------\n");
    fprintf(stdout, "alloc memory and init data\n");

    float* input_host = (float*)malloc(mem_size_input);
    
    float* output_host_thr = (float*)malloc(mem_size_output);
    float* output_host_mkl = (float*)malloc(mem_size_output);

    random_element(n, input_host);

    float* input_device;
    float* output_device;

    cudaMalloc(&input_device, mem_size_input);
    cudaMalloc(&output_device, mem_size_output);

    cudaMemset(output_device, 0, mem_size_output);
    cudaMemcpy(input_device, input_host, mem_size_input, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    fprintf(stdout, "---------- step two ----------\n");
    fprintf(stdout, "Calc prefix sum use thrust\n");

    thrust::device_vector<float> input_thr(input_host, input_host + n);
    thrust::device_vector<float> output_thr(n);

    cudaEventRecord(start);

    for (int i = 0; i < nIter; i ++) {
        thrust::inclusive_scan(input_thr.begin(), input_thr.end(), output_thr.begin());
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float thr_ms = 0;
    cudaEventElapsedTime(&thr_ms, start, stop);
    fprintf(stdout, "Thrust spend time: %f ms\n", thr_ms / nIter);

    thrust::copy(output_thr.begin(), output_thr.end(), output_host_thr);

    fprintf(stdout, "---------- step three ----------\n");
    fprintf(stdout, "Calc prefix sum use MyKernel\n");

    cudaEventRecord(start);

    for (int i = 0; i < nIter; i ++) {
        prefix_sum(n, input_device, output_device);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float mkl_ms = 0.;
    cudaEventElapsedTime(&mkl_ms, start, stop);
    fprintf(stdout, "MyKernel spend time: %f ms\n", mkl_ms / nIter);

    cudaMemcpy(output_host_mkl, output_device, mem_size_output, cudaMemcpyDeviceToHost);

    fprintf(stdout, "---------- step four ----------\n");
    fprintf(stdout, "compare answer between Thrust and MyKernel\n");

    float diff = compare_matrices(output_host_mkl, output_host_thr, n);
    fprintf(stdout, "max of diff between Thrust and MyKernel: %.3f\n", diff);

    if (diff < LEN * 1e-6) {
        fprintf(stdout, "\033[32m---------- Accept ----------\033[37m \n");
    } else {
        fprintf(stdout, "\033[31m---------- Wrong Answer ----------\033[37m \n");
    }

    free(input_host);
    free(output_host_thr);
    free(output_host_mkl);

    cudaFree(input_device);
    cudaFree(output_device);
}