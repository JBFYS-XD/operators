#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>

#define LEN 32 * 1024 * 1024

extern float reduce(int n, float* input);

void random_element(int n, float* a) {
    for (int i = 0; i < n; i ++) {
        // a[i] = 2.0 * (float)drand48() - 1.0;
        a[i] = 1.;
    }
}

int main() {
    const int n = LEN;
    const size_t mem_size_input = n * sizeof(float);

    int nIter = 1;

    fprintf(stdout, "---------- reduce test ----------\n");
    fprintf(stdout, "element array input: %d\n", n);

    fprintf(stdout, "---------- step one ----------\n");
    fprintf(stdout, "alloc memory and init data\n");

    float* input_host = (float*)malloc(mem_size_input);
    float output_thr;
    float output_mkl;

    random_element(n, input_host);

    float* input_device;
    cudaMalloc(&input_device, mem_size_input);
    cudaMemcpy(input_device, input_host, mem_size_input, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    fprintf(stdout, "---------- step two ----------\n");
    fprintf(stdout, "Calc reduce use thrust\n");

    thrust::device_vector<float> input_thr(input_host, input_host + n);
    
    cudaEventRecord(start);

    for (int i = 0; i < nIter; i ++) {
        output_thr = thrust::reduce(thrust::device, input_thr.begin(), input_thr.end(), 0.f, thrust::plus<float>());
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float thr_ms = 0;
    cudaEventElapsedTime(&thr_ms, start, stop);
    fprintf(stdout, "Thrust spend time: %f ms\n", thr_ms / nIter);

    fprintf(stdout, "---------- step three ----------\n");
    fprintf(stdout, "Calc reduce use MyKernel\n");

    cudaEventRecord(start);

    for (int i = 0; i < nIter; i ++) {
        output_mkl = reduce(n, input_device);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float mkl_ms = 0.;
    cudaEventElapsedTime(&mkl_ms, start, stop);
    fprintf(stdout, "MyKernel spend time: %f ms\n", mkl_ms / nIter);

    fprintf(stdout, "---------- step four ----------\n");
    fprintf(stdout, "compare answer between Thrust and MyKernel\n");

    float diff = abs(output_mkl - output_thr);
    fprintf(stdout, "diff between Thrust and MyKernel: %.3f\n", diff);

    if (diff < 0.5f) {
        fprintf(stdout, "\033[32m---------- Accept ----------\033[37m \n");
    } else {
        fprintf(stdout, "\033[31m---------- Wrong Answer ----------\033[37m \n");
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    free(input_host);

    cudaFree(input_device);
}