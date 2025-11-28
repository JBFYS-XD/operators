#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <cfloat>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)


__device__ __forceinline__
float block_reduce_max(float val, float* smem) {
    
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(0xFFFFFFFF, val, offset);
        val = fmaxf(other, val);
    }

    if (lane == 0) {
        smem[wid] = val;
    }
    __syncthreads();

    if (wid == 0) {
        val = lane < (blockDim.x / 32) ? smem[lane] : FLT_MIN;
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other = __shfl_xor_sync(0xFFFFFFFF, val, offset);
            val = fmaxf(other, val);
        }
        if (lane == 0)
            smem[0] = val;   
    }
    __syncthreads();

    return smem[0];
}

__device__ __forceinline__
float block_reduce_sum(float val, float* smem) {
    
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(0xFFFFFFFF, val, offset);
        val += other;
    }

    if (lane == 0) {
        smem[wid] = val;
    }
    __syncthreads();

    if (wid == 0) {
        val = lane < (blockDim.x / 32) ? smem[lane] : 0.;
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other = __shfl_xor_sync(0xFFFFFFFF, val, offset);
            val += other;
        }
        if (lane == 0)
            smem[0] = val;   
    }
    __syncthreads();

    return smem[0];
}

__global__ void self_attentionKernel(
    const float* Q, const float* K, const float* V, float* O,
    int batch_size, int sequence_len, int number_heads, int head_dimension, 
    bool is_causal, float softmax_scale
) {

    auto mul_dim_pos = [=](int x, int y, int z, int w) {
        return x * sequence_len * number_heads * head_dimension +
                              y * number_heads * head_dimension +
                                             z * head_dimension +
                                                               w;
    };

    const int x = blockIdx.x;
    const int z = blockIdx.y;

    extern __shared__ unsigned char shared_mem[];
    float* q_k = reinterpret_cast<float*>(shared_mem);
    float* smem = q_k + sequence_len;

    for (int y1 = 0; y1 < sequence_len; y1 ++) {
        __syncthreads();
        float mx = FLT_MIN;
        for (int y2 = threadIdx.x; y2 < sequence_len; y2 += blockDim.x) {
            if (is_causal && y1 < y2) {
                q_k[y2] = 1e-8;
            } else {
                float res = 0.;
                for (int w = 0; w < head_dimension; w ++) {
                    res += Q[mul_dim_pos(x, y1, z, w)] * K[mul_dim_pos(x, y2, z, w)];
                }
                q_k[y2] = res * softmax_scale;
                mx = fmaxf(mx, q_k[y2]);
            }
        }
        __syncthreads();

        mx = block_reduce_max(mx, smem);

        float sum = 0.;
        for (int y2 = threadIdx.x; y2 < sequence_len; y2 += blockDim.x) {
            q_k[y2] = expf(q_k[y2] - mx);
            sum += q_k[y2];
        }
        __syncthreads();

        sum = block_reduce_sum(sum, smem);

        for (int y2 = threadIdx.x; y2 < sequence_len; y2 += blockDim.x) {
            q_k[y2] = q_k[y2] / sum;
        }
        __syncthreads();

        for (int w = threadIdx.x; w < head_dimension; w += blockDim.x) {
            float res = 0.;
            for (int y2 = 0; y2 < sequence_len; y2 ++) {
                res += q_k[y2] * V[mul_dim_pos(x, y2, z, w)];
            }
            O[mul_dim_pos(x, y1, z, w)] = res;
        }

    }
    
}


void self_attention(
    const float* Q, const float* K, const float* V, float* O, 
    int batch_size, int sequence_len, int number_heads, int head_dimension, 
    bool is_causal, float softmax_scale
) {

    int blockSize = 512;
    dim3 gridSize(batch_size, number_heads);
    size_t shared_mem = sequence_len * sizeof(float) + (blockSize / 32) * sizeof(float);
    self_attentionKernel<<<gridSize, blockSize, shared_mem>>>(
      Q, K, V, O,
      batch_size, sequence_len, number_heads, head_dimension, 
      is_causal, softmax_scale
    );

    CUDA_CHECK(cudaDeviceSynchronize());

}