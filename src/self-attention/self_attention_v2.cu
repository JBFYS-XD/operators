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
void block_reduce_attn(float d, float m, float* smem) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;
    int warpSize = blockDim.x / 32;

    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_d = __shfl_xor_sync(0xFFFFFFFF, d, offset);
        float other_m = __shfl_xor_sync(0xFFFFFFFF, m, offset);
        float now_mx = fmaxf(m, other_m);

        d = d * expf(m - now_mx) + other_d * expf(other_m - now_mx);
        m = now_mx;
    }

    if (lane == 0) {
        smem[wid] = d;
        smem[wid + warpSize] = m;
    }
    __syncthreads();

    if (wid == 0) {
        d = lane < warpSize ? smem[lane] : 0.;
        m = lane < warpSize ? smem[lane + warpSize] : -1e9;
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_d = __shfl_xor_sync(0xFFFFFFFF, d, offset);
            float other_m = __shfl_xor_sync(0xFFFFFFFF, m, offset);
            float now_mx = fmaxf(m, other_m);

            d = d * expf(m - now_mx) + other_d * expf(other_m - now_mx);
            m = now_mx;
        }
        if (lane == 0) {
            smem[0] = d;
            smem[1] = m;
        }
    }

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
        float mx = -1e9;
        float d = 0.;
        for (int y2 = threadIdx.x; y2 < sequence_len; y2 += blockDim.x) {
            if (is_causal && y1 < y2) {
                q_k[y2] = 1e-8;
            } else {
                float res = 0.;
                for (int w = 0; w < head_dimension; w ++) {
                    res += Q[mul_dim_pos(x, y1, z, w)] * K[mul_dim_pos(x, y2, z, w)];
                }
                q_k[y2] = res * softmax_scale;
                float now_mx = fmaxf(mx, q_k[y2]);
                d = d * expf(mx - now_mx) + expf(q_k[y2] - now_mx);
                mx = now_mx;
            }
        }
        __syncthreads();

        block_reduce_attn(d, mx, smem);
        __syncthreads();

        d = smem[0];
        mx = smem[1];

        for (int y2 = threadIdx.x; y2 < sequence_len; y2 += blockDim.x) {
            q_k[y2] = expf(q_k[y2] - mx) / d;
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
    size_t shared_mem = sequence_len * sizeof(float) + (blockSize / 32) * 2 * sizeof(float);
    self_attentionKernel<<<gridSize, blockSize, shared_mem>>>(
      Q, K, V, O,
      batch_size, sequence_len, number_heads, head_dimension, 
      is_causal, softmax_scale
    );

    CUDA_CHECK(cudaDeviceSynchronize());

}