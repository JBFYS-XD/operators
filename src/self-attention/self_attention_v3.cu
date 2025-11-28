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


__global__ 
void self_attentionKernel(
    const float* Q, const float* K, const float* V, float* O, float* D, float* M,
    int batch_size, int sequence_len, int number_heads, int head_dimension, 
    bool is_causal, float softmax_scale,
    int block_rows, int block_cols
) {

    // 索引 x, y, z, w 分别对应 
    // batch_size, sequence_len, number_heads, head_dimension,
    auto feature_pos = [=](int x, int y, int z, int w) {
        return x * sequence_len * number_heads * head_dimension +
                              y * number_heads * head_dimension +
                                             z * head_dimension +
                                                               w;
    };

    auto D_M_pos = [=](int x, int y, int z) {
        return x * number_heads * sequence_len +
                              z * sequence_len +
                                              y;
    };

    const int x = blockIdx.x;
    const int z = blockIdx.y;
    const int tid = threadIdx.x;

    for (int y = tid; y < sequence_len; y += block_rows) {
        D[D_M_pos(x, y, z)] = 0.;
        M[D_M_pos(x, y, z)] = -1e9;
    }

    extern __shared__ unsigned char shared_mem[];
    float* smem_Q = reinterpret_cast<float*>(shared_mem);
    float* smem_K = smem_Q + block_rows * head_dimension;
    float* smem_V = smem_K + block_cols * head_dimension;
    float* smem_S = smem_V + block_cols * head_dimension; // block_rows * block_cols


    for (int y2_start = 0; y2_start  < sequence_len + block_cols - 1; y2_start += block_cols) {
        for (int w = 0; w < head_dimension; w ++) {
            int y2 = y2_start + tid;
            // if (tid > block_cols)   break;
            smem_K[tid * head_dimension + w] = y2 < sequence_len ? K[feature_pos(x, y2, z, w)] : 0.;
            smem_V[tid * head_dimension + w] = y2 < sequence_len ? V[feature_pos(x, y2, z, w)] : 0;
        }

        for (int y1_start = 0; y1_start < sequence_len + block_rows - 1; y1_start += block_rows) {
            int y1 = y1_start + tid;
            if (y1 >= sequence_len) break;

            for (int w = 0; w < head_dimension; w ++) {
                smem_Q[tid * head_dimension + w] = Q[feature_pos(x, y1_start + tid, z, w)];
            }
            __syncthreads();

            float m = -1e9;
            float d = 0.;
            for (int y2 = y2_start; y2 < y2_start + block_cols; y2 ++) {
                if (y2 >= sequence_len) break;
                
                int smem_y1 = tid, smem_y2 = y2 - y2_start;

                if (!(is_causal && y1 < y2)) {
                    float sum = 0.;
                    for (int w = 0; w < head_dimension; w ++) {
                        sum += smem_Q[smem_y1 * head_dimension + w] * smem_K[smem_y2 * head_dimension + w];
                    }
                    sum *= softmax_scale;
                    m = fmaxf(m, sum);
                    smem_S[(smem_y1 * block_cols) + smem_y2] = sum;
                } else {
                    smem_S[(smem_y1 * block_cols) + smem_y2] = -1e9;
                }

            }

            for (int y2 = y2_start; y2 < y2_start + block_cols; y2 ++) {
                if (y2 >= sequence_len) break;

                int smem_y1 = tid, smem_y2 = y2 - y2_start;

                float& now_S = smem_S[(smem_y1 * block_cols) + smem_y2];
                now_S = __expf(now_S - m);

                d += now_S;
            }

            float pre_m = M[D_M_pos(x, y1, z)];
            float pre_d = D[D_M_pos(x, y1, z)];

            float now_m = fmaxf(m, pre_m);
            float now_d = pre_d * __expf(pre_m - now_m) + d * __expf(m - now_m);

            for (int w = 0; w < head_dimension; w ++) {
                float sum = 0.;
                int smem_y1 = tid;
                for (int y2 = y2_start; y2 < y2_start + block_cols; y2 ++) {
                    if (y2 >= sequence_len) break;
                    
                    int smem_y2 = y2 - y2_start;

                    sum += smem_S[(smem_y1 * block_cols) + smem_y2] * smem_V[smem_y2 * head_dimension + w];

                }

                float& o = O[feature_pos(x, y1, z, w)];
                o = o * pre_d * __expf(pre_m - now_m) / now_d + sum * (__expf(m - now_m)) / now_d;
            }

            M[D_M_pos(x, y1, z)] = now_m;
            D[D_M_pos(x, y1, z)] = now_d;

        }
        __syncthreads();
    }
    
}


void self_attention(
    const float* Q, const float* K, const float* V, float* O, 
    int batch_size, int sequence_len, int number_heads, int head_dimension, 
    bool is_causal, float softmax_scale
) {

    int block_rows = 32, block_cols = 32;
    dim3 blockSize(block_rows);
    dim3 gridSize(batch_size, number_heads);
    
    float* D; float* M;
    cudaMalloc(&D, batch_size * number_heads * sequence_len * sizeof(float));
    cudaMalloc(&M, batch_size * number_heads * sequence_len * sizeof(float));

    cudaMemset(O, 0, batch_size * sequence_len * number_heads * head_dimension * sizeof(float));

    size_t shared_mem = (block_rows * head_dimension +
                         block_cols * head_dimension * 2 +
                         block_rows * block_cols) * sizeof(float);
    self_attentionKernel<<<gridSize, blockSize, shared_mem>>>(
      Q, K, V, O, D, M,
      batch_size, sequence_len, number_heads, head_dimension, 
      is_causal, softmax_scale,
      block_rows, block_cols
    );

    CUDA_CHECK(cudaDeviceSynchronize());

}