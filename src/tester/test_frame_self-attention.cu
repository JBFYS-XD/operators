#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <cfloat>

// #define USE_CUDNN


#define BATCH_SIZE 16
#define SEQUENCE_LEN 256
#define NUMBER_HEADS 12
#define HEAD_DIMENSION 64
#define IS_CAUSAL true


void random_element(int n, float* a) {
    for (int i = 0; i < n; i ++) {
        // a[i] = (float)drand48() * 2 - 1.;
        a[i] = (float)drand48();
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

#ifdef USE_CUDNN

#include <memory>
#include <cudnn_frontend.h>
#include <cudnn.h>
namespace fe = cudnn_frontend;

void self_attention_cudnn(
    float* Q, float* K, float* V, float* O, 
    int batch_size, int sequence_len, int number_heads, int head_dimension, bool is_causal,
    int Iter
);

#else

void self_attention_bycpu(    
    float* Q, float* K, float* V, float* O, 
    int batch_size, int sequence_len, int number_heads, int head_dimension, 
    bool is_causal, float softmax_scale
);

#endif

void self_attention(
    const float* Q, const float* K, const float* V, float* O, 
    int batch_size, int sequence_len, int number_heads, int head_dimension, 
    bool is_causal, float softmax_scale
);


int main() {
    constexpr int batch_size = BATCH_SIZE;
    constexpr int sequence_len = SEQUENCE_LEN;
    constexpr int number_heads = NUMBER_HEADS;
    constexpr int head_dimension = HEAD_DIMENSION;
    constexpr bool is_causal = IS_CAUSAL;
    float softmax_scale = 1.0;
    // float softmax_scale = 1.0 / sqrt(head_dimension);

    const size_t mem_size_Q = batch_size * number_heads * sequence_len * head_dimension * sizeof(float);
    const size_t mem_size_K = batch_size * number_heads * sequence_len * head_dimension * sizeof(float);
    const size_t mem_size_V = batch_size * number_heads * sequence_len * head_dimension * sizeof(float);
    const size_t mem_size_O = batch_size * number_heads * sequence_len * head_dimension * sizeof(float);

    int nIter = 1;

    fprintf(stdout, "---------- gemm test ----------\n");
    fprintf(stdout, "BATCH_SIZE: %d\n", batch_size);
    fprintf(stdout, "NUMBER_HEADS: %d\n", number_heads);
    fprintf(stdout, "SEQUENCE_LEN: %d\n", sequence_len);
    fprintf(stdout, "HEAD_DIMENSION: %d\n", head_dimension);

    fprintf(stdout, "---------- step one ----------\n");
    fprintf(stdout, "alloc memory and init data\n");

    float* Q_host = (float*)malloc(mem_size_Q);
    float* K_host = (float*)malloc(mem_size_K);
    float* V_host = (float*)malloc(mem_size_V);
    
    float* O_host_currt = (float*)malloc(mem_size_O);
    float* O_host_myknl = (float*)malloc(mem_size_O);

    random_element(batch_size * sequence_len * number_heads * head_dimension, Q_host);
    random_element(batch_size * sequence_len * number_heads * head_dimension, K_host);
    random_element(batch_size * sequence_len * number_heads * head_dimension, V_host);
    memset(O_host_currt, 0, mem_size_O);
    memset(O_host_myknl, 0, mem_size_O);

    float* Q_device;
    float* K_device;
    float* V_device;
    float* O_device;

    cudaMalloc(&Q_device, mem_size_Q);
    cudaMalloc(&K_device, mem_size_K);
    cudaMalloc(&V_device, mem_size_V);
    cudaMalloc(&O_device, mem_size_O);

    cudaMemset(O_device, 0, sizeof(mem_size_O));

    cudaMemcpy(Q_device, Q_host, mem_size_Q, cudaMemcpyHostToDevice);
    cudaMemcpy(K_device, K_host, mem_size_K, cudaMemcpyHostToDevice);
    cudaMemcpy(V_device, V_host, mem_size_V, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    fprintf(stdout, "---------- step two ----------\n");
    fprintf(stdout, "Calc self-attention use CUDNN\n");

    #ifdef USE_CUDNN

    cudaEventRecord(start);

    self_attention_cudnn(
        Q_device, K_device, V_device, O_device, 
        batch_size, sequence_len, number_heads, head_dimension, is_causal, 
        nIter
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float cdn_ms = 0.;
    cudaEventElapsedTime(&cdn_ms, start, stop);
    fprintf(stdout, "CUBLAS spend time: %f ms\n", cdn_ms / nIter);
    cudaMemcpy(O_host_currt, O_device, mem_size_O, cudaMemcpyDeviceToHost);

    #else


    fprintf(stdout, "\033[31m");
    fprintf(stdout, "Not set up to use cudnn, The CPU will be used for calculations but performance will not be evaluated\n");
    fprintf(stdout, "\033[37m");

    self_attention_bycpu(
        Q_host, K_host, V_host, O_host_currt,
        batch_size, sequence_len, number_heads, head_dimension, 
        is_causal, softmax_scale
    );

    #endif

    fprintf(stdout, "---------- step three ----------\n");
    fprintf(stdout, "Calc self-attention use MyKernel\n");


    cudaEventRecord(start);

    for (int i = 0; i < nIter; i ++) {
        self_attention(
            Q_device, K_device, V_device, O_device,
            batch_size, sequence_len, number_heads, head_dimension, 
            is_causal, softmax_scale
        );
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float mkl_ms = 0.;
    cudaEventElapsedTime(&mkl_ms, start, stop);
    fprintf(stdout, "MyKernel spend time: %f ms\n", mkl_ms / nIter);

    cudaMemcpy(O_host_myknl, O_device, mem_size_O, cudaMemcpyDeviceToHost);

    fprintf(stdout, "---------- step four ----------\n");
    fprintf(stdout, "compare answer between correct and MyKernel\n");

    float diff = compare_matrices(O_host_myknl, O_host_currt, batch_size * sequence_len * number_heads * head_dimension);
    fprintf(stdout, "max of diff between CUBLAS and MyKernel: %.3f\n", diff);

    if (diff < 0.5f) {
        fprintf(stdout, "\033[32m---------- Accept ----------\033[37m \n");
    } else {
        fprintf(stdout, "\033[31m---------- Wrong Answer ----------\033[37m \n");
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);


    free(Q_host);
    free(K_host);
    free(V_host);
    free(O_host_currt);
    free(O_host_myknl);

    cudaFree(Q_device);
    cudaFree(K_device);
    cudaFree(V_device);
    cudaFree(O_device);

    return 0;
}

#ifdef USE_CUDNN

template<typename INP_F, typename OTP_F>
__global__ void input_copy(
    INP_F* input_Q, INP_F* input_K, INP_F* input_V,
    OTP_F* output_Q, OTP_F* output_K, OTP_F* output_V, int N
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        output_Q[idx] = input_Q[idx];
        output_K[idx] = input_K[idx];
        output_V[idx] = input_V[idx];
    }
}

template<typename INP_F, typename OTP_F>
__global__ void output_copy(
    INP_F* input_O, OTP_F* output_O, int N
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        output_O[idx] = input_O[idx];
    }
}


auto build_graph(cudnnHandle_t cudnn_handle, int B, int S, int NH, int HS, bool is_causal) {

    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::HALF)
            .set_intermediate_data_type(fe::DataType_t::FLOAT)
            .set_compute_data_type(fe::DataType_t::FLOAT);

    auto Q = graph->tensor(fe::graph::Tensor_attributes()
                            .set_name("Q")
                            .set_dim({B, NH, S, HS})
                            .set_stride({NH * S * HS, S * HS, HS, 1})
                            .set_data_type(fe::DataType_t::HALF));
    auto K = graph->tensor(fe::graph::Tensor_attributes()
                            .set_name("K")
                            .set_dim({B, NH, S, HS})
                            .set_stride({NH * S * HS, S * HS, HS, 1})
                            .set_data_type(fe::DataType_t::HALF));
    auto V = graph->tensor(fe::graph::Tensor_attributes()
                            .set_name("V")
                            .set_dim({B, NH, S, HS})
                            .set_stride({NH * S * HS, S * HS, HS, 1})
                            .set_data_type(fe::DataType_t::HALF));

    // auto attn_scale = graph->tensor(fe::graph::Tensor_attributes()
    //                         .set_name("attn_scale")
    //                         .set_dim({1, 1, 1, 1})
    //                         .set_stride({1, 1, 1, 1})
    //                         .set_is_pass_by_value(true)
    //                         .set_data_type(fe::DataType_t::FLOAT));
    
    auto sdpa_options = fe::graph::SDPA_attributes()
                            .set_generate_stats(false)
                            .set_attn_scale(1.0f)
                            .set_causal_mask(is_causal);
    
    auto O = graph->sdpa(Q, K, V, sdpa_options)[0];

    O->set_output(true)
                            .set_dim({B, NH, S, HS})
                            .set_stride({NH * S * HS, S * HS, HS, 1})
                            .set_data_type(fe::DataType_t::HALF);

    assert(graph->validate().is_good());
    
    assert(graph->build_operation_graph(cudnn_handle).is_good());
    auto plans = graph->create_execution_plans({fe::HeurMode_t::A});
    auto status = graph->check_support(cudnn_handle);
    if (!status.is_good()) {
        std::cerr << "check_support failed!" << "\n";
        std::cerr << "Error: " << status << "\n";
        assert(false);
    }
    assert(graph->check_support(cudnn_handle).is_good());
    assert(graph->build_plans(cudnn_handle).is_good());

    auto tuple = std::make_tuple(graph, Q, K, V, O);

    return tuple;
}

void self_attention_cudnn(
    float* Q, float* K, float* V, float* O, 
    int batch_size, int sequence_len, int number_heads, int head_dimension, bool is_causal,
    int Iter
) {
    half* inp_Q;
    half* inp_K;
    half* inp_V;
    half* otp_O;
    // float attn_scale = 1.0;

    size_t num_elem = batch_size * sequence_len * number_heads * head_dimension;

    cudaMalloc(&inp_Q, sizeof(half) * num_elem);
    cudaMalloc(&inp_K, sizeof(half) * num_elem);
    cudaMalloc(&inp_V, sizeof(half) * num_elem);
    cudaMalloc(&otp_O, sizeof(half) * num_elem);

    dim3 blockSize(1024);
    dim3 gridSize((num_elem + blockSize.x - 1) / blockSize.x);

    input_copy<<<blockSize, gridSize>>>(Q, K, V, inp_Q, inp_K, inp_V, num_elem);
    cudaDeviceSynchronize();

    cudnnHandle_t cudnn_handle;
    cudnnCreate(&cudnn_handle);


    auto [graph, tensor_Q, tensor_K, tensor_V, tensor_O] =
        build_graph(cudnn_handle, batch_size, sequence_len, number_heads, head_dimension, is_causal);

    std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
        {tensor_Q, inp_Q}, {tensor_K, inp_K}, {tensor_V, inp_V}, 
        {tensor_O, otp_O}
    };

    void* cudnn_workspace_size;

    cudaMalloc(&cudnn_workspace_size, graph->get_workspace_size());

    while (Iter --)
        assert(graph->execute(cudnn_handle, variant_pack, cudnn_workspace_size).is_good());

    output_copy<<<blockSize, gridSize>>>(O, otp_O, num_elem);
    cudaDeviceSynchronize();

    cudnnDestroy(cudnn_handle);
}

#else


void self_attention_bycpu(    
    float* Q, float* K, float* V, float* O, 
    int batch_size, int sequence_len, int number_heads, int head_dimension, 
    bool is_causal, float softmax_scale
) {

    auto mul_dim_pos = [=](int x, int y, int z, int w) {
        return x * sequence_len * number_heads * head_dimension +
                              y * number_heads * head_dimension +
                                             z * head_dimension +
                                                               w;
    };

    float* q_k = (float*)malloc(sizeof(float) * sequence_len);


    for (int x = 0; x < batch_size; x ++) {
        for (int z = 0; z < number_heads; z ++) {
            for (int y1 = 0; y1 < sequence_len; y1 ++) {
            
                // Compute the y1 row of the matrix 
                memset(q_k, 0, sizeof(float) * sequence_len);
                float val_max = FLT_MIN;
                for (int y2 = 0; y2 < sequence_len; y2 ++) {
                    float res = 0.;
                    for (int w = 0; w < head_dimension; w ++) {
                        res += Q[mul_dim_pos(x, y1, z, w)] * K[mul_dim_pos(x, y2, z, w)];
                    }
                    if (is_causal && y2 > y1)
                        q_k[y2] = -1e9f;
                    else
                        q_k[y2] = res * softmax_scale;
                    // Get the maximum value
                    val_max = std::fmax(val_max, q_k[y2]);
                }
                
                // Calculate e^q_k and get the sum of the values
                float flt_sum = 0.;
                for (int y2 = 0; y2 < sequence_len; y2 ++) {
                    q_k[y2] = std::exp(q_k[y2] - val_max);
                    flt_sum += q_k[y2];
                }
                
                for (int y2 = 0; y2 < sequence_len; y2 ++) {
                    q_k[y2] /= flt_sum + 1e-8;
                }

                for (int w = 0; w < head_dimension; w ++) {
                    float res = 0.f;
                    for (int y2 = 0; y2 < sequence_len; y2 ++) {
                        res += q_k[y2] * V[mul_dim_pos(x, y2, z, w)];
                    }
                    O[mul_dim_pos(x, y1, z, w)] = res;
                }
            }
        }
    }
}

#endif