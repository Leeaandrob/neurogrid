// layer.cpp - Single transformer layer forward pass
// Implements the complete Llama layer architecture

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "layer.h"
#include "../cuda/kernels.h"
#include "../cuda/matmul.h"
#include "../cuda/attention.h"
#include "../cuda/memory.h"

// Error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        return -1; \
    } \
} while(0)

// ============================================================================
// Layer Weights Structure
// ============================================================================

struct LayerWeights {
    // Normalization weights (FP16)
    half* attn_norm;        // [hidden_size]
    half* ffn_norm;         // [hidden_size]

    // Attention weights (INT8 quantized)
    // Stored as [out_dim, in_dim], GEMM uses transpose_b=true
    int8_t* q_proj;         // [hidden_size, hidden_size] (out=hidden, in=hidden)
    int8_t* k_proj;         // [kv_dim, hidden_size] (out=kv_dim, in=hidden)
    int8_t* v_proj;         // [kv_dim, hidden_size] (out=kv_dim, in=hidden)
    int8_t* o_proj;         // [hidden_size, hidden_size] (out=hidden, in=hidden)

    // Attention scales (FP32) - per output dimension
    float* q_scale;         // [hidden_size]
    float* k_scale;         // [kv_dim]
    float* v_scale;         // [kv_dim]
    float* o_scale;         // [hidden_size]

    // FFN weights (INT8 quantized)
    // Stored as [out_dim, in_dim], GEMM uses transpose_b=true
    int8_t* gate_proj;      // [intermediate_size, hidden_size] (out=inter, in=hidden)
    int8_t* up_proj;        // [intermediate_size, hidden_size] (out=inter, in=hidden)
    int8_t* down_proj;      // [hidden_size, intermediate_size] (out=hidden, in=inter)

    // FFN scales (FP32) - per output dimension
    float* gate_scale;      // [intermediate_size]
    float* up_scale;        // [intermediate_size]
    float* down_scale;      // [hidden_size]

    // Dimensions
    int hidden_size;
    int intermediate_size;
    int num_heads;
    int num_kv_heads;
    int head_dim;
};

// ============================================================================
// Weight Management
// ============================================================================

extern "C" void cuda_free_layer_weights(void* weights) {
    if (!weights) return;

    LayerWeights* w = (LayerWeights*)weights;

    // Free normalization weights
    if (w->attn_norm) cudaFree(w->attn_norm);
    if (w->ffn_norm) cudaFree(w->ffn_norm);

    // Free attention weights
    if (w->q_proj) cudaFree(w->q_proj);
    if (w->k_proj) cudaFree(w->k_proj);
    if (w->v_proj) cudaFree(w->v_proj);
    if (w->o_proj) cudaFree(w->o_proj);

    // Free attention scales
    if (w->q_scale) cudaFree(w->q_scale);
    if (w->k_scale) cudaFree(w->k_scale);
    if (w->v_scale) cudaFree(w->v_scale);
    if (w->o_scale) cudaFree(w->o_scale);

    // Free FFN weights
    if (w->gate_proj) cudaFree(w->gate_proj);
    if (w->up_proj) cudaFree(w->up_proj);
    if (w->down_proj) cudaFree(w->down_proj);

    // Free FFN scales
    if (w->gate_scale) cudaFree(w->gate_scale);
    if (w->up_scale) cudaFree(w->up_scale);
    if (w->down_scale) cudaFree(w->down_scale);

    free(w);
}

extern "C" int cuda_load_layer_weights(void** weights, const char* path) {
    // TODO: Implement safetensors loading
    // For now, return error - weights must be created with create_random_layer_weights
    fprintf(stderr, "load_layer_weights not implemented. Use create_random_layer_weights for testing.\n");
    return -1;
}

// Forward declaration of quantization function from quantize.cu
extern "C" int cuda_quantize_weights_per_row(
    void* output,         // INT8 [rows, cols]
    void* scales,         // FP32 [rows] - one scale per row (per output channel)
    const void* input,    // FP16 [rows, cols]
    int rows,
    int cols
);

// Helper: Upload FP16 data from host to device and quantize to INT8
// Allocates both INT8 weight and FP32 scale on GPU
// Weights are stored as [out_dim, in_dim], quantizes per-row (per output dimension)
static int upload_and_quantize(
    int8_t** d_weight_int8,
    float** d_scale,
    const void* h_weight_fp16,
    int out_dim,    // rows = output dimension
    int in_dim      // cols = input dimension
) {
    // Allocate temporary FP16 buffer on GPU
    half* d_weight_fp16;
    size_t fp16_size = (size_t)out_dim * in_dim * sizeof(half);
    CUDA_CHECK(cudaMalloc(&d_weight_fp16, fp16_size));

    // Copy FP16 from host to device
    CUDA_CHECK(cudaMemcpy(d_weight_fp16, h_weight_fp16, fp16_size, cudaMemcpyHostToDevice));

    // Allocate INT8 weight and scale on GPU
    // scales[out_dim] - one scale per output dimension (per row)
    CUDA_CHECK(cudaMalloc(d_weight_int8, (size_t)out_dim * in_dim * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(d_scale, out_dim * sizeof(float)));

    // Quantize per-row (per output dimension)
    int result = cuda_quantize_weights_per_row(*d_weight_int8, *d_scale, d_weight_fp16, out_dim, in_dim);

    // Free temporary FP16 buffer
    cudaFree(d_weight_fp16);

    return result;
}

// Helper: Upload FP16 norm weights (no quantization needed)
static int upload_fp16_norm(
    half** d_norm,
    const void* h_norm,
    int size
) {
    CUDA_CHECK(cudaMalloc(d_norm, size * sizeof(half)));
    CUDA_CHECK(cudaMemcpy(*d_norm, h_norm, size * sizeof(half), cudaMemcpyHostToDevice));
    return 0;
}

extern "C" int cuda_create_layer_weights_from_host(
    void** weights,
    // All weights are stored as [out_dim, in_dim] (PyTorch convention)
    const void* h_q_proj,       // FP16 [hidden_size, hidden_size]
    const void* h_k_proj,       // FP16 [kv_dim, hidden_size]
    const void* h_v_proj,       // FP16 [kv_dim, hidden_size]
    const void* h_o_proj,       // FP16 [hidden_size, hidden_size]
    const void* h_gate_proj,    // FP16 [intermediate_size, hidden_size]
    const void* h_up_proj,      // FP16 [intermediate_size, hidden_size]
    const void* h_down_proj,    // FP16 [hidden_size, intermediate_size]
    const void* h_attn_norm,    // FP16 [hidden_size]
    const void* h_ffn_norm,     // FP16 [hidden_size]
    int hidden_size,
    int intermediate_size,
    int num_heads,
    int num_kv_heads,
    int head_dim
) {
    LayerWeights* w = (LayerWeights*)malloc(sizeof(LayerWeights));
    if (!w) return -1;

    memset(w, 0, sizeof(LayerWeights));
    w->hidden_size = hidden_size;
    w->intermediate_size = intermediate_size;
    w->num_heads = num_heads;
    w->num_kv_heads = num_kv_heads;
    w->head_dim = head_dim;

    int kv_dim = num_kv_heads * head_dim;
    int result;

    // Upload and quantize attention weights
    // All weights stored as [out_dim, in_dim], quantize per-row (per output dimension)
    // Scales array has one scale per output channel (out_dim)

    // Q projection: [hidden_size, hidden_size] -> out=hidden, in=hidden
    result = upload_and_quantize(&w->q_proj, &w->q_scale, h_q_proj, hidden_size, hidden_size);
    if (result != 0) goto cleanup;

    // K projection: [kv_dim, hidden_size] -> out=kv_dim, in=hidden
    result = upload_and_quantize(&w->k_proj, &w->k_scale, h_k_proj, kv_dim, hidden_size);
    if (result != 0) goto cleanup;

    // V projection: [kv_dim, hidden_size] -> out=kv_dim, in=hidden
    result = upload_and_quantize(&w->v_proj, &w->v_scale, h_v_proj, kv_dim, hidden_size);
    if (result != 0) goto cleanup;

    // O projection: [hidden_size, hidden_size] -> out=hidden, in=hidden
    result = upload_and_quantize(&w->o_proj, &w->o_scale, h_o_proj, hidden_size, hidden_size);
    if (result != 0) goto cleanup;

    // Upload and quantize FFN weights
    // Gate projection: [intermediate_size, hidden_size] -> out=inter, in=hidden
    result = upload_and_quantize(&w->gate_proj, &w->gate_scale, h_gate_proj, intermediate_size, hidden_size);
    if (result != 0) goto cleanup;

    // Up projection: [intermediate_size, hidden_size] -> out=inter, in=hidden
    result = upload_and_quantize(&w->up_proj, &w->up_scale, h_up_proj, intermediate_size, hidden_size);
    if (result != 0) goto cleanup;

    // Down projection: [hidden_size, intermediate_size] -> out=hidden, in=inter
    result = upload_and_quantize(&w->down_proj, &w->down_scale, h_down_proj, hidden_size, intermediate_size);
    if (result != 0) goto cleanup;

    // Upload normalization weights (FP16, no quantization)
    result = upload_fp16_norm(&w->attn_norm, h_attn_norm, hidden_size);
    if (result != 0) goto cleanup;

    result = upload_fp16_norm(&w->ffn_norm, h_ffn_norm, hidden_size);
    if (result != 0) goto cleanup;

    CUDA_CHECK(cudaDeviceSynchronize());

    *weights = w;
    return 0;

cleanup:
    cuda_free_layer_weights(w);
    return -1;
}

// Random initialization kernel
__global__ void init_random_fp16(half* data, int n, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Simple pseudo-random based on index and seed
        unsigned int x = idx ^ seed;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        float val = (float)(x & 0xFFFF) / 65535.0f * 0.02f - 0.01f;
        data[idx] = __float2half(val);
    }
}

__global__ void init_random_int8(int8_t* data, int n, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        unsigned int x = idx ^ seed;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        data[idx] = (int8_t)((x % 255) - 127);
    }
}

__global__ void init_constant_fp32(float* data, int n, float val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = val;
    }
}

extern "C" int cuda_create_random_layer_weights(
    void** weights,
    int hidden_size,
    int intermediate_size,
    int num_heads,
    int num_kv_heads,
    int head_dim
) {
    LayerWeights* w = (LayerWeights*)malloc(sizeof(LayerWeights));
    if (!w) return -1;

    memset(w, 0, sizeof(LayerWeights));
    w->hidden_size = hidden_size;
    w->intermediate_size = intermediate_size;
    w->num_heads = num_heads;
    w->num_kv_heads = num_kv_heads;
    w->head_dim = head_dim;

    int kv_dim = num_kv_heads * head_dim;

    int block_size = 256;

    // Allocate and initialize normalization weights
    CUDA_CHECK(cudaMalloc(&w->attn_norm, hidden_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&w->ffn_norm, hidden_size * sizeof(half)));

    // Initialize norm weights to 1.0
    __global__ void init_ones_fp16(half* data, int n);
    // Use constant init
    int num_blocks = (hidden_size + block_size - 1) / block_size;
    init_random_fp16<<<num_blocks, block_size>>>(w->attn_norm, hidden_size, 42);
    init_random_fp16<<<num_blocks, block_size>>>(w->ffn_norm, hidden_size, 43);

    // Allocate attention weights (INT8)
    CUDA_CHECK(cudaMalloc(&w->q_proj, hidden_size * hidden_size * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&w->k_proj, hidden_size * kv_dim * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&w->v_proj, hidden_size * kv_dim * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&w->o_proj, hidden_size * hidden_size * sizeof(int8_t)));

    // Initialize attention weights
    num_blocks = (hidden_size * hidden_size + block_size - 1) / block_size;
    init_random_int8<<<num_blocks, block_size>>>(w->q_proj, hidden_size * hidden_size, 100);
    init_random_int8<<<num_blocks, block_size>>>(w->o_proj, hidden_size * hidden_size, 103);

    num_blocks = (hidden_size * kv_dim + block_size - 1) / block_size;
    init_random_int8<<<num_blocks, block_size>>>(w->k_proj, hidden_size * kv_dim, 101);
    init_random_int8<<<num_blocks, block_size>>>(w->v_proj, hidden_size * kv_dim, 102);

    // Allocate attention scales
    CUDA_CHECK(cudaMalloc(&w->q_scale, hidden_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&w->k_scale, kv_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&w->v_scale, kv_dim * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&w->o_scale, hidden_size * sizeof(float)));

    // Initialize scales to 0.01 (typical quantization scale)
    num_blocks = (hidden_size + block_size - 1) / block_size;
    init_constant_fp32<<<num_blocks, block_size>>>(w->q_scale, hidden_size, 0.01f);
    init_constant_fp32<<<num_blocks, block_size>>>(w->o_scale, hidden_size, 0.01f);

    num_blocks = (kv_dim + block_size - 1) / block_size;
    init_constant_fp32<<<num_blocks, block_size>>>(w->k_scale, kv_dim, 0.01f);
    init_constant_fp32<<<num_blocks, block_size>>>(w->v_scale, kv_dim, 0.01f);

    // Allocate FFN weights (INT8)
    CUDA_CHECK(cudaMalloc(&w->gate_proj, hidden_size * intermediate_size * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&w->up_proj, hidden_size * intermediate_size * sizeof(int8_t)));
    CUDA_CHECK(cudaMalloc(&w->down_proj, intermediate_size * hidden_size * sizeof(int8_t)));

    num_blocks = (hidden_size * intermediate_size + block_size - 1) / block_size;
    init_random_int8<<<num_blocks, block_size>>>(w->gate_proj, hidden_size * intermediate_size, 200);
    init_random_int8<<<num_blocks, block_size>>>(w->up_proj, hidden_size * intermediate_size, 201);

    num_blocks = (intermediate_size * hidden_size + block_size - 1) / block_size;
    init_random_int8<<<num_blocks, block_size>>>(w->down_proj, intermediate_size * hidden_size, 202);

    // Allocate FFN scales
    CUDA_CHECK(cudaMalloc(&w->gate_scale, intermediate_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&w->up_scale, intermediate_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&w->down_scale, hidden_size * sizeof(float)));

    num_blocks = (intermediate_size + block_size - 1) / block_size;
    init_constant_fp32<<<num_blocks, block_size>>>(w->gate_scale, intermediate_size, 0.01f);
    init_constant_fp32<<<num_blocks, block_size>>>(w->up_scale, intermediate_size, 0.01f);

    num_blocks = (hidden_size + block_size - 1) / block_size;
    init_constant_fp32<<<num_blocks, block_size>>>(w->down_scale, hidden_size, 0.01f);

    CUDA_CHECK(cudaDeviceSynchronize());

    *weights = w;
    return 0;
}

// ============================================================================
// Layer Forward
// ============================================================================

extern "C" int cuda_layer_forward(
    void* output,
    const void* input,
    const void* weights,
    void* kv_cache,
    const int* positions,
    int batch_size,
    int seq_len,
    int hidden_size,
    int intermediate_size,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float rms_norm_eps
) {
    LayerWeights* w = (LayerWeights*)weights;

    int num_tokens = batch_size * seq_len;
    int kv_dim = num_kv_heads * head_dim;

    // Allocate temporary buffers
    half* normed;
    half* q;
    half* k;
    half* v;
    half* attn_out;
    half* residual;
    half* gate;
    half* up;

    CUDA_CHECK(cudaMalloc(&normed, num_tokens * hidden_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&q, num_tokens * hidden_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&k, num_tokens * kv_dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&v, num_tokens * kv_dim * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&attn_out, num_tokens * hidden_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&residual, num_tokens * hidden_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&gate, num_tokens * intermediate_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&up, num_tokens * intermediate_size * sizeof(half)));

    int result;

    // Debug helper - check a buffer for NaN
    auto checkNaN = [](const half* d_buf, int size, const char* name) {
        half* h_buf = (half*)malloc(size * sizeof(half));
        cudaMemcpy(h_buf, d_buf, size * sizeof(half), cudaMemcpyDeviceToHost);
        for (int i = 0; i < size && i < 100; i++) {
            float val = __half2float(h_buf[i]);
            if (isnan(val) || isinf(val)) {
                fprintf(stderr, "[CUDA DEBUG] %s has NaN/Inf at idx %d: %.6f\n", name, i, val);
                free(h_buf);
                return true;
            }
        }
        // Print first few values
        fprintf(stderr, "[CUDA DEBUG] %s first 5 values: %.4f, %.4f, %.4f, %.4f, %.4f\n",
                name,
                __half2float(h_buf[0]), __half2float(h_buf[1]),
                __half2float(h_buf[2]), __half2float(h_buf[3]),
                __half2float(h_buf[4]));
        free(h_buf);
        return false;
    };

    // Check input
    checkNaN((const half*)input, hidden_size, "INPUT");

    // Save residual
    CUDA_CHECK(cudaMemcpy(residual, input, num_tokens * hidden_size * sizeof(half),
                          cudaMemcpyDeviceToDevice));

    // Check attn_norm weights
    checkNaN(w->attn_norm, hidden_size, "ATTN_NORM_W");

    // 1. Input RMSNorm
    result = cuda_rmsnorm(normed, input, w->attn_norm, num_tokens, hidden_size, rms_norm_eps);
    checkNaN(normed, hidden_size, "AFTER_RMSNORM");
    if (result != 0) goto cleanup;

    // Debug: Check weight scales
    {
        float* h_scale = (float*)malloc(hidden_size * sizeof(float));
        cudaMemcpy(h_scale, w->q_scale, hidden_size * sizeof(float), cudaMemcpyDeviceToHost);
        fprintf(stderr, "[CUDA DEBUG] Q_SCALE first 5: %.6f, %.6f, %.6f, %.6f, %.6f\n",
                h_scale[0], h_scale[1], h_scale[2], h_scale[3], h_scale[4]);
        free(h_scale);
    }

    // 2. Q, K, V projections (INT8 GEMM with dequant)
    // Weights are stored as [out_dim, in_dim], need transpose_b=true
    // Q: [hidden_size, hidden_size] -> transpose to get [hidden_size, hidden_size]
    result = cuda_gemm_int8(q, normed, w->q_proj, w->q_scale, num_tokens, hidden_size, hidden_size, true);
    checkNaN(q, hidden_size, "AFTER_Q_PROJ");
    if (result != 0) goto cleanup;

    // K: [kv_dim, hidden_size] -> transpose to get [hidden_size, kv_dim]
    result = cuda_gemm_int8(k, normed, w->k_proj, w->k_scale, num_tokens, hidden_size, kv_dim, true);
    checkNaN(k, kv_dim, "AFTER_K_PROJ");
    if (result != 0) goto cleanup;

    // V: [kv_dim, hidden_size] -> transpose to get [hidden_size, kv_dim]
    result = cuda_gemm_int8(v, normed, w->v_proj, w->v_scale, num_tokens, hidden_size, kv_dim, true);
    checkNaN(v, kv_dim, "AFTER_V_PROJ");
    if (result != 0) goto cleanup;

    // 3. Reshape for attention: [batch, seq, hidden] -> [batch, seq, num_heads, head_dim]
    //    and apply RoPE
    {
        // For RoPE, we need [batch, seq, num_heads, head_dim] layout
        // Q: [num_tokens, hidden] -> [batch, seq, num_heads, head_dim]
        // Already in correct layout for our purposes

        result = cuda_rope(q, q, positions, batch_size, seq_len, num_heads, head_dim);
        checkNaN(q, hidden_size, "AFTER_Q_ROPE");
        if (result != 0) goto cleanup;

        result = cuda_rope(k, k, positions, batch_size, seq_len, num_kv_heads, head_dim);
        checkNaN(k, kv_dim, "AFTER_K_ROPE");
        if (result != 0) goto cleanup;
    }

    // 4. Attention
    if (seq_len == 1 && kv_cache != nullptr) {
        // Single token with cache - use incremental attention with GQA support
        // Copy position from device to host (positions is now a device pointer)
        int position;
        CUDA_CHECK(cudaMemcpy(&position, positions, sizeof(int), cudaMemcpyDeviceToHost));
        result = cuda_attention_with_kvcache(
            attn_out, q, k, v, kv_cache,
            batch_size, num_heads, num_kv_heads, head_dim, position
        );
    } else {
        // Multiple tokens or no cache - use full attention
        result = cuda_basic_attention(
            attn_out, q, k, v,
            batch_size, num_heads, seq_len, head_dim,
            true  // causal
        );
    }
    checkNaN(attn_out, hidden_size, "AFTER_ATTENTION");
    if (result != 0) goto cleanup;

    // 5. Output projection
    // O: [hidden_size, hidden_size] -> transpose to get [hidden_size, hidden_size]
    result = cuda_gemm_int8(normed, attn_out, w->o_proj, w->o_scale,
                            num_tokens, hidden_size, hidden_size, true);
    checkNaN(normed, hidden_size, "AFTER_O_PROJ");
    if (result != 0) goto cleanup;

    // 6. Residual add
    result = cuda_add(normed, normed, residual, num_tokens * hidden_size);
    checkNaN(normed, hidden_size, "AFTER_RESIDUAL1");
    if (result != 0) goto cleanup;

    // Save new residual
    CUDA_CHECK(cudaMemcpy(residual, normed, num_tokens * hidden_size * sizeof(half),
                          cudaMemcpyDeviceToDevice));

    // 7. FFN RMSNorm
    result = cuda_rmsnorm(normed, normed, w->ffn_norm, num_tokens, hidden_size, rms_norm_eps);
    if (result != 0) goto cleanup;

    // 8. FFN: gate_proj and up_proj
    // Gate: [intermediate_size, hidden_size] -> transpose to get [hidden_size, intermediate_size]
    result = cuda_gemm_int8(gate, normed, w->gate_proj, w->gate_scale,
                            num_tokens, hidden_size, intermediate_size, true);
    if (result != 0) goto cleanup;

    // Up: [intermediate_size, hidden_size] -> transpose to get [hidden_size, intermediate_size]
    result = cuda_gemm_int8(up, normed, w->up_proj, w->up_scale,
                            num_tokens, hidden_size, intermediate_size, true);
    if (result != 0) goto cleanup;

    // 9. SiLU(gate) * up
    result = cuda_silu(gate, gate, num_tokens * intermediate_size);
    if (result != 0) goto cleanup;

    result = cuda_mul(gate, gate, up, num_tokens * intermediate_size);
    if (result != 0) goto cleanup;

    // 10. down_proj
    // Down: [hidden_size, intermediate_size] -> transpose to get [intermediate_size, hidden_size]
    result = cuda_gemm_int8(normed, gate, w->down_proj, w->down_scale,
                            num_tokens, intermediate_size, hidden_size, true);
    if (result != 0) goto cleanup;

    // 11. Final residual add
    result = cuda_add(output, normed, residual, num_tokens * hidden_size);

cleanup:
    cudaFree(normed);
    cudaFree(q);
    cudaFree(k);
    cudaFree(v);
    cudaFree(attn_out);
    cudaFree(residual);
    cudaFree(gate);
    cudaFree(up);

    return result;
}
