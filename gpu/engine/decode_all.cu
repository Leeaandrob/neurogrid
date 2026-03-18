// decode_all.cu — Full decode step: all layers in a single CUDA call
// Eliminates Go↔CUDA round-trips between layers.
// For LFM2: runs 10 conv layers + 6 attention layers sequentially on GPU.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "decode_all.h"

// Forward declarations from other translation units
extern "C" int cuda_conv_layer_forward_bf16(void*, const void*, const void*, void*, int, int, int);
extern "C" int cuda_layer_forward_fp16_with_workspace(
    void*, const void*, const void*, void*, const int*, int, int,
    int, int, int, int, int, float, float, int, void*);
extern "C" int cuda_layer_forward_fp16(
    void*, const void*, const void*, void*, const int*, int, int,
    int, int, int, int, int, float, float, int);
extern "C" int cuda_fp16_to_bf16(void*, const void*, size_t);
extern "C" int cuda_bf16_to_fp16(void*, const void*, size_t);

// Error checking
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        return -1; \
    } \
} while(0)

// ============================================================================
// Full Decode Context — holds all layer weights and workspace buffers
// ============================================================================

struct DecodeContext {
    int num_layers;
    int hidden_size;
    int intermediate_size;
    int num_heads;
    int num_kv_heads;
    int head_dim;
    float norm_eps;
    float rope_theta;
    int rope_style;
    int conv_kernel_size;

    // Per-layer info
    int* layer_types;           // 0=conv, 1=attention
    void** layer_weights;       // ConvLayerWeights* or LayerWeightsFP16*
    void** layer_caches;        // conv_state* or KVCache*

    // Shared workspace (attention layers)
    void* attn_workspace;       // LayerWorkspaceFP16*

    // Shared buffers for FP16↔BF16 conversion (conv layers)
    half* conv_fp16_buf;        // [hidden_size] FP16
    __nv_bfloat16* conv_bf16_in;  // [hidden_size] BF16
    __nv_bfloat16* conv_bf16_out; // [hidden_size] BF16
    half* conv_fp16_out;        // [hidden_size] FP16

    // Temp buffers for decode
    half* hidden_a;             // [hidden_size] ping buffer
    half* hidden_b;             // [hidden_size] pong buffer

    // GPU position buffer (avoids per-step allocation)
    int* d_position;
};

extern "C" int cuda_create_decode_context(
    void** ctx_out,
    int num_layers,
    int hidden_size,
    int intermediate_size,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float norm_eps,
    float rope_theta,
    int rope_style,
    int conv_kernel_size
) {
    DecodeContext* ctx = (DecodeContext*)calloc(1, sizeof(DecodeContext));
    if (!ctx) return -1;

    ctx->num_layers = num_layers;
    ctx->hidden_size = hidden_size;
    ctx->intermediate_size = intermediate_size;
    ctx->num_heads = num_heads;
    ctx->num_kv_heads = num_kv_heads;
    ctx->head_dim = head_dim;
    ctx->norm_eps = norm_eps;
    ctx->rope_theta = rope_theta;
    ctx->rope_style = rope_style;
    ctx->conv_kernel_size = conv_kernel_size;

    ctx->layer_types = (int*)calloc(num_layers, sizeof(int));
    ctx->layer_weights = (void**)calloc(num_layers, sizeof(void*));
    ctx->layer_caches = (void**)calloc(num_layers, sizeof(void*));

    size_t hs = hidden_size * sizeof(half);

    // Allocate conversion buffers for conv layers
    CUDA_CHECK(cudaMalloc(&ctx->conv_bf16_in, hidden_size * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&ctx->conv_bf16_out, hidden_size * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&ctx->conv_fp16_out, hs));

    // Allocate ping-pong hidden state buffers
    CUDA_CHECK(cudaMalloc(&ctx->hidden_a, hs));
    CUDA_CHECK(cudaMalloc(&ctx->hidden_b, hs));

    // Allocate position buffer (single int)
    CUDA_CHECK(cudaMalloc(&ctx->d_position, sizeof(int)));

    *ctx_out = ctx;
    return 0;
}

extern "C" void cuda_set_decode_layer(
    void* ctx_ptr,
    int layer_id,
    int layer_type,     // 0=conv, 1=attention
    void* weights,
    void* cache
) {
    DecodeContext* ctx = (DecodeContext*)ctx_ptr;
    ctx->layer_types[layer_id] = layer_type;
    ctx->layer_weights[layer_id] = weights;
    ctx->layer_caches[layer_id] = cache;
}

extern "C" void cuda_set_decode_workspace(void* ctx_ptr, void* workspace) {
    DecodeContext* ctx = (DecodeContext*)ctx_ptr;
    ctx->attn_workspace = workspace;
}

// ============================================================================
// Full decode step — all layers in a single call
// ============================================================================
// Input:  FP16 hidden state [hidden_size] on HOST
// Output: FP16 hidden state [hidden_size] on HOST
// Runs all num_layers layers sequentially without returning to Go.

extern "C" int cuda_decode_step(
    void* ctx_ptr,
    void* h_output,         // HOST: output hidden [hidden_size] FP16
    const void* h_input,    // HOST: input hidden [hidden_size] FP16
    int position
) {
    DecodeContext* ctx = (DecodeContext*)ctx_ptr;
    int H = ctx->hidden_size;
    size_t hs = H * sizeof(half);

    // Copy input to GPU
    CUDA_CHECK(cudaMemcpy(ctx->hidden_a, h_input, hs, cudaMemcpyHostToDevice));

    // Copy position to GPU
    CUDA_CHECK(cudaMemcpy(ctx->d_position, &position, sizeof(int), cudaMemcpyHostToDevice));

    half* current = ctx->hidden_a;
    half* next = ctx->hidden_b;

    for (int i = 0; i < ctx->num_layers; i++) {
        int result;

        if (ctx->layer_types[i] == 0) {
            // Conv layer: FP16 → BF16 → conv forward → BF16 → FP16
            result = cuda_fp16_to_bf16(ctx->conv_bf16_in, current, H);
            if (result != 0) return result;

            result = cuda_conv_layer_forward_bf16(
                ctx->conv_bf16_out, ctx->conv_bf16_in,
                ctx->layer_weights[i], ctx->layer_caches[i],
                1, 1, position);
            if (result != 0) return result;

            result = cuda_bf16_to_fp16(next, ctx->conv_bf16_out, H);
            if (result != 0) return result;
        } else {
            // Attention layer: FP16 forward
            if (ctx->attn_workspace) {
                result = cuda_layer_forward_fp16_with_workspace(
                    next, current, ctx->layer_weights[i], ctx->layer_caches[i],
                    ctx->d_position, 1, 1,
                    H, ctx->intermediate_size, ctx->num_heads,
                    ctx->num_kv_heads, ctx->head_dim,
                    ctx->norm_eps, ctx->rope_theta, ctx->rope_style,
                    ctx->attn_workspace);
            } else {
                result = cuda_layer_forward_fp16(
                    next, current, ctx->layer_weights[i], ctx->layer_caches[i],
                    ctx->d_position, 1, 1,
                    H, ctx->intermediate_size, ctx->num_heads,
                    ctx->num_kv_heads, ctx->head_dim,
                    ctx->norm_eps, ctx->rope_theta, ctx->rope_style);
            }
            if (result != 0) return result;
        }

        // Ping-pong: next becomes current
        half* tmp = current;
        current = next;
        next = tmp;
    }

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_output, current, hs, cudaMemcpyDeviceToHost));

    return 0;
}

extern "C" void cuda_free_decode_context(void* ctx_ptr) {
    if (!ctx_ptr) return;
    DecodeContext* ctx = (DecodeContext*)ctx_ptr;

    if (ctx->conv_bf16_in) cudaFree(ctx->conv_bf16_in);
    if (ctx->conv_bf16_out) cudaFree(ctx->conv_bf16_out);
    if (ctx->conv_fp16_out) cudaFree(ctx->conv_fp16_out);
    if (ctx->hidden_a) cudaFree(ctx->hidden_a);
    if (ctx->hidden_b) cudaFree(ctx->hidden_b);
    if (ctx->d_position) cudaFree(ctx->d_position);
    if (ctx->layer_types) free(ctx->layer_types);
    if (ctx->layer_weights) free(ctx->layer_weights);
    if (ctx->layer_caches) free(ctx->layer_caches);
    free(ctx);
}
