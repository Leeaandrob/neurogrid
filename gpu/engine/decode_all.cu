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
extern "C" int cublas_set_stream(void* stream);

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

    // CUDA Graph for decode step replay
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    cudaStream_t stream;
    bool graph_captured;
    int warmup_count;         // Number of warmup steps before capture
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

    // Create CUDA stream for graph capture
    CUDA_CHECK(cudaStreamCreate(&ctx->stream));
    ctx->graph = nullptr;
    ctx->graph_exec = nullptr;
    ctx->graph_captured = false;
    ctx->warmup_count = 0;

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
// Internal: run all layers (used for both normal and graph-captured paths)
// ============================================================================
static int run_all_layers(DecodeContext* ctx, cudaStream_t stream) {
    int H = ctx->hidden_size;
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
                1, 1, 0);  // position not used by conv decode kernel (uses state)
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

        // Ping-pong
        half* tmp = current;
        current = next;
        next = tmp;
    }

    // If num_layers is odd, result is in hidden_b; if even, in hidden_a.
    // Copy to hidden_a if needed (so output is always in hidden_a after even layers,
    // hidden_b after odd layers — we track this in the caller)
    return 0;
}

// ============================================================================
// Full decode step with CUDA Graph support
// ============================================================================
// First 2 calls: warmup (run normally)
// 3rd call: capture CUDA graph
// Subsequent calls: replay graph (much faster)

extern "C" int cuda_decode_step(
    void* ctx_ptr,
    void* h_output,
    const void* h_input,
    int position
) {
    DecodeContext* ctx = (DecodeContext*)ctx_ptr;
    int H = ctx->hidden_size;
    size_t hs = H * sizeof(half);

    // Always: copy input and position to GPU (outside graph)
    CUDA_CHECK(cudaMemcpy(ctx->hidden_a, h_input, hs, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->d_position, &position, sizeof(int), cudaMemcpyHostToDevice));

    // Determine output buffer: if num_layers is odd, result is in hidden_b; if even, hidden_a
    half* result_buf = (ctx->num_layers % 2 == 0) ? ctx->hidden_a : ctx->hidden_b;

    if (ctx->graph_captured && ctx->graph_exec) {
        // Fast path: replay captured CUDA graph
        cudaError_t err = cudaGraphLaunch(ctx->graph_exec, ctx->stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA graph launch failed: %s, falling back\n", cudaGetErrorString(err));
            ctx->graph_captured = false;
            int res = run_all_layers(ctx, nullptr);
            if (res != 0) return res;
        } else {
            CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
        }
    } else if (ctx->warmup_count >= 2 && !ctx->graph_captured) {
        // Capture CUDA graph on the dedicated stream
        // Must redirect cuBLAS to this stream for capture to work
        fprintf(stderr, "[NeuroGrid] Capturing CUDA graph for decode step...\n");

        // Set cuBLAS to use our capture stream
        cublas_set_stream(ctx->stream);

        cudaError_t err = cudaStreamBeginCapture(ctx->stream, cudaStreamCaptureModeGlobal);
        if (err != cudaSuccess) {
            fprintf(stderr, "[NeuroGrid] Graph capture failed to start: %s, running normally\n",
                    cudaGetErrorString(err));
            cublas_set_stream(nullptr);  // Reset to default stream
            int res = run_all_layers(ctx, nullptr);
            if (res != 0) return res;
        } else {
            int res = run_all_layers(ctx, ctx->stream);

            cudaGraph_t graph = nullptr;
            err = cudaStreamEndCapture(ctx->stream, &graph);

            // Reset cuBLAS to default stream
            cublas_set_stream(nullptr);

            if (err != cudaSuccess || graph == nullptr || res != 0) {
                fprintf(stderr, "[NeuroGrid] Graph capture failed: %s\n",
                        err != cudaSuccess ? cudaGetErrorString(err) : "run_all_layers failed");
                if (graph) cudaGraphDestroy(graph);
                // Run normally since capture may have consumed the kernels
                res = run_all_layers(ctx, nullptr);
                if (res != 0) return res;
            } else {
                cudaGraphExec_t exec = nullptr;
                err = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
                if (err != cudaSuccess) {
                    fprintf(stderr, "[NeuroGrid] Graph instantiate failed: %s\n", cudaGetErrorString(err));
                    cudaGraphDestroy(graph);
                } else {
                    ctx->graph = graph;
                    ctx->graph_exec = exec;
                    ctx->graph_captured = true;
                    fprintf(stderr, "[NeuroGrid] CUDA graph captured! (%d layers, replaying for future tokens)\n",
                            ctx->num_layers);
                }
            }
        }
        ctx->warmup_count++;
    } else {
        // Warmup: run normally
        int res = run_all_layers(ctx, nullptr);
        if (res != 0) return res;
        ctx->warmup_count++;
    }

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_output, result_buf, hs, cudaMemcpyDeviceToHost));

    return 0;
}

// ============================================================================
// GPU-Resident decode: hidden stays on GPU, only logits returned to host
// ============================================================================
// Eliminates 4 cudaMemcpy per token → only 1 small copy (position int)
// + 1 output copy (logits float32 array).
// The hidden state from the previous step stays in hidden_a/hidden_b.

// Set hidden state from another GPU buffer (GPU→GPU, zero-copy)
extern "C" int cuda_decode_set_hidden_from_gpu(void* ctx_ptr, const void* d_hidden) {
    DecodeContext* ctx = (DecodeContext*)ctx_ptr;
    CUDA_CHECK(cudaMemcpy(ctx->hidden_a, d_hidden,
        ctx->hidden_size * sizeof(half), cudaMemcpyDeviceToDevice));
    return 0;
}

// Set initial hidden state from host (called once for first token after prefill)
extern "C" int cuda_decode_set_hidden(void* ctx_ptr, const void* h_hidden) {
    DecodeContext* ctx = (DecodeContext*)ctx_ptr;
    CUDA_CHECK(cudaMemcpy(ctx->hidden_a, h_hidden,
        ctx->hidden_size * sizeof(half), cudaMemcpyHostToDevice));
    return 0;
}

// Get hidden state to host (for distributed mode: transfer to next peer)
extern "C" int cuda_decode_get_hidden(void* ctx_ptr, void* h_hidden) {
    DecodeContext* ctx = (DecodeContext*)ctx_ptr;
    half* result_buf = (ctx->num_layers % 2 == 0) ? ctx->hidden_a : ctx->hidden_b;
    CUDA_CHECK(cudaMemcpy(h_hidden, result_buf,
        ctx->hidden_size * sizeof(half), cudaMemcpyDeviceToHost));
    return 0;
}

// Run all layers, hidden stays on GPU. Only position is copied from host.
// After this call, result is in hidden_a or hidden_b (depending on num_layers parity).
extern "C" int cuda_decode_step_gpu(void* ctx_ptr, int position) {
    DecodeContext* ctx = (DecodeContext*)ctx_ptr;

    // Only copy position (4 bytes) — hidden already on GPU from previous step
    CUDA_CHECK(cudaMemcpy(ctx->d_position, &position, sizeof(int), cudaMemcpyHostToDevice));

    int res = run_all_layers(ctx, nullptr);
    if (res != 0) return res;

    // Swap buffers: result is in the "next" buffer after run_all_layers
    // For next call, we need input in hidden_a. If num_layers is odd, result is in hidden_b.
    if (ctx->num_layers % 2 != 0) {
        // Swap a↔b so result is always in hidden_a for next call
        half* tmp = ctx->hidden_a;
        ctx->hidden_a = ctx->hidden_b;
        ctx->hidden_b = tmp;
    }
    // Now result is always in hidden_a
    return 0;
}

// Get the GPU pointer to current hidden state (for LM head application on GPU)
extern "C" void* cuda_decode_get_hidden_gpu_ptr(void* ctx_ptr) {
    DecodeContext* ctx = (DecodeContext*)ctx_ptr;
    return ctx->hidden_a;  // Always contains latest result after cuda_decode_step_gpu
}

extern "C" void cuda_free_decode_context(void* ctx_ptr) {
    if (!ctx_ptr) return;
    DecodeContext* ctx = (DecodeContext*)ctx_ptr;

    if (ctx->graph_exec) cudaGraphExecDestroy(ctx->graph_exec);
    if (ctx->graph) cudaGraphDestroy(ctx->graph);
    if (ctx->stream) cudaStreamDestroy(ctx->stream);
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
