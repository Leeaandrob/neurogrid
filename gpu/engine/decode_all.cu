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
#include "../cuda/stream.h"

// Forward declarations from other translation units
extern "C" int cuda_conv_layer_forward_bf16(void*, const void*, const void*, void*, int, int, int);
extern "C" int cuda_conv_layer_forward_bf16_ws(void*, const void*, const void*, void*, int, int, int, void*);
extern "C" int cuda_layer_forward_fp16_with_workspace(
    void*, const void*, const void*, void*, const int*, int, int,
    int, int, int, int, int, float, float, int, void*);
extern "C" int cuda_layer_forward_fp16(
    void*, const void*, const void*, void*, const int*, int, int,
    int, int, int, int, int, float, float, int);
extern "C" int cuda_layer_forward_fp16_paged(
    void*, const void*, const void*, void*, const int*, const int*, const int*, int, int,
    int, int, int, int, int, float, float, int, void*);
extern "C" int cuda_fp16_to_bf16(void*, const void*, size_t);
extern "C" int cuda_bf16_to_fp16(void*, const void*, size_t);
extern "C" int cublas_set_stream(void* stream);
extern "C" int cuda_layer_forward_bf16_native(
    void*, const void*, const void*, void*, const int*, const int*, int, int,
    int, int, int, int, int, float, float, int, void*);
extern "C" int cuda_layer_forward_bf16_paged(
    void*, const void*, const void*, void*, const int*, const int*, const int*, int, int,
    int, int, int, int, int, float, float, int, void*);

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
    int current_position;  // Host-side position (avoids D2H copy during graph capture)

    // Per-layer info
    int* layer_types;           // 0=conv, 1=attention
    void** layer_weights;       // ConvLayerWeights* or LayerWeightsFP16*
    void** layer_caches;        // conv_state* or KVCache*

    // Shared workspace (attention layers)
    void* attn_workspace;       // LayerWorkspaceFP16*
    void* conv_workspace;       // ConvWorkspace* (for CUDA Graph safe conv forward)

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
    int* d_seq_lens;      // GPU buffer [1]: seq_len = position + 1 (CUDA Graph safe)

    // Paged KV Cache — per-layer (replaces per-layer KVCache* for attention layers)
    void** paged_caches;         // [num_layers] PagedKVCache* (NULL for conv layers)
    int* d_block_table;          // GPU block table buffer (shared, updated before each step)
    int max_blocks_per_seq;
    bool use_paged;

    // BF16-native attention support (eliminates FP16↔BF16 conversions)
    bool use_bf16_native;             // true if attention layers use BF16 native weights
    void* bf16_attn_workspace;        // LayerWorkspaceBF16*
    __nv_bfloat16* bf16_hidden_a;     // [hidden_size] BF16 ping buffer
    __nv_bfloat16* bf16_hidden_b;     // [hidden_size] BF16 pong buffer

    // CUDA Graph for decode step replay
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    cudaStream_t stream;
    bool graph_captured;
    int warmup_count;         // Number of warmup steps before capture (cuda_decode_step)
    int warmup_count_gpu;     // Separate counter for GPU-resident path (cuda_decode_step_gpu)
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

    // Allocate position and seq_lens buffers (single int each)
    CUDA_CHECK(cudaMalloc(&ctx->d_position, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&ctx->d_seq_lens, sizeof(int)));

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

extern "C" void cuda_set_decode_conv_workspace(void* ctx_ptr, void* workspace) {
    DecodeContext* ctx = (DecodeContext*)ctx_ptr;
    ctx->conv_workspace = workspace;
}

extern "C" void cuda_set_decode_paged_cache(void* ctx_ptr, void* paged_cache,
    int* d_block_table, int max_blocks_per_seq) {
    DecodeContext* ctx = (DecodeContext*)ctx_ptr;
    ctx->d_block_table = d_block_table;
    ctx->max_blocks_per_seq = max_blocks_per_seq;
    ctx->use_paged = true;
    // Note: per-layer caches set via cuda_set_decode_paged_layer
}

// Set per-layer paged cache (call once per attention layer during init)
extern "C" void cuda_set_decode_paged_layer(void* ctx_ptr, int layer_id, void* paged_cache) {
    DecodeContext* ctx = (DecodeContext*)ctx_ptr;
    if (!ctx->paged_caches) {
        ctx->paged_caches = (void**)calloc(ctx->num_layers, sizeof(void*));
    }
    ctx->paged_caches[layer_id] = paged_cache;
}

// Enable BF16-native mode: allocate BF16 hidden state buffers, set workspace
extern "C" int cuda_set_decode_bf16_native(void* ctx_ptr, void* bf16_workspace) {
    DecodeContext* ctx = (DecodeContext*)ctx_ptr;
    int H = ctx->hidden_size;

    // Allocate BF16 ping-pong hidden state buffers
    if (!ctx->bf16_hidden_a) {
        CUDA_CHECK(cudaMalloc(&ctx->bf16_hidden_a, H * sizeof(__nv_bfloat16)));
    }
    if (!ctx->bf16_hidden_b) {
        CUDA_CHECK(cudaMalloc(&ctx->bf16_hidden_b, H * sizeof(__nv_bfloat16)));
    }

    ctx->bf16_attn_workspace = bf16_workspace;
    ctx->use_bf16_native = true;
    return 0;
}

// Set BF16 hidden state from host (for first token after prefill)
extern "C" int cuda_decode_set_hidden_bf16(void* ctx_ptr, const void* h_hidden) {
    DecodeContext* ctx = (DecodeContext*)ctx_ptr;
    CUDA_CHECK(cudaMemcpy(ctx->bf16_hidden_a, h_hidden,
        ctx->hidden_size * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    return 0;
}

// Convert FP16 hidden_a → BF16 bf16_hidden_a (called after SetHidden when BF16 native is active)
extern "C" int cuda_decode_convert_fp16_to_bf16(void* ctx_ptr) {
    DecodeContext* ctx = (DecodeContext*)ctx_ptr;
    if (!ctx->bf16_hidden_a || !ctx->hidden_a) return -1;
    return cuda_fp16_to_bf16(ctx->bf16_hidden_a, ctx->hidden_a, ctx->hidden_size);
}

// Convert BF16 bf16_hidden_a → FP16 hidden_a (called after BF16 decode so LM head gets FP16)
extern "C" int cuda_decode_convert_bf16_to_fp16(void* ctx_ptr) {
    DecodeContext* ctx = (DecodeContext*)ctx_ptr;
    if (!ctx->bf16_hidden_a || !ctx->hidden_a) return -1;
    return cuda_bf16_to_fp16(ctx->hidden_a, ctx->bf16_hidden_a, ctx->hidden_size);
}

// Get BF16 hidden state to host
extern "C" int cuda_decode_get_hidden_bf16(void* ctx_ptr, void* h_hidden) {
    DecodeContext* ctx = (DecodeContext*)ctx_ptr;
    __nv_bfloat16* result_buf = (ctx->num_layers % 2 == 0) ? ctx->bf16_hidden_a : ctx->bf16_hidden_b;
    CUDA_CHECK(cudaMemcpy(h_hidden, result_buf,
        ctx->hidden_size * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    return 0;
}

// Set BF16 hidden state from GPU pointer
extern "C" int cuda_decode_set_hidden_bf16_from_gpu(void* ctx_ptr, const void* d_hidden) {
    DecodeContext* ctx = (DecodeContext*)ctx_ptr;
    CUDA_CHECK(cudaMemcpy(ctx->bf16_hidden_a, d_hidden,
        ctx->hidden_size * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice));
    return 0;
}

// Get BF16 hidden GPU pointer
extern "C" void* cuda_decode_get_hidden_bf16_gpu_ptr(void* ctx_ptr) {
    DecodeContext* ctx = (DecodeContext*)ctx_ptr;
    return ctx->bf16_hidden_a;
}

// ============================================================================
// Internal: run all layers (used for both normal and graph-captured paths)
// ============================================================================
// Forward declaration for paged attention (CUDA Graph safe — reads position/seq_len from GPU buffers)
extern "C" int cuda_paged_attention(void*, const void*, const void*, const void*, void*, const int*, const int*, const int*, int, int, int);

static int run_all_layers(DecodeContext* ctx, cudaStream_t stream) {
    int H = ctx->hidden_size;

    // BF16-native path: conv layers already BF16, attention layers now BF16 too
    // No FP16↔BF16 conversions needed at layer boundaries!
    if (ctx->use_bf16_native && ctx->bf16_hidden_a && ctx->bf16_attn_workspace) {
        __nv_bfloat16* current = ctx->bf16_hidden_a;
        __nv_bfloat16* next = ctx->bf16_hidden_b;

        static int bf16_trace = 0;
        if (bf16_trace < 3) {
            __nv_bfloat16 dbg[4];
            cudaMemcpy(dbg, current, 4*sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
            fprintf(stderr, "[BF16] input: [%.6f, %.6f, %.6f, %.6f]\n",
                __bfloat162float(dbg[0]), __bfloat162float(dbg[1]),
                __bfloat162float(dbg[2]), __bfloat162float(dbg[3]));
        }

        for (int i = 0; i < ctx->num_layers; i++) {
            int result;

            if (ctx->layer_types[i] == 0) {
                // Conv layer: already BF16 native — no conversion needed
                if (ctx->conv_workspace) {
                    result = cuda_conv_layer_forward_bf16_ws(
                        next, current,
                        ctx->layer_weights[i], ctx->layer_caches[i],
                        1, 1, 0, ctx->conv_workspace);
                } else {
                    result = cuda_conv_layer_forward_bf16(
                        next, current,
                        ctx->layer_weights[i], ctx->layer_caches[i],
                        1, 1, 0);
                }
                if (result != 0) return result;
            } else {
                // Attention layer: BF16 native forward
                if (ctx->use_paged && ctx->paged_caches && ctx->paged_caches[i]) {
                    result = cuda_layer_forward_bf16_paged(
                        next, current, ctx->layer_weights[i],
                        ctx->paged_caches[i], ctx->d_block_table,
                        ctx->d_position, ctx->d_seq_lens, 1, 1,
                        H, ctx->intermediate_size, ctx->num_heads,
                        ctx->num_kv_heads, ctx->head_dim,
                        ctx->norm_eps, ctx->rope_theta, ctx->rope_style,
                        ctx->bf16_attn_workspace);
                } else {
                    result = cuda_layer_forward_bf16_native(
                        next, current, ctx->layer_weights[i],
                        ctx->layer_caches[i],
                        ctx->d_position, ctx->d_seq_lens, 1, 1,
                        H, ctx->intermediate_size, ctx->num_heads,
                        ctx->num_kv_heads, ctx->head_dim,
                        ctx->norm_eps, ctx->rope_theta, ctx->rope_style,
                        ctx->bf16_attn_workspace);
                }
                if (result != 0) return result;
            }

            if (bf16_trace < 1) {
                __nv_bfloat16 dbg[4];
                cudaMemcpy(dbg, next, 4*sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
                fprintf(stderr, "[BF16] L%d(%s): [%.6f, %.6f, %.6f, %.6f] rc=%d\n",
                    i, ctx->layer_types[i]==0?"conv":"attn",
                    __bfloat162float(dbg[0]), __bfloat162float(dbg[1]),
                    __bfloat162float(dbg[2]), __bfloat162float(dbg[3]), result);
            }

            // Ping-pong (BF16)
            __nv_bfloat16* tmp = current;
            current = next;
            next = tmp;
        }
        if (bf16_trace < 3) bf16_trace++;
        return 0;
    }

    // FP16 path (original): conv layers need FP16↔BF16 conversion
    half* current = ctx->hidden_a;
    half* next = ctx->hidden_b;

    for (int i = 0; i < ctx->num_layers; i++) {
        int result;

        if (ctx->layer_types[i] == 0) {
            // Conv layer: FP16 → BF16 → conv forward → BF16 → FP16
            result = cuda_fp16_to_bf16(ctx->conv_bf16_in, current, H);
            if (result != 0) return result;

            if (ctx->conv_workspace) {
                // CUDA Graph safe: no cudaMalloc inside
                result = cuda_conv_layer_forward_bf16_ws(
                    ctx->conv_bf16_out, ctx->conv_bf16_in,
                    ctx->layer_weights[i], ctx->layer_caches[i],
                    1, 1, 0, ctx->conv_workspace);
            } else {
                result = cuda_conv_layer_forward_bf16(
                    ctx->conv_bf16_out, ctx->conv_bf16_in,
                    ctx->layer_weights[i], ctx->layer_caches[i],
                    1, 1, 0);
            }
            if (result != 0) return result;

            result = cuda_bf16_to_fp16(next, ctx->conv_bf16_out, H);
            if (result != 0) return result;
        } else {
            // Attention layer: FP16 forward
            if (ctx->use_paged && ctx->paged_caches && ctx->paged_caches[i] && ctx->attn_workspace) {
                // Paged attention path: use workspace forward but replace attention
                // with paged attention call using host position (no D2H copy - CUDA Graph safe)
                result = cuda_layer_forward_fp16_paged(
                    next, current, ctx->layer_weights[i],
                    ctx->paged_caches[i], ctx->d_block_table,
                    ctx->d_position, ctx->d_seq_lens, 1, 1,
                    H, ctx->intermediate_size, ctx->num_heads,
                    ctx->num_kv_heads, ctx->head_dim,
                    ctx->norm_eps, ctx->rope_theta, ctx->rope_style,
                    ctx->attn_workspace);
            } else if (ctx->attn_workspace) {
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

    // Copy input, position, and seq_len to persistent GPU buffers (OUTSIDE graph)
    CUDA_CHECK(cudaMemcpy(ctx->hidden_a, h_input, hs, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->d_position, &position, sizeof(int), cudaMemcpyHostToDevice));
    int seq_len_val = position + 1;
    CUDA_CHECK(cudaMemcpy(ctx->d_seq_lens, &seq_len_val, sizeof(int), cudaMemcpyHostToDevice));
    ctx->current_position = position;  // Host-side position (used inside graph to avoid D2H copy)

    // BF16 native: convert FP16 hidden_a → BF16 bf16_hidden_a
    if (ctx->use_bf16_native && ctx->bf16_hidden_a) {
        int cvt = cuda_fp16_to_bf16(ctx->bf16_hidden_a, ctx->hidden_a, H);
        if (cvt != 0) return cvt;
    }

    // Determine output buffer
    half* result_buf = (ctx->num_layers % 2 == 0) ? ctx->hidden_a : ctx->hidden_b;

    // cuda_decode_step: host path (used during prefill)
    // Graph capture is ONLY in cuda_decode_step_gpu (decode path)
    // to avoid leaving stream in capture mode during prefill.
    // Host path (cuda_decode_step): NO graph capture — just run layers normally.
    // Graph capture is only in cuda_decode_step_gpu (decode path).
    // This avoids leaving the stream in capture mode during prefill.
    if (ctx->graph_captured && ctx->graph_exec) {
        cudaError_t err = cudaGraphLaunch(ctx->graph_exec, (cudaStream_t)0);
        if (err != cudaSuccess) {
            ctx->graph_captured = false;
            run_all_layers(ctx, nullptr);
        } else {
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    } else {
        int res = run_all_layers(ctx, nullptr);
        if (res != 0) return res;
    }

    // BF16→FP16 post-conversion: result_buf is FP16, but BF16 path wrote to bf16_hidden_a
    if (ctx->use_bf16_native && ctx->bf16_hidden_a && ctx->hidden_a) {
        // After BF16 run_all_layers, result is in bf16_hidden_a (even layers → no swap)
        int cvt = cuda_bf16_to_fp16(ctx->hidden_a, ctx->bf16_hidden_a, H);
        if (cvt != 0) return cvt;
        // result_buf points to hidden_a (for even num_layers) or hidden_b
        // We converted into hidden_a, which is result_buf for even num_layers
        if (ctx->num_layers % 2 != 0) {
            // Odd layers: result is in hidden_b, but BF16 result was converted to hidden_a
            // Need to copy to hidden_b OR adjust result_buf
            result_buf = ctx->hidden_a;
        }
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

    // Debug: check hidden_a content on first few calls
    static int gpu_dbg = 0;
    if (gpu_dbg < 3) {
        half h_fp16[4];
        cudaMemcpy(h_fp16, ctx->hidden_a, 4*sizeof(half), cudaMemcpyDeviceToHost);
        fprintf(stderr, "[GPU-DBG] step_gpu pos=%d hidden_a=[%.6f, %.6f, %.6f, %.6f] bf16=%s warmup=%d\n",
            position, __half2float(h_fp16[0]), __half2float(h_fp16[1]),
            __half2float(h_fp16[2]), __half2float(h_fp16[3]),
            ctx->use_bf16_native ? "yes" : "no", ctx->warmup_count_gpu);
        if (ctx->use_bf16_native && ctx->bf16_hidden_a) {
            __nv_bfloat16 h_bf16[4];
            cudaMemcpy(h_bf16, ctx->bf16_hidden_a, 4*sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
            fprintf(stderr, "[GPU-DBG] bf16_hidden_a=[%.6f, %.6f, %.6f, %.6f]\n",
                __bfloat162float(h_bf16[0]), __bfloat162float(h_bf16[1]),
                __bfloat162float(h_bf16[2]), __bfloat162float(h_bf16[3]));
        }
        gpu_dbg++;
    }

    // Update position and seq_len GPU buffers (OUTSIDE graph — H2D copies)
    CUDA_CHECK(cudaMemcpy(ctx->d_position, &position, sizeof(int), cudaMemcpyHostToDevice));
    int seq_len_val = position + 1;
    CUDA_CHECK(cudaMemcpy(ctx->d_seq_lens, &seq_len_val, sizeof(int), cudaMemcpyHostToDevice));

    if (ctx->graph_captured && ctx->graph_exec) {
        // REPLAY captured graph
        cudaError_t err = cudaGraphLaunch(ctx->graph_exec, (cudaStream_t)0);
        if (err != cudaSuccess) {
            fprintf(stderr, "[NeuroGrid] GPU graph replay failed: %s\n", cudaGetErrorString(err));
            ctx->graph_captured = false;
            int res = run_all_layers(ctx, nullptr);
            if (res != 0) return res;
        } else {
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    } else if (ctx->warmup_count_gpu == 2 && !ctx->graph_captured) {
        // CAPTURE on step 2
        fprintf(stderr, "[NeuroGrid] Capturing CUDA graph on dedicated stream...\n");
        ctx->warmup_count_gpu = 3;
        cudaGetLastError();

        // Route ALL kernels to the capture stream
        ng_set_stream(ctx->stream);
        cublas_set_stream(ctx->stream);

        cudaError_t err = cudaStreamBeginCapture(ctx->stream, cudaStreamCaptureModeGlobal);
        if (err != cudaSuccess) {
            fprintf(stderr, "[NeuroGrid] Graph capture failed: %s (running without graphs)\n", cudaGetErrorString(err));
            ng_set_stream(nullptr);
            cublas_set_stream(nullptr);
            int res = run_all_layers(ctx, nullptr);
            if (res != 0) return res;
        } else {
            run_all_layers(ctx, nullptr);
            cudaGraph_t graph = nullptr;
            err = cudaStreamEndCapture(ctx->stream, &graph);
            ng_set_stream(nullptr);
            cublas_set_stream(nullptr);
            if (err == cudaSuccess && graph != nullptr) {
                cudaGraphExec_t exec = nullptr;
                err = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
                if (err == cudaSuccess && exec != nullptr) {
                    ctx->graph = graph;
                    ctx->graph_exec = exec;
                    ctx->graph_captured = true;
                    size_t numNodes = 0;
                    cudaGraphGetNodes(graph, nullptr, &numNodes);
                    fprintf(stderr, "[NeuroGrid] CUDA graph captured! %zu nodes\n", numNodes);
                    // Replay to get this step's result
                    cudaGraphLaunch(exec, (cudaStream_t)0);
                    CUDA_CHECK(cudaDeviceSynchronize());
                } else {
                    fprintf(stderr, "[NeuroGrid] Graph instantiate failed: %s\n",
                            err != cudaSuccess ? cudaGetErrorString(err) : "null");
                    if (graph) cudaGraphDestroy(graph);
                    run_all_layers(ctx, nullptr);
                }
            } else {
                fprintf(stderr, "[NeuroGrid] Graph end capture failed: %s\n",
                        err != cudaSuccess ? cudaGetErrorString(err) : "null graph");
                run_all_layers(ctx, nullptr);
            }
        }
    } else {
        // Warmup or post-failed-capture: run normally
        int res = run_all_layers(ctx, nullptr);
        if (res != 0) return res;
        ctx->warmup_count_gpu++;
    }

    // Swap buffers for next call
    if (ctx->num_layers % 2 != 0) {
        if (ctx->use_bf16_native && ctx->bf16_hidden_a) {
            __nv_bfloat16* tmp = ctx->bf16_hidden_a;
            ctx->bf16_hidden_a = ctx->bf16_hidden_b;
            ctx->bf16_hidden_b = tmp;
        } else {
            half* tmp = ctx->hidden_a;
            ctx->hidden_a = ctx->hidden_b;
            ctx->hidden_b = tmp;
        }
    }

    // BF16→FP16 post-conversion: LM head reads from hidden_a (FP16)
    if (ctx->use_bf16_native && ctx->bf16_hidden_a && ctx->hidden_a) {
        int cvt = cuda_bf16_to_fp16(ctx->hidden_a, ctx->bf16_hidden_a, ctx->hidden_size);
        if (cvt != 0) return cvt;
    }

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
    if (ctx->bf16_hidden_a) cudaFree(ctx->bf16_hidden_a);
    if (ctx->bf16_hidden_b) cudaFree(ctx->bf16_hidden_b);
    if (ctx->d_position) cudaFree(ctx->d_position);
    if (ctx->d_seq_lens) cudaFree(ctx->d_seq_lens);
    if (ctx->layer_types) free(ctx->layer_types);
    if (ctx->layer_weights) free(ctx->layer_weights);
    if (ctx->layer_caches) free(ctx->layer_caches);
    if (ctx->paged_caches) free(ctx->paged_caches);
    free(ctx);
}
