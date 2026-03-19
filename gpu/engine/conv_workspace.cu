// conv_workspace.cu — Conv layer forward with pre-allocated workspace
// Eliminates cudaMalloc/cudaFree during forward pass (required for CUDA Graph capture)

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <stdlib.h>

#include "../cuda/stream.h"
#include "conv_workspace.h"
#include "conv_layer.h"
#include "../cuda/bf16_utils.h"
#include "../cuda/conv.h"
#include "../cuda/matmul.h"

#include "../cuda/stream.h"
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        return -1; \
    } \
} while(0)

// Forward declarations
extern "C" int cuda_gemm_bf16(void*, const void*, const void*, int, int, int, bool, bool);
extern "C" int cuda_bf16_rmsnorm(void*, const void*, const void*, int, int, float);
extern "C" int cuda_bf16_silu(void*, const void*, size_t);
extern "C" int cuda_bf16_add(void*, const void*, const void*, size_t);
extern "C" int cuda_bf16_mul(void*, const void*, const void*, size_t);
extern "C" int cuda_causal_conv1d_update_bf16(const void*, void*, void*, const void*, int, int, int);

// Pre-allocated workspace for single-token conv layer forward (decode mode)
struct ConvWorkspace {
    __nv_bfloat16* normed;      // [H]
    __nv_bfloat16* projected;   // [3*H]
    __nv_bfloat16* Bx;          // [H]
    __nv_bfloat16* conv_out;    // [H]
    __nv_bfloat16* y;           // [H]
    __nv_bfloat16* block_out;   // [H]
    __nv_bfloat16* residual;    // [H]
    __nv_bfloat16* ffn_normed;  // [H]
    __nv_bfloat16* gate_out;    // [I]
    __nv_bfloat16* up_out;      // [I]
    __nv_bfloat16* ffn_out;     // [H]
    int hidden_size;
    int intermediate_size;
};

extern "C" int cuda_create_conv_workspace(void** ws_out, int hidden_size, int intermediate_size) {
    ConvWorkspace* ws = (ConvWorkspace*)calloc(1, sizeof(ConvWorkspace));
    if (!ws) return -1;

    ws->hidden_size = hidden_size;
    ws->intermediate_size = intermediate_size;
    size_t bf16 = sizeof(__nv_bfloat16);
    int H = hidden_size;
    int I = intermediate_size;

    CUDA_CHECK(cudaMalloc(&ws->normed, H * bf16));
    CUDA_CHECK(cudaMalloc(&ws->projected, 3 * H * bf16));
    CUDA_CHECK(cudaMalloc(&ws->Bx, H * bf16));
    CUDA_CHECK(cudaMalloc(&ws->conv_out, H * bf16));
    CUDA_CHECK(cudaMalloc(&ws->y, H * bf16));
    CUDA_CHECK(cudaMalloc(&ws->block_out, H * bf16));
    CUDA_CHECK(cudaMalloc(&ws->residual, H * bf16));
    CUDA_CHECK(cudaMalloc(&ws->ffn_normed, H * bf16));
    CUDA_CHECK(cudaMalloc(&ws->gate_out, I * bf16));
    CUDA_CHECK(cudaMalloc(&ws->up_out, I * bf16));
    CUDA_CHECK(cudaMalloc(&ws->ffn_out, H * bf16));

    *ws_out = ws;
    return 0;
}

extern "C" void cuda_free_conv_workspace(void* ws_ptr) {
    if (!ws_ptr) return;
    ConvWorkspace* ws = (ConvWorkspace*)ws_ptr;
    if (ws->normed) cudaFree(ws->normed);
    if (ws->projected) cudaFree(ws->projected);
    if (ws->Bx) cudaFree(ws->Bx);
    if (ws->conv_out) cudaFree(ws->conv_out);
    if (ws->y) cudaFree(ws->y);
    if (ws->block_out) cudaFree(ws->block_out);
    if (ws->residual) cudaFree(ws->residual);
    if (ws->ffn_normed) cudaFree(ws->ffn_normed);
    if (ws->gate_out) cudaFree(ws->gate_out);
    if (ws->up_out) cudaFree(ws->up_out);
    if (ws->ffn_out) cudaFree(ws->ffn_out);
    free(ws);
}

// Conv layer forward for decode (seq_len=1) using pre-allocated workspace.
// No cudaMalloc/cudaFree — safe for CUDA Graph capture.
extern "C" int cuda_conv_layer_forward_bf16_ws(
    void* output, const void* input, const void* weights,
    void* conv_state, int batch, int seq_len, int position, void* workspace
) {
    ConvLayerWeights* w = (ConvLayerWeights*)weights;
    ConvWorkspace* ws = (ConvWorkspace*)workspace;
    int H = w->hidden_size;
    int I = w->intermediate_size;
    int K = w->conv_kernel_size;
    int BS = batch * seq_len;
    int result;

    // For seq_len > 1, fall back to the original (with cudaMalloc)
    if (seq_len != 1) {
        return cuda_conv_layer_forward_bf16(output, input, weights, conv_state, batch, seq_len, position);
    }

    // 1. RMSNorm
    result = cuda_bf16_rmsnorm(ws->normed, input, w->operator_norm, BS, H, w->norm_eps);
    if (result != 0) return result;

    // 2. in_proj: [BS, H] → [BS, 3H]
    result = cuda_gemm_bf16(ws->projected, ws->normed, w->in_proj_weight, BS, H, 3*H, false, true);
    if (result != 0) return result;

    // 3. For seq_len=1: projected is [1, 3H]. Chunk into B, C, x each [H]
    // B = projected[0:H], C = projected[H:2H], x = projected[2H:3H]
    __nv_bfloat16* B_ptr = ws->projected;
    __nv_bfloat16* C_ptr = ws->projected + H;
    __nv_bfloat16* x_ptr = ws->projected + 2*H;

    // 4. Bx = B * x
    result = cuda_bf16_mul(ws->Bx, B_ptr, x_ptr, H);
    if (result != 0) return result;

    // 5. Conv1d update (single token)
    result = cuda_causal_conv1d_update_bf16(ws->Bx, ws->conv_out, conv_state, w->conv_weight, batch, H, K);
    if (result != 0) return result;

    // 6. y = C * conv_out
    result = cuda_bf16_mul(ws->y, C_ptr, ws->conv_out, H);
    if (result != 0) return result;

    // 7. out_proj: [BS, H] → [BS, H]
    result = cuda_gemm_bf16(ws->block_out, ws->y, w->out_proj_weight, BS, H, H, false, true);
    if (result != 0) return result;

    // 8. Residual
    result = cuda_bf16_add(ws->residual, input, ws->block_out, BS * H);
    if (result != 0) return result;

    // 9. FFN
    result = cuda_bf16_rmsnorm(ws->ffn_normed, ws->residual, w->ffn_norm, BS, H, w->norm_eps);
    if (result != 0) return result;

    result = cuda_gemm_bf16(ws->gate_out, ws->ffn_normed, w->gate_weight, BS, H, I, false, true);
    if (result != 0) return result;
    result = cuda_gemm_bf16(ws->up_out, ws->ffn_normed, w->up_weight, BS, H, I, false, true);
    if (result != 0) return result;

    result = cuda_bf16_silu(ws->gate_out, ws->gate_out, (size_t)BS * I);
    if (result != 0) return result;
    result = cuda_bf16_mul(ws->gate_out, ws->gate_out, ws->up_out, (size_t)BS * I);
    if (result != 0) return result;

    result = cuda_gemm_bf16(ws->ffn_out, ws->gate_out, w->down_weight, BS, I, H, false, true);
    if (result != 0) return result;

    // 10. Final residual
    result = cuda_bf16_add(output, ws->residual, ws->ffn_out, BS * H);
    return result;
}
