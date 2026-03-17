// conv_layer.cu - LIV Conv layer forward pass for LFM2
// Implements the complete conv block:
// RMSNorm -> in_proj(2048->6144) -> chunk(B,C,x) -> Bx=B*x -> conv1d(Bx)
// -> y=C*conv_out -> out_proj -> residual -> RMSNorm -> SwiGLU FFN -> residual

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <stdlib.h>

#include "conv_layer.h"
#include "../cuda/bf16_utils.h"
#include "../cuda/conv.h"
#include "../cuda/matmul.h"

// Error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        goto cleanup; \
    } \
} while(0)

#define KERNEL_CHECK(call) do { \
    int ret = call; \
    if (ret != 0) { \
        fprintf(stderr, "Kernel error at %s:%d: %d\n", __FILE__, __LINE__, ret); \
        goto cleanup; \
    } \
} while(0)

// Forward declarations for BF16 GEMM and kernels
extern "C" int cuda_gemm_bf16(void*, const void*, const void*, int, int, int, bool, bool);
extern "C" int cuda_bf16_rmsnorm(void*, const void*, const void*, int, int, float);
extern "C" int cuda_bf16_silu(void*, const void*, size_t);
extern "C" int cuda_bf16_add(void*, const void*, const void*, size_t);
extern "C" int cuda_bf16_mul(void*, const void*, const void*, size_t);
extern "C" int cuda_causal_conv1d_fwd_bf16(const void*, const void*, void*, void*, int, int, int, int);
extern "C" int cuda_causal_conv1d_update_bf16(const void*, void*, void*, const void*, int, int, int);

// Helper: transpose [batch, seq, dim] -> [batch, dim, seq] for conv1d
__global__ void transpose_bsd_to_bds_bf16(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ in,
    int batch, int seq, int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * seq * dim;
    if (idx >= total) return;

    int b = idx / (seq * dim);
    int s = (idx / dim) % seq;
    int d = idx % dim;

    // in[b][s][d] -> out[b][d][s]
    out[b * dim * seq + d * seq + s] = in[b * seq * dim + s * dim + d];
}

// Helper: transpose [batch, dim, seq] -> [batch, seq, dim]
__global__ void transpose_bds_to_bsd_bf16(
    __nv_bfloat16* __restrict__ out,
    const __nv_bfloat16* __restrict__ in,
    int batch, int dim, int seq
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * dim * seq;
    if (idx >= total) return;

    int b = idx / (dim * seq);
    int d = (idx / seq) % dim;
    int s = idx % seq;

    // in[b][d][s] -> out[b][s][d]
    out[b * seq * dim + s * dim + d] = in[b * dim * seq + d * seq + s];
}

// ============================================================================
// Weight Management
// ============================================================================

extern "C" int cuda_create_conv_layer_weights_bf16(
    void** weights_out,
    const void* h_in_proj,
    const void* h_conv,
    const void* h_out_proj,
    const void* h_op_norm,
    const void* h_ffn_norm,
    const void* h_gate,
    const void* h_up,
    const void* h_down,
    int hidden, int intermediate, int kernel_size, float norm_eps
) {
    ConvLayerWeights* w = (ConvLayerWeights*)calloc(1, sizeof(ConvLayerWeights));
    if (!w) return -1;

    w->hidden_size = hidden;
    w->intermediate_size = intermediate;
    w->conv_kernel_size = kernel_size;
    w->norm_eps = norm_eps;

    cudaError_t err;
    int result = -1;

    // Allocate and copy each weight buffer
    #define ALLOC_AND_COPY(field, src, num_bytes) do { \
        err = cudaMalloc(&w->field, num_bytes); \
        if (err != cudaSuccess) goto cleanup; \
        err = cudaMemcpy(w->field, src, num_bytes, cudaMemcpyHostToDevice); \
        if (err != cudaSuccess) goto cleanup; \
    } while(0)

    size_t bf16 = sizeof(__nv_bfloat16);  // 2 bytes

    ALLOC_AND_COPY(in_proj_weight, h_in_proj, 3 * hidden * hidden * bf16);
    ALLOC_AND_COPY(conv_weight, h_conv, hidden * kernel_size * bf16);
    ALLOC_AND_COPY(out_proj_weight, h_out_proj, hidden * hidden * bf16);
    ALLOC_AND_COPY(operator_norm, h_op_norm, hidden * bf16);
    ALLOC_AND_COPY(ffn_norm, h_ffn_norm, hidden * bf16);
    ALLOC_AND_COPY(gate_weight, h_gate, intermediate * hidden * bf16);
    ALLOC_AND_COPY(up_weight, h_up, intermediate * hidden * bf16);
    ALLOC_AND_COPY(down_weight, h_down, hidden * intermediate * bf16);

    #undef ALLOC_AND_COPY

    *weights_out = w;
    return 0;

cleanup:
    cuda_free_conv_layer_weights(w);
    return result;
}

extern "C" void cuda_free_conv_layer_weights(void* weights) {
    if (!weights) return;
    ConvLayerWeights* w = (ConvLayerWeights*)weights;

    if (w->in_proj_weight) cudaFree(w->in_proj_weight);
    if (w->conv_weight) cudaFree(w->conv_weight);
    if (w->out_proj_weight) cudaFree(w->out_proj_weight);
    if (w->operator_norm) cudaFree(w->operator_norm);
    if (w->ffn_norm) cudaFree(w->ffn_norm);
    if (w->gate_weight) cudaFree(w->gate_weight);
    if (w->up_weight) cudaFree(w->up_weight);
    if (w->down_weight) cudaFree(w->down_weight);

    free(w);
}

// ============================================================================
// Forward Pass
// ============================================================================
// Flow:
// 1. RMSNorm(input, operator_norm) -> normed
// 2. in_proj GEMM: [B*S, H] -> [B*S, 3*H]
// 3. Chunk into B, C, x each [B*S, H], transpose to [B, H, S]
// 4. Bx = B * x (element-wise)
// 5. conv1d(Bx) -> conv_out
// 6. y = C * conv_out
// 7. Transpose back to [B, S, H], out_proj GEMM
// 8. Residual add
// 9. RMSNorm -> SwiGLU FFN -> residual add

extern "C" int cuda_conv_layer_forward_bf16(
    void* output,
    const void* input,
    const void* weights,
    void* conv_state,
    int batch, int seq_len, int position
) {
    ConvLayerWeights* w = (ConvLayerWeights*)weights;
    int H = w->hidden_size;
    int I = w->intermediate_size;
    int K = w->conv_kernel_size;
    int BS = batch * seq_len;
    size_t bf16 = sizeof(__nv_bfloat16);

    int result = -1;

    // Allocate temporary buffers
    void* normed = nullptr;
    void* projected = nullptr;     // [BS, 3*H]
    void* B_bds = nullptr;         // [batch, H, S] after transpose
    void* C_bds = nullptr;         // [batch, H, S] after transpose
    void* x_bds = nullptr;         // [batch, H, S] after transpose
    void* Bx = nullptr;            // [batch, H, S]
    void* conv_out = nullptr;      // [batch, H, S]
    void* y_bds = nullptr;         // [batch, H, S]
    void* y_bsd = nullptr;         // [batch, S, H] after transpose back
    void* block_out = nullptr;     // [BS, H]
    void* residual = nullptr;      // [BS, H]
    void* ffn_normed = nullptr;    // [BS, H]
    void* gate_out = nullptr;      // [BS, I]
    void* up_out = nullptr;        // [BS, I]
    void* ffn_mid = nullptr;       // [BS, I]
    void* ffn_out = nullptr;       // [BS, H]

    cudaError_t err;
    err = cudaMalloc(&normed, BS * H * bf16); if (err) goto cleanup;
    err = cudaMalloc(&projected, BS * 3 * H * bf16); if (err) goto cleanup;
    err = cudaMalloc(&B_bds, batch * H * seq_len * bf16); if (err) goto cleanup;
    err = cudaMalloc(&C_bds, batch * H * seq_len * bf16); if (err) goto cleanup;
    err = cudaMalloc(&x_bds, batch * H * seq_len * bf16); if (err) goto cleanup;
    err = cudaMalloc(&Bx, batch * H * seq_len * bf16); if (err) goto cleanup;
    err = cudaMalloc(&conv_out, batch * H * seq_len * bf16); if (err) goto cleanup;
    err = cudaMalloc(&y_bds, batch * H * seq_len * bf16); if (err) goto cleanup;
    err = cudaMalloc(&y_bsd, BS * H * bf16); if (err) goto cleanup;
    err = cudaMalloc(&block_out, BS * H * bf16); if (err) goto cleanup;
    err = cudaMalloc(&residual, BS * H * bf16); if (err) goto cleanup;
    err = cudaMalloc(&ffn_normed, BS * H * bf16); if (err) goto cleanup;
    err = cudaMalloc(&gate_out, BS * I * bf16); if (err) goto cleanup;
    err = cudaMalloc(&up_out, BS * I * bf16); if (err) goto cleanup;
    err = cudaMalloc(&ffn_mid, BS * I * bf16); if (err) goto cleanup;
    err = cudaMalloc(&ffn_out, BS * H * bf16); if (err) goto cleanup;

    // 1. RMSNorm
    KERNEL_CHECK(cuda_bf16_rmsnorm(normed, input, w->operator_norm, BS, H, w->norm_eps));

    // 2. in_proj GEMM: [BS, H] @ [H, 3H]^T = [BS, 3H]
    // Weight stored as [3H, H], transpose_b=true
    KERNEL_CHECK(cuda_gemm_bf16(projected, normed, w->in_proj_weight, BS, H, 3*H, false, true));

    // 3. Chunk projected [BS, 3H] into B,C,x each [BS, H] then transpose to [B, H, S]
    {
        // B = projected[:, 0:H], C = projected[:, H:2H], x = projected[:, 2H:3H]
        // These are contiguous chunks in the 3H dimension
        __nv_bfloat16* proj_ptr = (__nv_bfloat16*)projected;

        // For seq_len==1 (decode), skip transpose — just use directly
        if (seq_len == 1) {
            // [batch, 1, H] is effectively [batch, H] — same layout as [batch, H, 1]
            // B chunk: offset 0
            // C chunk: offset H
            // x chunk: offset 2*H
            // Bx = B * x (element-wise on [batch*H])
            __nv_bfloat16* B_ptr = proj_ptr;
            __nv_bfloat16* C_ptr = proj_ptr + batch * H;
            __nv_bfloat16* x_ptr = proj_ptr + batch * 2 * H;

            KERNEL_CHECK(cuda_bf16_mul(Bx, B_ptr, x_ptr, batch * H));

            // 5. Conv1d update (single token decode)
            KERNEL_CHECK(cuda_causal_conv1d_update_bf16(Bx, conv_out, conv_state, w->conv_weight,
                                                         batch, H, K));

            // 6. y = C * conv_out
            KERNEL_CHECK(cuda_bf16_mul(y_bsd, C_ptr, conv_out, batch * H));
        } else {
            // Transpose each chunk from [B, S, H] to [B, H, S]
            int total = batch * seq_len * H;
            int block = 256;
            int grid = (total + block - 1) / block;

            // We need to handle the chunking: projected is [BS, 3H] in row-major
            // For each sample in batch, row s has [B[s], C[s], x[s]] each of size H
            // We need to extract and transpose each

            // Extract B (first H of each row), transpose to [B, H, S]
            // Since projected is [BS, 3H], B_chunk starts at offset 0 in each row
            // We'll use a custom kernel for this
            // For simplicity, copy to [B, S, H] views then transpose

            // Actually, projected[b*S+s, 0..H-1] = B, projected[b*S+s, H..2H-1] = C, etc.
            // We can treat projected as 3 separate [BS, H] arrays with stride 3H

            // Simpler: allocate [BS, H] scratch for each chunk
            void* B_bsd = nullptr;
            void* C_bsd_tmp = nullptr;
            void* x_bsd = nullptr;
            err = cudaMalloc(&B_bsd, BS * H * bf16); if (err) goto cleanup;
            err = cudaMalloc(&C_bsd_tmp, BS * H * bf16); if (err) goto cleanup;
            err = cudaMalloc(&x_bsd, BS * H * bf16); if (err) goto cleanup;

            // Copy strided chunks (this could be a kernel, but cudaMemcpy2D works)
            err = cudaMemcpy2D(B_bsd, H * bf16, projected, 3 * H * bf16,
                              H * bf16, BS, cudaMemcpyDeviceToDevice);
            if (err) { cudaFree(B_bsd); cudaFree(C_bsd_tmp); cudaFree(x_bsd); goto cleanup; }

            err = cudaMemcpy2D(C_bsd_tmp, H * bf16,
                              (char*)projected + H * bf16, 3 * H * bf16,
                              H * bf16, BS, cudaMemcpyDeviceToDevice);
            if (err) { cudaFree(B_bsd); cudaFree(C_bsd_tmp); cudaFree(x_bsd); goto cleanup; }

            err = cudaMemcpy2D(x_bsd, H * bf16,
                              (char*)projected + 2 * H * bf16, 3 * H * bf16,
                              H * bf16, BS, cudaMemcpyDeviceToDevice);
            if (err) { cudaFree(B_bsd); cudaFree(C_bsd_tmp); cudaFree(x_bsd); goto cleanup; }

            // Transpose each: [B, S, H] -> [B, H, S]
            transpose_bsd_to_bds_bf16<<<grid, block>>>(
                (__nv_bfloat16*)B_bds, (const __nv_bfloat16*)B_bsd,
                batch, seq_len, H);
            transpose_bsd_to_bds_bf16<<<grid, block>>>(
                (__nv_bfloat16*)C_bds, (const __nv_bfloat16*)C_bsd_tmp,
                batch, seq_len, H);
            transpose_bsd_to_bds_bf16<<<grid, block>>>(
                (__nv_bfloat16*)x_bds, (const __nv_bfloat16*)x_bsd,
                batch, seq_len, H);

            cudaFree(B_bsd);
            cudaFree(C_bsd_tmp);
            cudaFree(x_bsd);

            // 4. Bx = B * x (element-wise) [batch, H, S]
            KERNEL_CHECK(cuda_bf16_mul(Bx, B_bds, x_bds, batch * H * seq_len));

            // 5. Causal conv1d (prefill)
            KERNEL_CHECK(cuda_causal_conv1d_fwd_bf16(Bx, w->conv_weight, conv_out, conv_state,
                                                      batch, H, seq_len, K));

            // 6. y = C * conv_out [batch, H, S]
            KERNEL_CHECK(cuda_bf16_mul(y_bds, C_bds, conv_out, batch * H * seq_len));

            // 7. Transpose back: [B, H, S] -> [B, S, H]
            transpose_bds_to_bsd_bf16<<<grid, block>>>(
                (__nv_bfloat16*)y_bsd, (const __nv_bfloat16*)y_bds,
                batch, H, seq_len);
        }
    }

    // 7b. out_proj GEMM: [BS, H] @ [H, H]^T = [BS, H]
    KERNEL_CHECK(cuda_gemm_bf16(block_out, y_bsd, w->out_proj_weight, BS, H, H, false, true));

    // 8. Residual add
    KERNEL_CHECK(cuda_bf16_add(residual, input, block_out, BS * H));

    // 9. FFN: RMSNorm -> SwiGLU -> residual
    KERNEL_CHECK(cuda_bf16_rmsnorm(ffn_normed, residual, w->ffn_norm, BS, H, w->norm_eps));

    // SwiGLU: down(silu(gate(x)) * up(x))
    KERNEL_CHECK(cuda_gemm_bf16(gate_out, ffn_normed, w->gate_weight, BS, H, I, false, true));
    KERNEL_CHECK(cuda_gemm_bf16(up_out, ffn_normed, w->up_weight, BS, H, I, false, true));
    KERNEL_CHECK(cuda_bf16_silu(gate_out, gate_out, (size_t)BS * I));
    KERNEL_CHECK(cuda_bf16_mul(ffn_mid, gate_out, up_out, (size_t)BS * I));
    KERNEL_CHECK(cuda_gemm_bf16(ffn_out, ffn_mid, w->down_weight, BS, I, H, false, true));

    // Final residual add
    KERNEL_CHECK(cuda_bf16_add(output, residual, ffn_out, BS * H));

    result = 0;

cleanup:
    if (normed) cudaFree(normed);
    if (projected) cudaFree(projected);
    if (B_bds) cudaFree(B_bds);
    if (C_bds) cudaFree(C_bds);
    if (x_bds) cudaFree(x_bds);
    if (Bx) cudaFree(Bx);
    if (conv_out) cudaFree(conv_out);
    if (y_bds) cudaFree(y_bds);
    if (y_bsd) cudaFree(y_bsd);
    if (block_out) cudaFree(block_out);
    if (residual) cudaFree(residual);
    if (ffn_normed) cudaFree(ffn_normed);
    if (gate_out) cudaFree(gate_out);
    if (up_out) cudaFree(up_out);
    if (ffn_mid) cudaFree(ffn_mid);
    if (ffn_out) cudaFree(ffn_out);

    return result;
}
