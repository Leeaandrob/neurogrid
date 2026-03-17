// conv.cu - Depthwise causal conv1d CUDA kernels for LFM2
// Implements prefill (full sequence) and decode (single token) kernels
// Conv state maintained in FP32 for numerical stability

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdio.h>

#include "conv.h"

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
// Prefill Kernel: process full sequence through depthwise causal conv1d
// ============================================================================
// Input x:     [batch, dim, seqlen] BF16
// Weight:      [dim, width] BF16
// Output:      [batch, dim, seqlen] BF16
// Conv state:  [batch, dim, width] FP32 (updated with final state)
//
// Grid: (batch, dim)
// Block: min(seqlen, 128)

__global__ void causal_conv1d_fwd_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ out,
    float* __restrict__ conv_state,
    int batch, int dim, int seqlen, int width
) {
    int b = blockIdx.x;
    int d = blockIdx.y;

    if (b >= batch || d >= dim) return;

    // Load weight for this channel into registers (width=3 typically)
    float w[8]; // Max supported width
    for (int i = 0; i < width && i < 8; i++) {
        w[i] = __bfloat162float(weight[d * width + i]);
    }

    // Process each output position with causal padding
    // Causal: pad left by (width-1) zeros, then sliding window
    for (int t = threadIdx.x; t < seqlen; t += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < width; k++) {
            // Input position for this tap: t - (width - 1) + k
            int in_pos = t - (width - 1) + k;
            float x_val = 0.0f;
            if (in_pos >= 0 && in_pos < seqlen) {
                x_val = __bfloat162float(x[b * dim * seqlen + d * seqlen + in_pos]);
            }
            acc += w[k] * x_val;
        }
        out[b * dim * seqlen + d * seqlen + t] = __float2bfloat16(acc);
    }

    // Update conv state with the last (width) values from the sequence
    // This allows seamless continuation in decode mode
    __syncthreads();
    if (threadIdx.x == 0) {
        int state_base = b * dim * width + d * width;
        for (int k = 0; k < width; k++) {
            int pos = seqlen - width + k;
            if (pos >= 0 && pos < seqlen) {
                conv_state[state_base + k] = __bfloat162float(
                    x[b * dim * seqlen + d * seqlen + pos]);
            } else {
                conv_state[state_base + k] = 0.0f;
            }
        }
    }
}

extern "C" int cuda_causal_conv1d_fwd_bf16(
    const void* x,
    const void* weight,
    void* out,
    void* conv_state,
    int batch, int dim, int seqlen, int width
) {
    dim3 grid(batch, dim);
    int block_size = min(seqlen, 128);

    causal_conv1d_fwd_bf16_kernel<<<grid, block_size>>>(
        (const __nv_bfloat16*)x,
        (const __nv_bfloat16*)weight,
        (__nv_bfloat16*)out,
        (float*)conv_state,
        batch, dim, seqlen, width
    );

    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// ============================================================================
// Decode Kernel: single token update with FIFO state management
// ============================================================================
// Input x:     [batch, dim] BF16 single token
// Output:      [batch, dim] BF16
// Conv state:  [batch, dim, width] FP32 (shift left, insert new, 3-tap FIR)
// Weight:      [dim, width] BF16
//
// Grid: (batch, ceil(dim/64))
// Block: 64

__global__ void causal_conv1d_update_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ out,
    float* __restrict__ conv_state,
    const __nv_bfloat16* __restrict__ weight,
    int batch, int dim, int width
) {
    int b = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (d >= dim) return;

    // Load weight for this channel (width=3, fits in registers)
    float w[8]; // Max supported width
    for (int i = 0; i < width && i < 8; i++) {
        w[i] = __bfloat162float(weight[d * width + i]);
    }

    // Read new input value
    float x_val = __bfloat162float(x[b * dim + d]);

    // FIFO: shift state left, insert new value at end
    int base = b * dim * width + d * width;
    for (int i = 0; i < width - 1; i++) {
        conv_state[base + i] = conv_state[base + i + 1];
    }
    conv_state[base + width - 1] = x_val;

    // 3-tap FIR: out = sum(w[i] * state[i])
    float acc = 0.0f;
    for (int i = 0; i < width; i++) {
        acc += w[i] * conv_state[base + i];
    }

    out[b * dim + d] = __float2bfloat16(acc);
}

extern "C" int cuda_causal_conv1d_update_bf16(
    const void* x,
    void* out,
    void* conv_state,
    const void* weight,
    int batch, int dim, int width
) {
    dim3 grid(batch, (dim + 63) / 64);
    int block_size = 64;

    causal_conv1d_update_bf16_kernel<<<grid, block_size>>>(
        (const __nv_bfloat16*)x,
        (__nv_bfloat16*)out,
        (float*)conv_state,
        (const __nv_bfloat16*)weight,
        batch, dim, width
    );

    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// ============================================================================
// Conv State Management
// ============================================================================

extern "C" void* cuda_conv_state_create(int batch, int dim, int width) {
    void* ptr = nullptr;
    size_t size = (size_t)batch * dim * width * sizeof(float);
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cuda_conv_state_create: cudaMalloc failed: %s\n",
                cudaGetErrorString(err));
        return nullptr;
    }
    // Zero-initialize
    err = cudaMemset(ptr, 0, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cuda_conv_state_create: cudaMemset failed: %s\n",
                cudaGetErrorString(err));
        cudaFree(ptr);
        return nullptr;
    }
    return ptr;
}

extern "C" int cuda_conv_state_reset(void* state, int batch, int dim, int width) {
    if (state == nullptr) return -1;
    size_t size = (size_t)batch * dim * width * sizeof(float);
    CUDA_CHECK(cudaMemset(state, 0, size));
    return 0;
}

extern "C" void cuda_conv_state_free(void* state) {
    if (state != nullptr) {
        cudaFree(state);
    }
}
