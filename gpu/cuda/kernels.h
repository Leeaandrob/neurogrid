// kernels.h - Basic CUDA kernels header
#ifndef KERNELS_H
#define KERNELS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// RMSNorm: y = x * rsqrt(mean(x^2) + eps) * weight
int cuda_rmsnorm(
    void* output,
    const void* input,
    const void* weight,
    int num_tokens,
    int hidden_dim,
    float eps
);

// SiLU activation: y = x * sigmoid(x)
int cuda_silu(void* output, const void* input, size_t num_elements);

// Element-wise addition: c = a + b
int cuda_add(void* c, const void* a, const void* b, size_t num_elements);

// Element-wise multiplication: c = a * b
int cuda_mul(void* c, const void* a, const void* b, size_t num_elements);

// RoPE style constants
#define ROPE_STYLE_SPLIT_HALF   0  // Llama 2, TinyLlama, Mistral: pairs (0,64), (1,65)...
#define ROPE_STYLE_INTERLEAVED  1  // Llama 3, GPT-NeoX: pairs (0,1), (2,3)...

// Rotary Position Embeddings with configurable style
int cuda_rope(
    void* output,
    const void* input,
    const int* positions,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim
);

// Rotary Position Embeddings with explicit style parameter
int cuda_rope_styled(
    void* output,
    const void* input,
    const int* positions,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim,
    int rope_style  // ROPE_STYLE_SPLIT_HALF or ROPE_STYLE_INTERLEAVED
);

// Rotary Position Embeddings with configurable theta
int cuda_rope_with_theta(
    void* output,
    const void* input,
    const int* positions,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim,
    int rope_style,
    float rope_theta  // Base frequency (10000.0 for Llama 2, 1000000.0 for Mistral Nemo)
);

// Fused SiLU + Mul (SwiGLU): output = SiLU(gate) * up
int cuda_silu_mul(void* output, const void* gate, const void* up, size_t num_elements);

// Fused Residual Add + RMSNorm: normed = RMSNorm(input + residual, weight)
int cuda_add_rmsnorm(
    void* normed_output,
    void* residual_output,
    const void* input,
    const void* residual_input,
    const void* weight,
    int num_tokens,
    int hidden_dim,
    float eps
);

// Softmax along last dimension
int cuda_softmax(void* output, const void* input, int num_rows, int row_size);

// Scale by constant: output = input * scale
int cuda_scale(void* output, const void* input, float scale, size_t num_elements);

#ifdef __cplusplus
}
#endif

#endif // KERNELS_H
