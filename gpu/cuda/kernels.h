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

// Rotary Position Embeddings
int cuda_rope(
    void* output,
    const void* input,
    const int* positions,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim
);

// Softmax along last dimension
int cuda_softmax(void* output, const void* input, int num_rows, int row_size);

// Scale by constant: output = input * scale
int cuda_scale(void* output, const void* input, float scale, size_t num_elements);

#ifdef __cplusplus
}
#endif

#endif // KERNELS_H
