// bf16_utils.h - BFloat16 element-wise CUDA kernels
// Requires compute capability >= 8.0 (Ampere: RTX 3090/A100+)
#ifndef BF16_UTILS_H
#define BF16_UTILS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// BF16 RMSNorm: y = x * rsqrt(mean(x^2) + eps) * weight
// Input/output: BF16, accumulation: FP32
int cuda_bf16_rmsnorm(
    void* output,        // BF16 [num_tokens, hidden_dim]
    const void* input,   // BF16 [num_tokens, hidden_dim]
    const void* weight,  // BF16 [hidden_dim]
    int num_tokens,
    int hidden_dim,
    float eps
);

// BF16 SiLU activation: y = x * sigmoid(x)
int cuda_bf16_silu(void* output, const void* input, size_t num_elements);

// BF16 element-wise addition: c = a + b
int cuda_bf16_add(void* c, const void* a, const void* b, size_t num_elements);

// BF16 element-wise multiplication: c = a * b
int cuda_bf16_mul(void* c, const void* a, const void* b, size_t num_elements);

// BF16 to FP32 conversion
int cuda_bf16_to_fp32(void* output, const void* input, size_t num_elements);

// FP32 to BF16 conversion
int cuda_fp32_to_bf16(void* output, const void* input, size_t num_elements);

// FP16 to BF16 conversion (via FP32 intermediate)
int cuda_fp16_to_bf16(void* output, const void* input, size_t num_elements);

// BF16 to FP16 conversion (via FP32 intermediate)
int cuda_bf16_to_fp16(void* output, const void* input, size_t num_elements);

// BF16 RoPE with configurable theta
int cuda_bf16_rope_with_theta(
    void* output,
    const void* input,
    const int* positions,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim,
    int rope_style,
    float rope_theta
);

// Check if BF16 is supported on current device (compute capability >= 8.0)
int cuda_check_bf16_support(int* supported);

#ifdef __cplusplus
}
#endif

#endif // BF16_UTILS_H
