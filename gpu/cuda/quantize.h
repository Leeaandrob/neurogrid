// quantize.h - INT8 quantization header
#ifndef QUANTIZE_H
#define QUANTIZE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Per-tensor quantization (FP16 -> INT8)
int cuda_quantize_per_tensor(
    void* output,         // INT8 output
    void* scale,          // FP32 scale [1]
    const void* input,    // FP16 input
    size_t num_elements
);

// Per-tensor dequantization (INT8 -> FP16)
int cuda_dequantize_per_tensor(
    void* output,         // FP16 output
    const void* input,    // INT8 input
    const void* scale,    // FP32 scale [1]
    size_t num_elements
);

// Per-column weight quantization
int cuda_quantize_weights(
    void* output,         // INT8 [K, N]
    void* scales,         // FP32 [N]
    const void* input,    // FP16 [K, N]
    int K,
    int N
);

// Per-row weight quantization (for transpose_b=true GEMM)
// Use this when weights are stored as [out_dim, in_dim] and will be transposed
int cuda_quantize_weights_per_row(
    void* output,         // INT8 [rows, cols]
    void* scales,         // FP32 [rows] - one scale per row (per output channel)
    const void* input,    // FP16 [rows, cols]
    int rows,
    int cols
);

#ifdef __cplusplus
}
#endif

#endif // QUANTIZE_H
