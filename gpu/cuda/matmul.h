// matmul.h - Matrix multiplication header
#ifndef MATMUL_H
#define MATMUL_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Initialize/shutdown cuBLAS
int cublas_init(void);
void cublas_shutdown(void);

// FP16 GEMM: C = A @ B
int cuda_gemm_fp16(
    void* c,
    const void* a,
    const void* b,
    int M,
    int K,
    int N,
    bool transpose_a,
    bool transpose_b
);

// INT8 GEMM with dequantization: C = A @ (B * scale)
// When transpose_b=true, B is stored as [N, K] and will be transposed
int cuda_gemm_int8(
    void* c,
    const void* a,        // FP16 [M, K]
    const void* b,        // INT8 [K, N] or [N, K] if transpose_b
    const void* scale,    // FP32 [N]
    int M,
    int K,
    int N,
    bool transpose_b      // If true, B is stored as [N, K]
);

// BF16 GEMM: C = A @ B using cublasGemmEx with CUDA_R_16BF + CUBLAS_COMPUTE_32F
int cuda_gemm_bf16(
    void* c,
    const void* a,
    const void* b,
    int M,
    int K,
    int N,
    bool transpose_a,
    bool transpose_b
);

// BF16 batched GEMM for attention
int cuda_batched_gemm_bf16(
    void* c,
    const void* a,
    const void* b,
    int batch_count,
    int M,
    int K,
    int N,
    bool transpose_a,
    bool transpose_b,
    long long int stride_a,
    long long int stride_b,
    long long int stride_c
);

// Batched GEMM for attention
int cuda_batched_gemm_fp16(
    void* c,
    const void* a,
    const void* b,
    int batch_count,
    int M,
    int K,
    int N,
    bool transpose_a,
    bool transpose_b,
    long long int stride_a,
    long long int stride_b,
    long long int stride_c
);

#ifdef __cplusplus
}
#endif

#endif // MATMUL_H
