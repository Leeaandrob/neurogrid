// matmul.cu - Matrix multiplication using cuBLAS
// Supports FP16, BF16, and INT8 operations

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <stdio.h>

#include "stream.h"
#include "matmul.h"

#include "stream.h"
// Error checking macros
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        return -1; \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d: %d\n", \
                __FILE__, __LINE__, status); \
        return -1; \
    } \
} while(0)

// Global cuBLAS handle
static cublasHandle_t g_cublas_handle = nullptr;
static cublasLtHandle_t g_cublaslt_handle = nullptr;

// Initialize cuBLAS (called from cuda_init)
int cublas_init() {
    if (g_cublas_handle == nullptr) {
        CUBLAS_CHECK(cublasCreate(&g_cublas_handle));
        CUBLAS_CHECK(cublasSetMathMode(g_cublas_handle, CUBLAS_TENSOR_OP_MATH));
    }
    if (g_cublaslt_handle == nullptr) {
        CUBLAS_CHECK(cublasLtCreate(&g_cublaslt_handle));
    }
    return 0;
}

// Set CUDA stream for cuBLAS (needed for CUDA Graph capture)
extern "C" int cublas_set_stream(void* stream) {
    if (g_cublas_handle == nullptr) {
        int ret = cublas_init();
        if (ret != 0) return ret;
    }
    CUBLAS_CHECK(cublasSetStream(g_cublas_handle, (cudaStream_t)stream));
    return 0;
}

void cublas_shutdown() {
    if (g_cublas_handle != nullptr) {
        cublasDestroy(g_cublas_handle);
        g_cublas_handle = nullptr;
    }
    if (g_cublaslt_handle != nullptr) {
        cublasLtDestroy(g_cublaslt_handle);
        g_cublaslt_handle = nullptr;
    }
}

// ============================================================================
// FP16 GEMM
// ============================================================================
// Computes C = alpha * A @ B + beta * C
// Uses Tensor Cores when available

extern "C" int cuda_gemm_fp16(
    void* c,
    const void* a,
    const void* b,
    int M,
    int K,
    int N,
    bool transpose_a,
    bool transpose_b
) {
    if (g_cublas_handle == nullptr) {
        int ret = cublas_init();
        if (ret != 0) return ret;
    }

    // Row-major to column-major conversion for cuBLAS:
    // We want: C[M,N] = op(A) @ op(B) in row-major
    //
    // Row-major data is interpreted by cuBLAS as transposed column-major:
    // - Row-major A[M,K] = Column-major (A^T)[K,M] with ld=K
    // - Row-major B[K,N] = Column-major (B^T)[N,K] with ld=N
    // - Row-major C[M,N] = Column-major (C^T)[N,M] with ld=N
    //
    // We want C = A @ B in row-major, which means:
    // C^T = (A @ B)^T = B^T @ A^T
    //
    // In cuBLAS terms: (C^T) = (B^T) @ (A^T)
    // cuBLAS sees: [N,M] = [N,K] @ [K,M]
    //
    // The data pointers already represent the transposed views,
    // so we call cuBLAS with NO operation flags for the base case.

    const half* h_a = (const half*)a;
    const half* h_b = (const half*)b;
    half* h_c = (half*)c;

    __half alpha = __float2half(1.0f);
    __half beta = __float2half(0.0f);

    // For row-major without user-requested transposes:
    // - cuBLAS sees A as (A^T)[K,M], we need (A^T), so op=N
    // - cuBLAS sees B as (B^T)[N,K], we need (B^T), so op=N
    //
    // When user requests transpose_a (compute A^T @ B):
    // - A is stored as [K,M] row-major (already the transposed shape)
    // - cuBLAS sees [M,K] col-major, we need (A^T)^T = A, so op=T
    //
    // When user requests transpose_b (compute A @ B^T):
    // - B is stored as [N,K] row-major (already the transposed shape)
    // - cuBLAS sees [K,N] col-major, we need (B^T)^T = B, so op=T
    cublasOperation_t op_a_cublas = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t op_b_cublas = transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;

    // Leading dimensions: always the number of columns in row-major storage
    // This equals the number of rows in cuBLAS's column-major interpretation
    int lda = transpose_a ? M : K;  // A stored as [?, lda] row-major
    int ldb = transpose_b ? K : N;  // B stored as [?, ldb] row-major
    int ldc = N;                     // C stored as [M, N] row-major

    // cuBLAS call: C^T = B^T @ A^T (with potential user-requested transposes)
    // Dimensions: [N,M] = [N,K] @ [K,M]
    CUBLAS_CHECK(cublasHgemm(
        g_cublas_handle,
        op_b_cublas,  // Operation on B (first operand in cuBLAS)
        op_a_cublas,  // Operation on A (second operand in cuBLAS)
        N,            // m: rows of cuBLAS result = cols of row-major C
        M,            // n: cols of cuBLAS result = rows of row-major C
        K,            // k: shared dimension
        &alpha,
        h_b,          // B matrix pointer (first operand)
        ldb,          // Leading dim of B
        h_a,          // A matrix pointer (second operand)
        lda,          // Leading dim of A
        &beta,
        h_c,          // C matrix pointer
        ldc           // Leading dim of C
    ));

    return 0;
}

// ============================================================================
// INT8 GEMM with Dequantization
// ============================================================================
// Computes: C = A @ (B * scale)
// Where A is FP16, B is INT8, scale is per-column FP32

// Kernel to dequantize INT8 weights during GEMM
// This is a simple approach - a more optimized version would fuse with GEMM
// rows: number of rows in storage
// cols: number of columns in storage
// scale_per_row: if true, use scales[row], else use scales[col]
__global__ void dequantize_and_copy(
    half* __restrict__ output,
    const int8_t* __restrict__ input,
    const float* __restrict__ scales,
    int rows,
    int cols,
    bool scale_per_row
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;

    if (idx < total) {
        int row = idx / cols;
        int col = idx % cols;
        float scale = scale_per_row ? scales[row] : scales[col];
        float val = float(input[idx]) * scale;
        output[idx] = __float2half(val);
    }
}

extern "C" int cuda_gemm_int8(
    void* c,
    const void* a,        // FP16 [M, K]
    const void* b,        // INT8 [K, N] or [N, K] if transpose_b
    const void* scale,    // FP32 [N]
    int M,
    int K,
    int N,
    bool transpose_b      // If true, B is stored as [N, K] and will be transposed
) {
    if (g_cublas_handle == nullptr) {
        int ret = cublas_init();
        if (ret != 0) return ret;
    }

    // Allocate temporary buffer for dequantized weights
    half* b_fp16;
    CUDA_CHECK(cudaMalloc(&b_fp16, K * N * sizeof(half)));

    // Dequantize B
    // When transpose_b=true, B is stored as [N, K] but we need to dequantize
    // in the stored layout, then let the GEMM handle the transpose
    int total = K * N;
    int block_size = 256;
    int num_blocks = (total + block_size - 1) / block_size;

    if (transpose_b) {
        // B is stored as [N, K], scales are per output dimension (N) = per row
        dequantize_and_copy<<<num_blocks, block_size, 0, ng_get_stream()>>>(
            b_fp16,
            (const int8_t*)b,
            (const float*)scale,
            N,     // rows in storage = N
            K,     // cols in storage = K
            true   // scale_per_row = true (scales[row] = scales[output_dim])
        );
    } else {
        // B is stored as [K, N], scales are per output dimension (N) = per column
        dequantize_and_copy<<<num_blocks, block_size, 0, ng_get_stream()>>>(
            b_fp16,
            (const int8_t*)b,
            (const float*)scale,
            K,     // rows in storage = K
            N,     // cols in storage = N
            false  // scale_per_row = false (scales[col] = scales[output_dim])
        );
    }
    CUDA_CHECK(cudaGetLastError());

    // Now perform FP16 GEMM
    int result = cuda_gemm_fp16(c, a, b_fp16, M, K, N, false, transpose_b);

    CUDA_CHECK(cudaFree(b_fp16));

    return result;
}

// ============================================================================
// Batched GEMM for Attention
// ============================================================================
// For computing Q @ K^T and scores @ V in attention

extern "C" int cuda_batched_gemm_fp16(
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
) {
    if (g_cublas_handle == nullptr) {
        int ret = cublas_init();
        if (ret != 0) return ret;
    }

    // Same row-major to column-major conversion as single GEMM
    // See cuda_gemm_fp16 for detailed explanation

    __half alpha = __float2half(1.0f);
    __half beta = __float2half(0.0f);

    cublasOperation_t op_a_cublas = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t op_b_cublas = transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;

    int lda = transpose_a ? M : K;
    int ldb = transpose_b ? K : N;
    int ldc = N;

    const half* h_a = (const half*)a;
    const half* h_b = (const half*)b;
    half* h_c = (half*)c;

    CUBLAS_CHECK(cublasHgemmStridedBatched(
        g_cublas_handle,
        op_b_cublas,
        op_a_cublas,
        N,
        M,
        K,
        &alpha,
        h_b,
        ldb,
        stride_b,
        h_a,
        lda,
        stride_a,
        &beta,
        h_c,
        ldc,
        stride_c,
        batch_count
    ));

    return 0;
}

// ============================================================================
// BF16 GEMM via cublasGemmEx
// ============================================================================
// Uses CUDA_R_16BF + CUBLAS_COMPUTE_32F (no CUBLAS_COMPUTE_16BF exists)

extern "C" int cuda_gemm_bf16(
    void* c,
    const void* a,
    const void* b,
    int M,
    int K,
    int N,
    bool transpose_a,
    bool transpose_b
) {
    if (g_cublas_handle == nullptr) {
        int ret = cublas_init();
        if (ret != 0) return ret;
    }

    float alpha = 1.0f;
    float beta = 0.0f;

    cublasOperation_t op_a_cublas = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t op_b_cublas = transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;

    int lda = transpose_a ? M : K;
    int ldb = transpose_b ? K : N;
    int ldc = N;

    CUBLAS_CHECK(cublasGemmEx(
        g_cublas_handle,
        op_b_cublas, op_a_cublas,
        N, M, K,
        &alpha,
        b, CUDA_R_16BF, ldb,
        a, CUDA_R_16BF, lda,
        &beta,
        c, CUDA_R_16BF, ldc,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));

    return 0;
}

extern "C" int cuda_batched_gemm_bf16(
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
) {
    if (g_cublas_handle == nullptr) {
        int ret = cublas_init();
        if (ret != 0) return ret;
    }

    float alpha = 1.0f;
    float beta = 0.0f;

    cublasOperation_t op_a_cublas = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t op_b_cublas = transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;

    int lda = transpose_a ? M : K;
    int ldb = transpose_b ? K : N;
    int ldc = N;

    CUBLAS_CHECK(cublasGemmStridedBatchedEx(
        g_cublas_handle,
        op_b_cublas, op_a_cublas,
        N, M, K,
        &alpha,
        b, CUDA_R_16BF, ldb, stride_b,
        a, CUDA_R_16BF, lda, stride_a,
        &beta,
        c, CUDA_R_16BF, ldc, stride_c,
        batch_count,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));

    return 0;
}
