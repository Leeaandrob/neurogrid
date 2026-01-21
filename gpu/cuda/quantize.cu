// quantize.cu - INT8 quantization kernels
// Supports per-tensor and per-channel quantization

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>
#include <stdio.h>

#include "quantize.h"

// Error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        return -1; \
    } \
} while(0)

// Warp reduction for max
__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Block reduction for max
__device__ float block_reduce_max(float val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warp_reduce_max(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : -FLT_MAX;
    if (wid == 0) {
        val = warp_reduce_max(val);
    }

    return val;
}

// ============================================================================
// Per-Tensor Quantization
// ============================================================================
// scale = max(abs(x)) / 127
// x_int8 = round(x / scale)

// Kernel to find max absolute value
__global__ void find_max_abs_kernel(
    float* __restrict__ max_out,
    const half* __restrict__ input,
    size_t n
) {
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    float local_max = 0.0f;
    for (size_t i = gid; i < n; i += blockDim.x * gridDim.x) {
        float val = fabsf(__half2float(input[i]));
        local_max = fmaxf(local_max, val);
    }

    local_max = block_reduce_max(local_max);

    if (tid == 0) {
        atomicMax((int*)max_out, __float_as_int(local_max));
    }
}

// Kernel to quantize
__global__ void quantize_kernel(
    int8_t* __restrict__ output,
    const half* __restrict__ input,
    float scale,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = __half2float(input[idx]);
        float scaled = val / scale;
        // Clamp to INT8 range
        scaled = fmaxf(-127.0f, fminf(127.0f, roundf(scaled)));
        output[idx] = (int8_t)scaled;
    }
}

extern "C" int cuda_quantize_per_tensor(
    void* output,         // INT8 output
    void* scale,          // FP32 scale [1]
    const void* input,    // FP16 input
    size_t num_elements
) {
    // Step 1: Find max absolute value
    float* d_max;
    CUDA_CHECK(cudaMalloc(&d_max, sizeof(float)));
    float init_max = 0.0f;
    CUDA_CHECK(cudaMemcpy(d_max, &init_max, sizeof(float), cudaMemcpyHostToDevice));

    int block_size = 256;
    int num_blocks = min((int)((num_elements + block_size - 1) / block_size), 1024);

    find_max_abs_kernel<<<num_blocks, block_size>>>(
        d_max,
        (const half*)input,
        num_elements
    );
    CUDA_CHECK(cudaGetLastError());

    // Copy max to host and compute scale
    float max_abs;
    CUDA_CHECK(cudaMemcpy(&max_abs, d_max, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_max));

    float scale_val = max_abs / 127.0f;
    if (scale_val == 0.0f) scale_val = 1.0f; // Avoid division by zero

    // Store scale
    CUDA_CHECK(cudaMemcpy(scale, &scale_val, sizeof(float), cudaMemcpyHostToDevice));

    // Step 2: Quantize
    num_blocks = (num_elements + block_size - 1) / block_size;
    quantize_kernel<<<num_blocks, block_size>>>(
        (int8_t*)output,
        (const half*)input,
        scale_val,
        num_elements
    );
    CUDA_CHECK(cudaGetLastError());

    return 0;
}

// ============================================================================
// Per-Tensor Dequantization
// ============================================================================

__global__ void dequantize_kernel(
    half* __restrict__ output,
    const int8_t* __restrict__ input,
    float scale,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = float(input[idx]) * scale;
        output[idx] = __float2half(val);
    }
}

extern "C" int cuda_dequantize_per_tensor(
    void* output,         // FP16 output
    const void* input,    // INT8 input
    const void* scale,    // FP32 scale [1]
    size_t num_elements
) {
    // Get scale
    float scale_val;
    CUDA_CHECK(cudaMemcpy(&scale_val, scale, sizeof(float), cudaMemcpyDeviceToHost));

    int block_size = 256;
    int num_blocks = (num_elements + block_size - 1) / block_size;

    dequantize_kernel<<<num_blocks, block_size>>>(
        (half*)output,
        (const int8_t*)input,
        scale_val,
        num_elements
    );
    CUDA_CHECK(cudaGetLastError());

    return 0;
}

// ============================================================================
// Per-Column Weight Quantization
// ============================================================================
// For weight matrices [K, N], compute per-column scales

__global__ void find_column_max_kernel(
    float* __restrict__ scales,
    const half* __restrict__ weights,
    int K,
    int N
) {
    int col = blockIdx.x;
    if (col >= N) return;

    float max_abs = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        float val = fabsf(__half2float(weights[k * N + col]));
        max_abs = fmaxf(max_abs, val);
    }

    // Block reduce
    max_abs = block_reduce_max(max_abs);

    if (threadIdx.x == 0) {
        float scale = max_abs / 127.0f;
        if (scale == 0.0f) scale = 1.0f;
        scales[col] = scale;
    }
}

__global__ void quantize_weights_kernel(
    int8_t* __restrict__ output,
    const half* __restrict__ weights,
    const float* __restrict__ scales,
    int K,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = K * N;

    if (idx < total) {
        int k = idx / N;
        int n = idx % N;
        float val = __half2float(weights[idx]);
        float scaled = val / scales[n];
        scaled = fmaxf(-127.0f, fminf(127.0f, roundf(scaled)));
        output[idx] = (int8_t)scaled;
    }
}

extern "C" int cuda_quantize_weights(
    void* output,         // INT8 [K, N]
    void* scales,         // FP32 [N]
    const void* input,    // FP16 [K, N]
    int K,
    int N
) {
    // Step 1: Find per-column max
    int block_size = min(256, K);
    block_size = ((block_size + 31) / 32) * 32;

    find_column_max_kernel<<<N, block_size>>>(
        (float*)scales,
        (const half*)input,
        K,
        N
    );
    CUDA_CHECK(cudaGetLastError());

    // Step 2: Quantize
    int total = K * N;
    block_size = 256;
    int num_blocks = (total + block_size - 1) / block_size;

    quantize_weights_kernel<<<num_blocks, block_size>>>(
        (int8_t*)output,
        (const half*)input,
        (const float*)scales,
        K,
        N
    );
    CUDA_CHECK(cudaGetLastError());

    return 0;
}

// ============================================================================
// Per-Row Weight Quantization (for transpose_b=true GEMM)
// ============================================================================
// For weight matrices [rows, cols] where we want per-row scales (per output channel)

__global__ void find_row_max_kernel(
    float* __restrict__ scales,
    const half* __restrict__ weights,
    int rows,
    int cols
) {
    int row = blockIdx.x;
    if (row >= rows) return;

    float max_abs = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float val = fabsf(__half2float(weights[row * cols + c]));
        max_abs = fmaxf(max_abs, val);
    }

    // Block reduce
    max_abs = block_reduce_max(max_abs);

    if (threadIdx.x == 0) {
        float scale = max_abs / 127.0f;
        if (scale == 0.0f) scale = 1.0f;
        scales[row] = scale;
    }
}

__global__ void quantize_weights_per_row_kernel(
    int8_t* __restrict__ output,
    const half* __restrict__ weights,
    const float* __restrict__ scales,
    int rows,
    int cols
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;

    if (idx < total) {
        int row = idx / cols;
        float val = __half2float(weights[idx]);
        float scaled = val / scales[row];
        scaled = fmaxf(-127.0f, fminf(127.0f, roundf(scaled)));
        output[idx] = (int8_t)scaled;
    }
}

// Per-row quantization for weights stored as [out_dim, in_dim]
// This produces scales[out_dim] for use with transpose_b=true GEMM
extern "C" int cuda_quantize_weights_per_row(
    void* output,         // INT8 [rows, cols]
    void* scales,         // FP32 [rows] - one scale per row (per output channel)
    const void* input,    // FP16 [rows, cols]
    int rows,
    int cols
) {
    // Step 1: Find per-row max
    int block_size = min(256, cols);
    block_size = ((block_size + 31) / 32) * 32;

    find_row_max_kernel<<<rows, block_size>>>(
        (float*)scales,
        (const half*)input,
        rows,
        cols
    );
    CUDA_CHECK(cudaGetLastError());

    // Step 2: Quantize
    int total = rows * cols;
    block_size = 256;
    int num_blocks = (total + block_size - 1) / block_size;

    quantize_weights_per_row_kernel<<<num_blocks, block_size>>>(
        (int8_t*)output,
        (const half*)input,
        (const float*)scales,
        rows,
        cols
    );
    CUDA_CHECK(cudaGetLastError());

    return 0;
}
