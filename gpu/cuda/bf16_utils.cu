// bf16_utils.cu - BFloat16 element-wise CUDA kernels for NeuroGrid
// Implements: BF16 RMSNorm, SiLU, Add, Mul, conversions, RoPE
// Requires compute capability >= 8.0 (Ampere: RTX 3090/A100+)

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdio.h>

#include "bf16_utils.h"

// Error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        return -1; \
    } \
} while(0)

// Warp reduction for BF16 kernels (FP32 accumulation)
__device__ __forceinline__ float bf16_warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ float bf16_block_reduce_sum(float val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = bf16_warp_reduce_sum(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    val = (threadIdx.x < blockDim.x / 32) ? shared[lane] : 0.0f;
    if (wid == 0) {
        val = bf16_warp_reduce_sum(val);
    }

    return val;
}

// ============================================================================
// BF16 RMSNorm Kernel
// ============================================================================

__global__ void bf16_rmsnorm_kernel(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    int hidden_dim,
    float eps
) {
    int token_idx = blockIdx.x;
    int tid = threadIdx.x;

    const __nv_bfloat16* x = input + token_idx * hidden_dim;
    __nv_bfloat16* y = output + token_idx * hidden_dim;

    // Step 1: Compute sum of squares using FP32 accumulation
    float sum_sq = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float val = __bfloat162float(x[i]);
        sum_sq += val * val;
    }

    sum_sq = bf16_block_reduce_sum(sum_sq);

    // Step 2: Compute rsqrt(mean + eps)
    __shared__ float s_rsqrt;
    if (tid == 0) {
        float mean = sum_sq / float(hidden_dim);
        s_rsqrt = rsqrtf(mean + eps);
    }
    __syncthreads();

    // Step 3: Normalize and scale
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float val = __bfloat162float(x[i]);
        float w = __bfloat162float(weight[i]);
        y[i] = __float2bfloat16(val * s_rsqrt * w);
    }
}

extern "C" int cuda_bf16_rmsnorm(
    void* output,
    const void* input,
    const void* weight,
    int num_tokens,
    int hidden_dim,
    float eps
) {
    int block_size = min(1024, hidden_dim);
    block_size = ((block_size + 31) / 32) * 32;

    bf16_rmsnorm_kernel<<<num_tokens, block_size>>>(
        (__nv_bfloat16*)output,
        (const __nv_bfloat16*)input,
        (const __nv_bfloat16*)weight,
        hidden_dim,
        eps
    );

    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// ============================================================================
// BF16 SiLU Kernel
// ============================================================================

__global__ void bf16_silu_kernel(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = __bfloat162float(input[idx]);
        float sigmoid_x = 1.0f / (1.0f + expf(-x));
        output[idx] = __float2bfloat16(x * sigmoid_x);
    }
}

extern "C" int cuda_bf16_silu(void* output, const void* input, size_t num_elements) {
    int block_size = 256;
    int num_blocks = (num_elements + block_size - 1) / block_size;

    bf16_silu_kernel<<<num_blocks, block_size>>>(
        (__nv_bfloat16*)output,
        (const __nv_bfloat16*)input,
        num_elements
    );

    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// ============================================================================
// BF16 Add Kernel
// ============================================================================

__global__ void bf16_add_kernel(
    __nv_bfloat16* __restrict__ c,
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float va = __bfloat162float(a[idx]);
        float vb = __bfloat162float(b[idx]);
        c[idx] = __float2bfloat16(va + vb);
    }
}

extern "C" int cuda_bf16_add(void* c, const void* a, const void* b, size_t num_elements) {
    int block_size = 256;
    int num_blocks = (num_elements + block_size - 1) / block_size;

    bf16_add_kernel<<<num_blocks, block_size>>>(
        (__nv_bfloat16*)c,
        (const __nv_bfloat16*)a,
        (const __nv_bfloat16*)b,
        num_elements
    );

    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// ============================================================================
// BF16 Mul Kernel
// ============================================================================

__global__ void bf16_mul_kernel(
    __nv_bfloat16* __restrict__ c,
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float va = __bfloat162float(a[idx]);
        float vb = __bfloat162float(b[idx]);
        c[idx] = __float2bfloat16(va * vb);
    }
}

extern "C" int cuda_bf16_mul(void* c, const void* a, const void* b, size_t num_elements) {
    int block_size = 256;
    int num_blocks = (num_elements + block_size - 1) / block_size;

    bf16_mul_kernel<<<num_blocks, block_size>>>(
        (__nv_bfloat16*)c,
        (const __nv_bfloat16*)a,
        (const __nv_bfloat16*)b,
        num_elements
    );

    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// ============================================================================
// BF16 <-> FP32 Conversion Kernels
// ============================================================================

__global__ void bf16_to_fp32_kernel(
    float* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __bfloat162float(input[idx]);
    }
}

extern "C" int cuda_bf16_to_fp32(void* output, const void* input, size_t num_elements) {
    int block_size = 256;
    int num_blocks = (num_elements + block_size - 1) / block_size;

    bf16_to_fp32_kernel<<<num_blocks, block_size>>>(
        (float*)output,
        (const __nv_bfloat16*)input,
        num_elements
    );

    CUDA_CHECK(cudaGetLastError());
    return 0;
}

__global__ void fp32_to_bf16_kernel(
    __nv_bfloat16* __restrict__ output,
    const float* __restrict__ input,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __float2bfloat16(input[idx]);
    }
}

extern "C" int cuda_fp32_to_bf16(void* output, const void* input, size_t num_elements) {
    int block_size = 256;
    int num_blocks = (num_elements + block_size - 1) / block_size;

    fp32_to_bf16_kernel<<<num_blocks, block_size>>>(
        (__nv_bfloat16*)output,
        (const float*)input,
        num_elements
    );

    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// ============================================================================
// FP16 <-> BF16 Direct Conversion Kernels
// ============================================================================

__global__ void fp16_to_bf16_kernel(
    __nv_bfloat16* __restrict__ output,
    const half* __restrict__ input,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __float2bfloat16(__half2float(input[idx]));
    }
}

extern "C" int cuda_fp16_to_bf16(void* output, const void* input, size_t num_elements) {
    int block_size = 256;
    int num_blocks = (num_elements + block_size - 1) / block_size;

    fp16_to_bf16_kernel<<<num_blocks, block_size>>>(
        (__nv_bfloat16*)output,
        (const half*)input,
        num_elements
    );

    CUDA_CHECK(cudaGetLastError());
    return 0;
}

__global__ void bf16_to_fp16_kernel(
    half* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __float2half(__bfloat162float(input[idx]));
    }
}

extern "C" int cuda_bf16_to_fp16(void* output, const void* input, size_t num_elements) {
    int block_size = 256;
    int num_blocks = (num_elements + block_size - 1) / block_size;

    bf16_to_fp16_kernel<<<num_blocks, block_size>>>(
        (half*)output,
        (const __nv_bfloat16*)input,
        num_elements
    );

    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// ============================================================================
// BF16 RoPE Kernel with configurable theta
// ============================================================================

#define ROPE_STYLE_SPLIT_HALF   0
#define ROPE_STYLE_INTERLEAVED  1

__global__ void bf16_rope_kernel_with_theta(
    __nv_bfloat16* __restrict__ output,
    const __nv_bfloat16* __restrict__ input,
    const int* __restrict__ positions,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim,
    int rope_style,
    float rope_theta
) {
    int batch_seq_head = blockIdx.x;
    int pair_idx = threadIdx.x;

    if (pair_idx >= head_dim / 2) return;

    int batch = batch_seq_head / (seq_len * num_heads);
    int seq = (batch_seq_head / num_heads) % seq_len;
    int head = batch_seq_head % num_heads;

    int pos = positions[batch * seq_len + seq];

    float inv_freq = 1.0f / powf(rope_theta, 2.0f * pair_idx / float(head_dim));
    float theta = float(pos) * inv_freq;
    float cos_theta = cosf(theta);
    float sin_theta = sinf(theta);

    int base = batch * seq_len * num_heads * head_dim +
               seq * num_heads * head_dim +
               head * head_dim;

    int idx1, idx2;
    if (rope_style == ROPE_STYLE_INTERLEAVED) {
        idx1 = base + pair_idx * 2;
        idx2 = base + pair_idx * 2 + 1;
    } else {
        idx1 = base + pair_idx;
        idx2 = base + pair_idx + head_dim / 2;
    }

    float x1 = __bfloat162float(input[idx1]);
    float x2 = __bfloat162float(input[idx2]);

    float x1_rot = x1 * cos_theta - x2 * sin_theta;
    float x2_rot = x1 * sin_theta + x2 * cos_theta;

    output[idx1] = __float2bfloat16(x1_rot);
    output[idx2] = __float2bfloat16(x2_rot);
}

extern "C" int cuda_bf16_rope_with_theta(
    void* output,
    const void* input,
    const int* positions,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim,
    int rope_style,
    float rope_theta
) {
    int num_blocks = batch_size * seq_len * num_heads;
    int block_size = head_dim / 2;
    if (block_size > 1024) block_size = 1024;

    bf16_rope_kernel_with_theta<<<num_blocks, block_size>>>(
        (__nv_bfloat16*)output,
        (const __nv_bfloat16*)input,
        positions,
        batch_size, seq_len, num_heads, head_dim,
        rope_style, rope_theta
    );

    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// ============================================================================
// BF16 Support Check
// ============================================================================

extern "C" int cuda_check_bf16_support(int* supported) {
    int device;
    cudaError_t err = cudaGetDevice(&device);
    if (err != cudaSuccess) {
        *supported = 0;
        return -1;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) {
        *supported = 0;
        return -1;
    }

    // BF16 requires compute capability >= 8.0 (Ampere)
    *supported = (prop.major > 8 || (prop.major == 8 && prop.minor >= 0)) ? 1 : 0;
    return 0;
}
