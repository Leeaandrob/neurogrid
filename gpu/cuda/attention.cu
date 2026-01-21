// attention.cu - Basic attention mechanism for NeuroGrid
// This is a naive implementation - FlashAttention comes in later phases

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cfloat>
#include <stdio.h>
#include <stdlib.h>

#include "attention.h"
#include "matmul.h"
#include "kernels.h"

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
// Causal Mask Kernel
// ============================================================================
// Sets attention scores to -inf where query position > key position

__global__ void apply_causal_mask_kernel(
    half* __restrict__ scores,
    int query_len,
    int key_len
) {
    // scores: [num_heads, query_len, key_len]
    int head = blockIdx.x;
    int q_idx = blockIdx.y * blockDim.x + threadIdx.x;
    int k_idx = blockIdx.z * blockDim.y + threadIdx.y;

    if (q_idx < query_len && k_idx < key_len) {
        // Causal: can only attend to positions <= current
        if (k_idx > q_idx) {
            int idx = head * query_len * key_len + q_idx * key_len + k_idx;
            scores[idx] = __float2half(-INFINITY);
        }
    }
}

// ============================================================================
// Row-wise Softmax
// ============================================================================

__device__ float warp_reduce_sum_attn(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ float warp_reduce_max_attn(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__global__ void softmax_rows_kernel(
    half* __restrict__ output,
    const half* __restrict__ input,
    int num_rows,
    int row_len
) {
    extern __shared__ float shared[];

    int row = blockIdx.x;
    int tid = threadIdx.x;

    if (row >= num_rows) return;

    const half* in_row = input + row * row_len;
    half* out_row = output + row * row_len;

    // Step 1: Find max
    float max_val = -FLT_MAX;
    for (int i = tid; i < row_len; i += blockDim.x) {
        float val = __half2float(in_row[i]);
        max_val = fmaxf(max_val, val);
    }

    // Warp reduce
    max_val = warp_reduce_max_attn(max_val);

    // Store to shared memory
    int lane = tid % 32;
    int wid = tid / 32;
    if (lane == 0) {
        shared[wid] = max_val;
    }
    __syncthreads();

    // Final reduction
    if (tid < 32) {
        max_val = (tid < blockDim.x / 32) ? shared[tid] : -FLT_MAX;
        max_val = warp_reduce_max_attn(max_val);
    }

    __shared__ float s_max;
    if (tid == 0) {
        s_max = max_val;
    }
    __syncthreads();

    // Step 2: Compute exp sum
    float sum_exp = 0.0f;
    for (int i = tid; i < row_len; i += blockDim.x) {
        float val = __half2float(in_row[i]);
        sum_exp += expf(val - s_max);
    }

    // Warp reduce
    sum_exp = warp_reduce_sum_attn(sum_exp);

    if (lane == 0) {
        shared[wid] = sum_exp;
    }
    __syncthreads();

    if (tid < 32) {
        sum_exp = (tid < blockDim.x / 32) ? shared[tid] : 0.0f;
        sum_exp = warp_reduce_sum_attn(sum_exp);
    }

    __shared__ float s_sum;
    if (tid == 0) {
        s_sum = sum_exp;
    }
    __syncthreads();

    // Step 3: Normalize
    float inv_sum = 1.0f / s_sum;
    for (int i = tid; i < row_len; i += blockDim.x) {
        float val = __half2float(in_row[i]);
        out_row[i] = __float2half(expf(val - s_max) * inv_sum);
    }
}

// ============================================================================
// Basic Attention (No KV Cache)
// ============================================================================
// Computes: output = softmax(Q @ K^T / sqrt(d)) @ V

extern "C" int cuda_basic_attention(
    void* output,
    const void* query,
    const void* key,
    const void* value,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    bool causal
) {
    // Input shapes: [batch, num_heads, seq_len, head_dim]
    // We process batch * num_heads as separate attention heads

    int total_heads = batch_size * num_heads;
    int qk_size = seq_len * seq_len;
    int head_size = seq_len * head_dim;

    // Allocate temporary storage for attention scores
    half* scores;
    CUDA_CHECK(cudaMalloc(&scores, total_heads * qk_size * sizeof(half)));

    // Step 1: Compute Q @ K^T for all heads
    // scores[b,h,i,j] = sum_d(Q[b,h,i,d] * K[b,h,j,d])

    // Strided batched GEMM
    int result = cuda_batched_gemm_fp16(
        scores,
        query,
        key,
        total_heads,
        seq_len,      // M
        head_dim,     // K
        seq_len,      // N
        false,        // no transpose Q
        true,         // transpose K
        head_size,    // stride_a (Q)
        head_size,    // stride_b (K)
        qk_size       // stride_c (scores)
    );
    if (result != 0) {
        cudaFree(scores);
        return result;
    }

    // Step 2: Scale by 1/sqrt(head_dim)
    float scale = 1.0f / sqrtf((float)head_dim);
    result = cuda_scale(scores, scores, scale, total_heads * qk_size);
    if (result != 0) {
        cudaFree(scores);
        return result;
    }

    // Step 3: Apply causal mask if needed
    if (causal) {
        dim3 block(16, 16);
        dim3 grid(total_heads,
                  (seq_len + block.x - 1) / block.x,
                  (seq_len + block.y - 1) / block.y);

        apply_causal_mask_kernel<<<grid, block>>>(
            scores,
            seq_len,
            seq_len
        );
        CUDA_CHECK(cudaGetLastError());
    }

    // Step 4: Softmax along last dimension
    int num_rows = total_heads * seq_len;
    int block_size = min(1024, seq_len);
    block_size = ((block_size + 31) / 32) * 32;
    int shared_size = (block_size / 32) * sizeof(float);

    softmax_rows_kernel<<<num_rows, block_size, shared_size>>>(
        scores,
        scores,
        num_rows,
        seq_len
    );
    CUDA_CHECK(cudaGetLastError());

    // Step 5: Compute scores @ V
    // output[b,h,i,d] = sum_j(scores[b,h,i,j] * V[b,h,j,d])

    result = cuda_batched_gemm_fp16(
        output,
        scores,
        value,
        total_heads,
        seq_len,      // M
        seq_len,      // K
        head_dim,     // N
        false,        // no transpose scores
        false,        // no transpose V
        qk_size,      // stride_a (scores)
        head_size,    // stride_b (V)
        head_size     // stride_c (output)
    );

    CUDA_CHECK(cudaFree(scores));

    return result;
}

// ============================================================================
// KV Cache Structure
// ============================================================================

struct KVCache {
    half* k_cache;      // [batch, num_heads, max_seq, head_dim]
    half* v_cache;      // [batch, num_heads, max_seq, head_dim]
    int batch_size;
    int num_heads;
    int head_dim;
    int max_seq_len;
    int current_len;    // Current sequence length
};

extern "C" int cuda_kvcache_create(
    void** cache,
    int batch_size,
    int num_heads,
    int head_dim,
    int max_seq_len
) {
    KVCache* kv = (KVCache*)malloc(sizeof(KVCache));
    if (!kv) return -1;

    kv->batch_size = batch_size;
    kv->num_heads = num_heads;
    kv->head_dim = head_dim;
    kv->max_seq_len = max_seq_len;
    kv->current_len = 0;

    size_t cache_size = batch_size * num_heads * max_seq_len * head_dim * sizeof(half);

    CUDA_CHECK(cudaMalloc(&kv->k_cache, cache_size));
    CUDA_CHECK(cudaMalloc(&kv->v_cache, cache_size));

    // Initialize to zero
    CUDA_CHECK(cudaMemset(kv->k_cache, 0, cache_size));
    CUDA_CHECK(cudaMemset(kv->v_cache, 0, cache_size));

    *cache = kv;
    return 0;
}

extern "C" void cuda_kvcache_free(void* cache) {
    if (!cache) return;

    KVCache* kv = (KVCache*)cache;
    if (kv->k_cache) cudaFree(kv->k_cache);
    if (kv->v_cache) cudaFree(kv->v_cache);
    free(kv);
}

extern "C" int cuda_kvcache_get_length(void* cache) {
    if (!cache) return 0;
    KVCache* kv = (KVCache*)cache;
    return kv->current_len;
}

// Kernel to copy K/V to cache at specific position
__global__ void update_cache_kernel(
    half* __restrict__ cache,
    const half* __restrict__ new_data,
    int batch_size,
    int num_heads,
    int head_dim,
    int max_seq_len,
    int position
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * num_heads * head_dim;

    if (idx < total) {
        int b = idx / (num_heads * head_dim);
        int h = (idx / head_dim) % num_heads;
        int d = idx % head_dim;

        int cache_idx = b * num_heads * max_seq_len * head_dim +
                        h * max_seq_len * head_dim +
                        position * head_dim +
                        d;

        cache[cache_idx] = new_data[idx];
    }
}

extern "C" int cuda_kvcache_update(
    void* cache,
    const void* k,
    const void* v,
    int position
) {
    KVCache* kv = (KVCache*)cache;

    if (position >= kv->max_seq_len) {
        fprintf(stderr, "Position %d exceeds max_seq_len %d\n", position, kv->max_seq_len);
        return -1;
    }

    int total = kv->batch_size * kv->num_heads * kv->head_dim;
    int block_size = 256;
    int num_blocks = (total + block_size - 1) / block_size;

    update_cache_kernel<<<num_blocks, block_size>>>(
        kv->k_cache,
        (const half*)k,
        kv->batch_size,
        kv->num_heads,
        kv->head_dim,
        kv->max_seq_len,
        position
    );
    CUDA_CHECK(cudaGetLastError());

    update_cache_kernel<<<num_blocks, block_size>>>(
        kv->v_cache,
        (const half*)v,
        kv->batch_size,
        kv->num_heads,
        kv->head_dim,
        kv->max_seq_len,
        position
    );
    CUDA_CHECK(cudaGetLastError());

    if (position >= kv->current_len) {
        kv->current_len = position + 1;
    }

    return 0;
}

// ============================================================================
// Attention with KV Cache
// ============================================================================

// GQA attention kernel: computes attention for query heads using corresponding KV heads
// Each group of (num_heads / num_kv_heads) query heads shares one KV head
// This is a simple single-thread-per-head implementation for correctness
__global__ void gqa_attention_kernel(
    half* __restrict__ output,          // [batch * num_heads, head_dim]
    const half* __restrict__ query,      // [batch * num_heads, head_dim]
    const half* __restrict__ k_cache,    // [batch * num_kv_heads, max_seq_len, head_dim]
    const half* __restrict__ v_cache,    // [batch * num_kv_heads, max_seq_len, head_dim]
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int max_seq_len,
    int kv_len,
    float scale
) {
    // One block per query head, threads cooperate on the computation
    int query_head = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    if (query_head >= batch_size * num_heads) return;

    // Map query head to KV head using GQA grouping
    int batch_idx = query_head / num_heads;
    int head_in_batch = query_head % num_heads;
    int heads_per_kv = num_heads / num_kv_heads;
    int kv_head_in_batch = head_in_batch / heads_per_kv;
    int kv_head = batch_idx * num_kv_heads + kv_head_in_batch;

    // Pointers for this query/KV head
    const half* q = query + query_head * head_dim;
    const half* k = k_cache + kv_head * max_seq_len * head_dim;
    const half* v = v_cache + kv_head * max_seq_len * head_dim;
    half* out = output + query_head * head_dim;

    // Shared memory for scores
    extern __shared__ float shared[];
    float* scores = shared;  // [kv_len]
    float* thread_max = shared + kv_len;  // [block_size]
    float* thread_sum = shared + kv_len + block_size;  // [block_size]

    // Step 1: Compute Q @ K^T scores (each thread handles some keys)
    float local_max = -FLT_MAX;
    for (int j = tid; j < kv_len; j += block_size) {
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += __half2float(q[d]) * __half2float(k[j * head_dim + d]);
        }
        score *= scale;
        scores[j] = score;
        local_max = fmaxf(local_max, score);
    }
    thread_max[tid] = local_max;
    __syncthreads();

    // Reduce to find global max
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < block_size) {
            thread_max[tid] = fmaxf(thread_max[tid], thread_max[tid + s]);
        }
        __syncthreads();
    }
    float global_max = thread_max[0];
    __syncthreads();

    // Step 2: Compute exp(score - max) and local sum
    float local_sum = 0.0f;
    for (int j = tid; j < kv_len; j += block_size) {
        float exp_score = expf(scores[j] - global_max);
        scores[j] = exp_score;
        local_sum += exp_score;
    }
    thread_sum[tid] = local_sum;
    __syncthreads();

    // Reduce to find global sum
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < block_size) {
            thread_sum[tid] += thread_sum[tid + s];
        }
        __syncthreads();
    }
    float global_sum = fmaxf(thread_sum[0], 1e-9f);
    __syncthreads();

    // Normalize scores
    for (int j = tid; j < kv_len; j += block_size) {
        scores[j] /= global_sum;
    }
    __syncthreads();

    // Step 3: Compute weighted sum of values (output = scores @ V)
    // Each thread handles some dimensions of the output
    for (int d = tid; d < head_dim; d += block_size) {
        float val = 0.0f;
        for (int j = 0; j < kv_len; j++) {
            val += scores[j] * __half2float(v[j * head_dim + d]);
        }
        out[d] = __float2half(val);
    }
}

extern "C" int cuda_attention_with_kvcache(
    void* output,
    const void* query,          // [batch, num_heads, 1, head_dim]
    const void* new_key,        // [batch, num_kv_heads, 1, head_dim]
    const void* new_value,      // [batch, num_kv_heads, 1, head_dim]
    void* cache,
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int position
) {
    KVCache* kv = (KVCache*)cache;

    // First, update cache with new K, V (uses num_kv_heads from cache)
    int result = cuda_kvcache_update(cache, new_key, new_value, position);
    if (result != 0) return result;

    int total_query_heads = batch_size * num_heads;
    int kv_len = position + 1;  // Number of KV pairs to attend to
    float scale = 1.0f / sqrtf((float)head_dim);

    // For GQA, we use a custom kernel that maps query heads to KV heads
    int block_size = 128;  // Threads per block
    // Shared memory: scores[kv_len] + thread_max[block_size] + thread_sum[block_size]
    int shared_size = (kv_len + 2 * block_size) * sizeof(float);

    gqa_attention_kernel<<<total_query_heads, block_size, shared_size>>>(
        (half*)output,
        (const half*)query,
        kv->k_cache,
        kv->v_cache,
        batch_size,
        num_heads,
        num_kv_heads,
        head_dim,
        kv->max_seq_len,
        kv_len,
        scale
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    return 0;
}
