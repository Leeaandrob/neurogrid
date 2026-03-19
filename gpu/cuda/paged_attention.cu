// paged_attention.cu — Paged Attention kernel for NeuroGrid
// Reads K/V from non-contiguous blocks via block_table indirection.
// Based on vLLM's PagedAttention V1 algorithm.
//
// Memory layout:
//   key_cache:   [num_blocks, num_kv_heads, block_size, head_dim] FP16
//   value_cache: [num_blocks, num_kv_heads, block_size, head_dim] FP16
//   block_table: [max_num_blocks_per_seq] int32 — maps logical → physical blocks
//
// Works on CC >= 7.0 (both RTX 2080 Ti and RTX 4090)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>
#include <stdio.h>

#include "paged_attention.h"

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        return -1; \
    } \
} while(0)

// ============================================================================
// Paged KV Cache Management
// ============================================================================

struct PagedKVCache {
    half* key_cache;     // [num_blocks, num_kv_heads, block_size, head_dim]
    half* value_cache;   // [num_blocks, num_kv_heads, block_size, head_dim]
    int num_blocks;
    int num_kv_heads;
    int head_dim;
    int block_size;
};

extern "C" int cuda_paged_kvcache_create(
    void** cache_out,
    int num_blocks,
    int num_kv_heads,
    int head_dim,
    int block_size
) {
    PagedKVCache* cache = (PagedKVCache*)calloc(1, sizeof(PagedKVCache));
    if (!cache) return -1;

    cache->num_blocks = num_blocks;
    cache->num_kv_heads = num_kv_heads;
    cache->head_dim = head_dim;
    cache->block_size = block_size;

    size_t block_data_size = (size_t)num_blocks * num_kv_heads * block_size * head_dim * sizeof(half);

    CUDA_CHECK(cudaMalloc(&cache->key_cache, block_data_size));
    CUDA_CHECK(cudaMalloc(&cache->value_cache, block_data_size));
    CUDA_CHECK(cudaMemset(cache->key_cache, 0, block_data_size));
    CUDA_CHECK(cudaMemset(cache->value_cache, 0, block_data_size));

    *cache_out = cache;
    return 0;
}

extern "C" void cuda_paged_kvcache_free(void* cache_ptr) {
    if (!cache_ptr) return;
    PagedKVCache* cache = (PagedKVCache*)cache_ptr;
    if (cache->key_cache) cudaFree(cache->key_cache);
    if (cache->value_cache) cudaFree(cache->value_cache);
    free(cache);
}

// ============================================================================
// Update: write new K,V into the correct block+slot
// ============================================================================

__global__ void paged_kvcache_update_kernel(
    half* __restrict__ key_cache,     // [num_blocks, num_kv_heads, block_size, head_dim]
    half* __restrict__ value_cache,   // [num_blocks, num_kv_heads, block_size, head_dim]
    const half* __restrict__ new_key, // [num_kv_heads, head_dim]
    const half* __restrict__ new_val, // [num_kv_heads, head_dim]
    const int* __restrict__ block_table, // [max_blocks_per_seq]
    int position,                     // Current token position
    int block_size,
    int num_kv_heads,
    int head_dim
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_kv_heads * head_dim;
    if (tid >= total) return;

    int kv_head = tid / head_dim;
    int d = tid % head_dim;

    // Which block and slot within block?
    int block_idx = position / block_size;
    int slot_in_block = position % block_size;
    int physical_block = block_table[block_idx];

    // Compute offset: [physical_block, kv_head, slot_in_block, d]
    size_t offset = (size_t)physical_block * num_kv_heads * block_size * head_dim
                  + kv_head * block_size * head_dim
                  + slot_in_block * head_dim
                  + d;

    key_cache[offset] = new_key[kv_head * head_dim + d];
    value_cache[offset] = new_val[kv_head * head_dim + d];
}

extern "C" int cuda_paged_kvcache_update(
    void* cache_ptr,
    const void* new_key,          // [num_kv_heads, head_dim] FP16 on GPU
    const void* new_value,        // [num_kv_heads, head_dim] FP16 on GPU
    const int* d_block_table,     // Block table on GPU
    int position
) {
    PagedKVCache* cache = (PagedKVCache*)cache_ptr;
    int total = cache->num_kv_heads * cache->head_dim;
    int block = 256;
    int grid = (total + block - 1) / block;

    paged_kvcache_update_kernel<<<grid, block>>>(
        cache->key_cache, cache->value_cache,
        (const half*)new_key, (const half*)new_value,
        d_block_table,
        position,
        cache->block_size,
        cache->num_kv_heads,
        cache->head_dim
    );
    CUDA_CHECK(cudaGetLastError());
    return 0;
}

// ============================================================================
// Paged Attention V1 Kernel — GQA support
// ============================================================================
// One block per query head. Reads K/V via block_table indirection.

__global__ void paged_attention_v1_kernel(
    half* __restrict__ output,          // [num_heads, head_dim]
    const half* __restrict__ query,      // [num_heads, head_dim]
    const half* __restrict__ key_cache,  // [num_blocks, num_kv_heads, block_size, head_dim]
    const half* __restrict__ value_cache,// [num_blocks, num_kv_heads, block_size, head_dim]
    const int* __restrict__ block_table, // [max_blocks_per_seq]
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int seq_len,                         // Number of tokens to attend to
    float scale
) {
    int head_idx = blockIdx.x;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    if (head_idx >= num_heads) return;

    // GQA: map query head to KV head
    int heads_per_kv = num_heads / num_kv_heads;
    int kv_head_idx = head_idx / heads_per_kv;

    const half* q = query + head_idx * head_dim;
    half* out = output + head_idx * head_dim;

    int num_blocks_used = (seq_len + block_size - 1) / block_size;

    // Shared memory for scores
    extern __shared__ float smem[];
    float* scores = smem;              // [seq_len] (padded)
    float* reduce_buf = smem + seq_len; // [num_threads]

    // Phase 1: Compute Q·K scores via block table
    float local_max = -FLT_MAX;

    for (int token = tid; token < seq_len; token += num_threads) {
        int block_idx = token / block_size;
        int slot = token % block_size;
        int phys_block = block_table[block_idx];

        // K offset: [phys_block, kv_head, slot, :]
        const half* k_ptr = key_cache
            + (size_t)phys_block * num_kv_heads * block_size * head_dim
            + kv_head_idx * block_size * head_dim
            + slot * head_dim;

        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += __half2float(q[d]) * __half2float(k_ptr[d]);
        }
        score *= scale;
        scores[token] = score;
        local_max = fmaxf(local_max, score);
    }

    // Reduce max
    reduce_buf[tid] = local_max;
    __syncthreads();
    for (int s = num_threads / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < num_threads)
            reduce_buf[tid] = fmaxf(reduce_buf[tid], reduce_buf[tid + s]);
        __syncthreads();
    }
    float global_max = reduce_buf[0];
    __syncthreads();

    // Phase 2: Softmax
    float local_sum = 0.0f;
    for (int token = tid; token < seq_len; token += num_threads) {
        float exp_score = expf(scores[token] - global_max);
        scores[token] = exp_score;
        local_sum += exp_score;
    }

    reduce_buf[tid] = local_sum;
    __syncthreads();
    for (int s = num_threads / 2; s > 0; s >>= 1) {
        if (tid < s && tid + s < num_threads)
            reduce_buf[tid] += reduce_buf[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / fmaxf(reduce_buf[0], 1e-9f);
    __syncthreads();

    // Normalize
    for (int token = tid; token < seq_len; token += num_threads) {
        scores[token] *= inv_sum;
    }
    __syncthreads();

    // Phase 3: Weighted sum of V via block table
    for (int d = tid; d < head_dim; d += num_threads) {
        float acc = 0.0f;
        for (int token = 0; token < seq_len; token++) {
            int block_idx = token / block_size;
            int slot = token % block_size;
            int phys_block = block_table[block_idx];

            const half* v_ptr = value_cache
                + (size_t)phys_block * num_kv_heads * block_size * head_dim
                + kv_head_idx * block_size * head_dim
                + slot * head_dim;

            acc += scores[token] * __half2float(v_ptr[d]);
        }
        out[d] = __float2half(acc);
    }
}

// ============================================================================
// Entry point: Paged Attention with KV Cache update
// ============================================================================

extern "C" int cuda_paged_attention(
    void* output,           // [num_heads, head_dim] FP16 on GPU
    const void* query,      // [num_heads, head_dim] FP16 on GPU
    const void* new_key,    // [num_kv_heads, head_dim] FP16 on GPU
    const void* new_value,  // [num_kv_heads, head_dim] FP16 on GPU
    void* cache_ptr,        // PagedKVCache*
    const int* d_block_table, // Block table on GPU [max_blocks]
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int position            // Current position (0-indexed)
) {
    PagedKVCache* cache = (PagedKVCache*)cache_ptr;

    // Step 1: Update cache with new K,V at position
    int result = cuda_paged_kvcache_update(
        cache_ptr, new_key, new_value, d_block_table, position);
    if (result != 0) return result;

    // Step 2: Run paged attention
    int seq_len = position + 1;
    float scale_val = 1.0f / sqrtf((float)head_dim);

    int num_threads = 128;
    // Use max shared memory to support CUDA Graph replay with different seq_lens
    // Max: 4096 tokens * 4 bytes + 128 threads * 4 bytes = ~16.5KB (well within limits)
    int max_seq_for_shared = 4096;
    int shared_size = (max_seq_for_shared + num_threads) * sizeof(float);

    paged_attention_v1_kernel<<<num_heads, num_threads, shared_size>>>(
        (half*)output,
        (const half*)query,
        (const half*)cache->key_cache,
        (const half*)cache->value_cache,
        d_block_table,
        num_heads,
        num_kv_heads,
        head_dim,
        cache->block_size,
        seq_len,
        scale_val
    );
    CUDA_CHECK(cudaGetLastError());

    return 0;
}
