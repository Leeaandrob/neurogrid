// flash_decode.cu — Flash Attention optimized decode kernel
// Uses online softmax (no full score materialization) and KV tiling.
// For decode: single query token attending to all cached KV pairs.
//
// Key optimizations over naive attention:
// 1. Online softmax: compute max/sum incrementally, no need to store all scores
// 2. Tiled K/V access: process KV in tiles that fit in registers
// 3. FP32 accumulation: better numerical stability
// 4. Warp-level reduction: minimize shared memory usage
//
// Requires: compute capability >= 7.0 (works on both 2080 Ti and 4090)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>
#include <stdio.h>

#include "stream.h"
#include "attention.h"

#include "stream.h"
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        return -1; \
    } \
} while(0)

// ============================================================================
// Flash Decode Kernel — Online Softmax with KV Tiling
// ============================================================================
// One block per query head. Threads tile over KV length.
// Uses online softmax algorithm:
//   m_new = max(m_old, max(current_tile_scores))
//   l_new = l_old * exp(m_old - m_new) + sum(exp(scores - m_new))
//   o_new = o_old * (l_old * exp(m_old - m_new) / l_new) + sum(exp(scores - m_new) * V) / l_new
//
// This never materializes the full [kv_len] score array.

static constexpr int FLASH_TILE_SIZE = 32;  // KV positions per tile

__global__ void flash_decode_gqa_kernel(
    half* __restrict__ output,          // [batch * num_heads, head_dim]
    const half* __restrict__ query,      // [batch * num_heads, head_dim]
    const half* __restrict__ k_cache,    // [batch * num_kv_heads, max_seq_len, head_dim]
    const half* __restrict__ v_cache,    // [batch * num_kv_heads, max_seq_len, head_dim]
    const int* __restrict__ d_kv_len,   // GPU buffer [1]: kv_len read at runtime (CUDA Graph safe)
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int max_seq_len,
    float scale
) {
    int kv_len = d_kv_len[0];  // Read kv_len from GPU buffer
    int query_head = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    if (query_head >= batch_size * num_heads) return;

    // GQA head mapping
    int batch_idx = query_head / num_heads;
    int head_in_batch = query_head % num_heads;
    int heads_per_kv = num_heads / num_kv_heads;
    int kv_head_in_batch = head_in_batch / heads_per_kv;
    int kv_head = batch_idx * num_kv_heads + kv_head_in_batch;

    const half* q = query + query_head * head_dim;
    const half* k = k_cache + kv_head * max_seq_len * head_dim;
    const half* v = v_cache + kv_head * max_seq_len * head_dim;
    half* out = output + query_head * head_dim;

    // Load query into registers (head_dim typically 64 or 128)
    // Each thread loads head_dim/block_size elements
    float q_reg[4]; // max head_dim elements per thread (128/32=4)
    int elems_per_thread = (head_dim + block_size - 1) / block_size;
    for (int i = 0; i < elems_per_thread && tid * elems_per_thread + i < head_dim; i++) {
        int d = tid * elems_per_thread + i;
        q_reg[i] = __half2float(q[d]) * scale;
    }

    // Online softmax state per output dimension (accumulated in registers)
    float o_acc[4] = {0}; // output accumulator per thread's dims
    float m_prev = -FLT_MAX;  // running max
    float l_prev = 0.0f;      // running sum of exp

    // Shared memory for tile scores and reductions
    extern __shared__ float smem[];
    float* tile_scores = smem;                    // [FLASH_TILE_SIZE]
    float* thread_max_buf = smem + FLASH_TILE_SIZE;  // [block_size]
    float* thread_sum_buf = smem + FLASH_TILE_SIZE + block_size;  // [block_size]

    // Process KV in tiles
    for (int tile_start = 0; tile_start < kv_len; tile_start += FLASH_TILE_SIZE) {
        int tile_end = min(tile_start + FLASH_TILE_SIZE, kv_len);
        int tile_size = tile_end - tile_start;

        // Step 1: Compute Q·K scores for this tile
        // Each thread computes scores for some positions in the tile
        float local_tile_max = -FLT_MAX;
        for (int j = tid; j < tile_size; j += block_size) {
            int kv_pos = tile_start + j;
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                score += __half2float(q[d]) * __half2float(k[kv_pos * head_dim + d]);
            }
            score *= scale;
            tile_scores[j] = score;
            local_tile_max = fmaxf(local_tile_max, score);
        }

        // Reduce tile max
        thread_max_buf[tid] = local_tile_max;
        __syncthreads();
        for (int s = block_size / 2; s > 0; s >>= 1) {
            if (tid < s && tid + s < block_size)
                thread_max_buf[tid] = fmaxf(thread_max_buf[tid], thread_max_buf[tid + s]);
            __syncthreads();
        }
        float tile_max = thread_max_buf[0];
        __syncthreads();

        // Step 2: Online softmax update
        // m_new = max(m_prev, tile_max)
        float m_new = fmaxf(m_prev, tile_max);
        float correction = expf(m_prev - m_new); // scale factor for previous accumulator

        // Compute exp(score - m_new) and tile sum
        float local_sum = 0.0f;
        for (int j = tid; j < tile_size; j += block_size) {
            float exp_score = expf(tile_scores[j] - m_new);
            tile_scores[j] = exp_score;  // reuse buffer
            local_sum += exp_score;
        }
        thread_sum_buf[tid] = local_sum;
        __syncthreads();
        for (int s = block_size / 2; s > 0; s >>= 1) {
            if (tid < s && tid + s < block_size)
                thread_sum_buf[tid] += thread_sum_buf[tid + s];
            __syncthreads();
        }
        float tile_sum = thread_sum_buf[0];
        __syncthreads();

        // Update running sum: l_new = l_prev * correction + tile_sum
        float l_new = l_prev * correction + tile_sum;

        // Step 3: Accumulate weighted values
        // o_new = o_prev * (l_prev * correction / l_new) + sum(exp_score * V) / l_new
        float rescale = (l_new > 0) ? (l_prev * correction / l_new) : 0.0f;
        float norm = (l_new > 0) ? (1.0f / l_new) : 0.0f;

        // Each thread accumulates its output dimensions
        for (int di = 0; di < elems_per_thread; di++) {
            int d = tid * elems_per_thread + di;
            if (d >= head_dim) break;

            // Rescale previous accumulation
            o_acc[di] *= rescale;

            // Add contribution from this tile
            float v_sum = 0.0f;
            for (int j = 0; j < tile_size; j++) {
                v_sum += tile_scores[j] * __half2float(v[(tile_start + j) * head_dim + d]);
            }
            o_acc[di] += v_sum * norm;
        }

        m_prev = m_new;
        l_prev = l_new;
    }

    // Write output
    for (int di = 0; di < elems_per_thread; di++) {
        int d = tid * elems_per_thread + di;
        if (d < head_dim) {
            out[d] = __float2half(o_acc[di]);
        }
    }
}

// ============================================================================
// Flash Attention decode with KV cache — drop-in replacement
// ============================================================================

extern "C" int cuda_flash_attention_with_kvcache(
    void* output,
    const void* query,
    const void* new_key,
    const void* new_value,
    void* cache,
    const int* d_position,      // GPU buffer [1]: position (CUDA Graph safe)
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int head_dim
) {
    KVCache* kv = (KVCache*)cache;

    // Read position from GPU for cache update (host-side metadata)
    int position;
    CUDA_CHECK(cudaMemcpy(&position, d_position, sizeof(int), cudaMemcpyDeviceToHost));

    // Update cache first
    int result = cuda_kvcache_update(cache, new_key, new_value, position);
    if (result != 0) return result;

    int total_query_heads = batch_size * num_heads;
    float scale = 1.0f / sqrtf((float)head_dim);

    // Compute kv_len = position + 1 and write to GPU buffer for kernel
    int kv_len = position + 1;
    int* d_kv_len;
    CUDA_CHECK(cudaMalloc(&d_kv_len, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_kv_len, &kv_len, sizeof(int), cudaMemcpyHostToDevice));

    // Block size: enough threads to cover head_dim elements
    int block_size = 128;
    // Shared memory: tile_scores[TILE] + thread_max[block] + thread_sum[block]
    int shared_size = (FLASH_TILE_SIZE + 2 * block_size) * sizeof(float);

    flash_decode_gqa_kernel<<<total_query_heads, block_size, shared_size, ng_get_stream()>>>(
        (half*)output,
        (const half*)query,
        (const half*)kv->k_cache,
        (const half*)kv->v_cache,
        d_kv_len,
        batch_size,
        num_heads,
        num_kv_heads,
        head_dim,
        kv->max_seq_len,
        scale
    );
    CUDA_CHECK(cudaGetLastError());

    cudaFree(d_kv_len);

    return 0;
}

// ============================================================================
// Runtime capability check
// ============================================================================

extern "C" int cuda_flash_attention_supported() {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    // Flash decode works on CC >= 7.0 (even 2080 Ti)
    return (prop.major >= 7) ? 1 : 0;
}
