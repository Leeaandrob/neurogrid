// paged_attention.h — Paged Attention kernel declarations
#ifndef PAGED_ATTENTION_H
#define PAGED_ATTENTION_H

#ifdef __cplusplus
extern "C" {
#endif

// Create paged KV cache: allocates num_blocks * block_size * kv_heads * head_dim * 2(K+V)
int cuda_paged_kvcache_create(void** cache, int num_blocks, int num_kv_heads, int head_dim, int block_size);
void cuda_paged_kvcache_free(void* cache);

// Update KV cache: write new K,V at position using block_table for indirection
int cuda_paged_kvcache_update(void* cache, const void* new_key, const void* new_value,
    const int* d_block_table, int position);

// Paged Attention: attention with KV in non-contiguous blocks
int cuda_paged_attention(void* output, const void* query, const void* new_key, const void* new_value,
    void* cache, const int* d_block_table,
    int num_heads, int num_kv_heads, int head_dim, int position);

#ifdef __cplusplus
}
#endif

#endif // PAGED_ATTENTION_H
