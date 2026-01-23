// attention.h - Attention mechanism header
#ifndef ATTENTION_H
#define ATTENTION_H

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Basic attention (no KV cache)
// Input/Output shapes: [batch, num_heads, seq_len, head_dim]
int cuda_basic_attention(
    void* output,
    const void* query,
    const void* key,
    const void* value,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    bool causal
);

// Basic attention with GQA support (for prompt processing)
// Q: [batch, num_heads, seq_len, head_dim]
// K, V: [batch, num_kv_heads, seq_len, head_dim]
int cuda_basic_attention_gqa(
    void* output,
    const void* query,
    const void* key,
    const void* value,
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int seq_len,
    int head_dim,
    bool causal
);

// KV Cache management
int cuda_kvcache_create(
    void** cache,
    int batch_size,
    int num_heads,
    int head_dim,
    int max_seq_len
);

void cuda_kvcache_free(void* cache);

int cuda_kvcache_update(
    void* cache,
    const void* k,
    const void* v,
    int position
);

// Attention with KV cache (for autoregressive generation)
// Supports GQA (Grouped Query Attention) when num_kv_heads < num_heads
int cuda_attention_with_kvcache(
    void* output,
    const void* query,          // [batch, num_heads, 1, head_dim]
    const void* new_key,        // [batch, num_kv_heads, 1, head_dim]
    const void* new_value,      // [batch, num_kv_heads, 1, head_dim]
    void* cache,
    int batch_size,
    int num_heads,
    int num_kv_heads,           // For GQA: num_kv_heads < num_heads
    int head_dim,
    int position
);

#ifdef __cplusplus
}
#endif

#endif // ATTENTION_H
