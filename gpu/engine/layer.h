// layer.h - Transformer layer header
#ifndef LAYER_H
#define LAYER_H

#ifdef __cplusplus
extern "C" {
#endif

// Load layer weights from directory (safetensors format)
int cuda_load_layer_weights(void** weights, const char* path);

// Free layer weights
void cuda_free_layer_weights(void* weights);

// Create random layer weights (for testing)
int cuda_create_random_layer_weights(
    void** weights,
    int hidden_size,
    int intermediate_size,
    int num_heads,
    int num_kv_heads,
    int head_dim
);

// Create layer weights from host FP16 data
// All weight pointers are FP16 host memory, will be quantized to INT8 on GPU
int cuda_create_layer_weights_from_host(
    void** weights,
    const void* h_q_proj,       // FP16 [hidden_size, hidden_size]
    const void* h_k_proj,       // FP16 [hidden_size, kv_dim]
    const void* h_v_proj,       // FP16 [hidden_size, kv_dim]
    const void* h_o_proj,       // FP16 [hidden_size, hidden_size]
    const void* h_gate_proj,    // FP16 [hidden_size, intermediate_size]
    const void* h_up_proj,      // FP16 [hidden_size, intermediate_size]
    const void* h_down_proj,    // FP16 [intermediate_size, hidden_size]
    const void* h_attn_norm,    // FP16 [hidden_size]
    const void* h_ffn_norm,     // FP16 [hidden_size]
    int hidden_size,
    int intermediate_size,
    int num_heads,
    int num_kv_heads,
    int head_dim
);

// Complete transformer layer forward pass
int cuda_layer_forward(
    void* output,
    const void* input,
    const void* weights,
    void* kv_cache,
    const int* positions,
    int batch_size,
    int seq_len,
    int hidden_size,
    int intermediate_size,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float rms_norm_eps,
    float rope_theta
);

// FP16-pure layer forward (no INT8 quantization) — for LFM2 attention layers
int cuda_create_layer_weights_from_host_fp16(
    void** weights,
    const void* h_q_proj, const void* h_k_proj, const void* h_v_proj, const void* h_o_proj,
    const void* h_gate_proj, const void* h_up_proj, const void* h_down_proj,
    const void* h_attn_norm, const void* h_ffn_norm,
    const void* h_q_layernorm, const void* h_k_layernorm,
    int hidden_size, int intermediate_size, int num_heads, int num_kv_heads, int head_dim
);

int cuda_layer_forward_fp16(
    void* output, const void* input, const void* weights, void* kv_cache,
    const int* positions, int batch_size, int seq_len,
    int hidden_size, int intermediate_size, int num_heads, int num_kv_heads, int head_dim,
    float rms_norm_eps, float rope_theta, int rope_style
);

void cuda_free_layer_weights_fp16(void* weights);

// FP16 layer workspace — pre-allocated buffers for zero-alloc forward passes
int cuda_create_layer_workspace_fp16(
    void** workspace,
    int max_tokens,
    int hidden_size,
    int intermediate_size,
    int num_kv_heads,
    int head_dim
);

void cuda_free_layer_workspace_fp16(void* workspace);

int cuda_layer_forward_fp16_with_workspace(
    void* output, const void* input, const void* weights, void* kv_cache,
    const int* positions, int batch_size, int seq_len,
    int hidden_size, int intermediate_size, int num_heads, int num_kv_heads, int head_dim,
    float rms_norm_eps, float rope_theta, int rope_style,
    void* workspace
);

// FP16 layer forward with paged attention (block-based KV cache, CUDA Graph safe)
int cuda_layer_forward_fp16_paged(
    void* output, const void* input, const void* weights,
    void* paged_cache,           // PagedKVCache*
    const int* d_block_table,    // GPU block table
    const int* positions,        // GPU buffer [1]: position
    const int* d_seq_lens,       // GPU buffer [1]: seq_len = position + 1
    int batch_size, int seq_len,
    int hidden_size, int intermediate_size, int num_heads, int num_kv_heads, int head_dim,
    float rms_norm_eps, float rope_theta, int rope_style,
    void* workspace
);

// ============================================================================
// BF16-Native Layer (no FP16↔BF16 conversions for attention layers)
// ============================================================================

// Create BF16-native layer weights from BF16 host data (NO conversion)
int cuda_create_layer_weights_bf16_native(
    void** weights,
    const void* h_q_proj, const void* h_k_proj, const void* h_v_proj, const void* h_o_proj,
    const void* h_gate_proj, const void* h_up_proj, const void* h_down_proj,
    const void* h_attn_norm, const void* h_ffn_norm,
    const void* h_q_layernorm, const void* h_k_layernorm,
    int hidden_size, int intermediate_size, int num_heads, int num_kv_heads, int head_dim
);

void cuda_free_layer_weights_bf16_native(void* weights);

// BF16 layer workspace
int cuda_create_layer_workspace_bf16(
    void** workspace,
    int max_tokens,
    int hidden_size,
    int intermediate_size,
    int num_kv_heads,
    int head_dim
);

void cuda_free_layer_workspace_bf16(void* workspace);

// BF16-native layer forward pass (uses BF16 throughout, FP16 only for attention)
int cuda_layer_forward_bf16_native(
    void* output, const void* input, const void* weights,
    void* kv_cache, const int* positions, const int* d_seq_lens,
    int batch_size, int seq_len,
    int hidden_size, int intermediate_size, int num_heads, int num_kv_heads, int head_dim,
    float rms_norm_eps, float rope_theta, int rope_style,
    void* workspace
);

// BF16-native layer forward with paged attention
int cuda_layer_forward_bf16_paged(
    void* output, const void* input, const void* weights,
    void* paged_cache, const int* d_block_table,
    const int* positions, const int* d_seq_lens,
    int batch_size, int seq_len,
    int hidden_size, int intermediate_size, int num_heads, int num_kv_heads, int head_dim,
    float rms_norm_eps, float rope_theta, int rope_style,
    void* workspace
);

#ifdef __cplusplus
}
#endif

#endif // LAYER_H
