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
    float rms_norm_eps
);

#ifdef __cplusplus
}
#endif

#endif // LAYER_H
