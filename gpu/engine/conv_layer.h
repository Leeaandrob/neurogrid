// conv_layer.h - LIV Conv layer forward pass for LFM2
#ifndef CONV_LAYER_H
#define CONV_LAYER_H

#ifdef __cplusplus
extern "C" {
#endif

// Conv layer weights structure (all BF16 on GPU)
typedef struct {
    void* in_proj_weight;   // [3*hidden, hidden] BF16
    void* conv_weight;      // [hidden, kernel_size] BF16
    void* out_proj_weight;  // [hidden, hidden] BF16
    void* operator_norm;    // [hidden] BF16 RMSNorm weight
    void* ffn_norm;         // [hidden] BF16 RMSNorm weight
    void* gate_weight;      // [intermediate, hidden] BF16
    void* up_weight;        // [intermediate, hidden] BF16
    void* down_weight;      // [hidden, intermediate] BF16
    int hidden_size;
    int intermediate_size;
    int conv_kernel_size;
    float norm_eps;
} ConvLayerWeights;

// Create conv layer weights from host BF16 data
// All pointers are host BF16 data that will be copied to GPU
int cuda_create_conv_layer_weights_bf16(
    void** weights_out,
    const void* h_in_proj,     // [3*hidden, hidden] BF16
    const void* h_conv,        // [hidden, kernel_size] BF16
    const void* h_out_proj,    // [hidden, hidden] BF16
    const void* h_op_norm,     // [hidden] BF16
    const void* h_ffn_norm,    // [hidden] BF16
    const void* h_gate,        // [intermediate, hidden] BF16
    const void* h_up,          // [intermediate, hidden] BF16
    const void* h_down,        // [hidden, intermediate] BF16
    int hidden, int intermediate, int kernel_size, float norm_eps
);

// Forward pass: RMSNorm -> LIV Conv -> Residual -> RMSNorm -> SwiGLU FFN -> Residual
// Input/Output: [batch, seq_len, hidden] BF16
// Conv state:   [batch, hidden, kernel_size] FP32 (updated in-place)
int cuda_conv_layer_forward_bf16(
    void* output,
    const void* input,
    const void* weights,
    void* conv_state,
    int batch, int seq_len, int position
);

// Free conv layer weights
void cuda_free_conv_layer_weights(void* weights);

#ifdef __cplusplus
}
#endif

#endif // CONV_LAYER_H
