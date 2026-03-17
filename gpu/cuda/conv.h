// conv.h - Depthwise causal conv1d CUDA kernels for LFM2
#ifndef CONV_H
#define CONV_H

#ifdef __cplusplus
extern "C" {
#endif

// Prefill: process full sequence through depthwise causal conv1d
// Input x: [batch, dim, seqlen] BF16
// Weight:  [dim, width] BF16
// Output:  [batch, dim, seqlen] BF16
// Conv state: [batch, dim, width] FP32 (updated in-place with final state)
int cuda_causal_conv1d_fwd_bf16(
    const void* x,
    const void* weight,
    void* out,
    void* conv_state,
    int batch, int dim, int seqlen, int width
);

// Decode: single token update with state management
// Input x: [batch, dim] BF16 single token
// Output:  [batch, dim] BF16
// Conv state: [batch, dim, width] FP32 (shift left, insert new, compute output)
int cuda_causal_conv1d_update_bf16(
    const void* x,
    void* out,
    void* conv_state,
    const void* weight,
    int batch, int dim, int width
);

// Conv state management
void* cuda_conv_state_create(int batch, int dim, int width);
int cuda_conv_state_reset(void* state, int batch, int dim, int width);
void cuda_conv_state_free(void* state);

#ifdef __cplusplus
}
#endif

#endif // CONV_H
