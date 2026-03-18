// decode_all.h — Full decode step: all layers in a single CUDA call
#ifndef DECODE_ALL_H
#define DECODE_ALL_H

#ifdef __cplusplus
extern "C" {
#endif

int cuda_create_decode_context(
    void** ctx_out,
    int num_layers, int hidden_size, int intermediate_size,
    int num_heads, int num_kv_heads, int head_dim,
    float norm_eps, float rope_theta, int rope_style, int conv_kernel_size);

void cuda_set_decode_layer(void* ctx, int layer_id, int layer_type, void* weights, void* cache);
void cuda_set_decode_workspace(void* ctx, void* workspace);

int cuda_decode_step(void* ctx, void* h_output, const void* h_input, int position);

void cuda_free_decode_context(void* ctx);

#ifdef __cplusplus
}
#endif

#endif // DECODE_ALL_H
