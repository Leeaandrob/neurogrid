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
void cuda_set_decode_paged_cache(void* ctx, void* paged_cache, int* d_block_table, int max_blocks_per_seq);

int cuda_decode_step(void* ctx, void* h_output, const void* h_input, int position);

// GPU-resident decode: hidden stays on GPU between tokens
int cuda_decode_set_hidden(void* ctx, const void* h_hidden);
int cuda_decode_set_hidden_from_gpu(void* ctx, const void* d_hidden);
int cuda_decode_get_hidden(void* ctx, void* h_hidden);
int cuda_decode_step_gpu(void* ctx, int position);
void* cuda_decode_get_hidden_gpu_ptr(void* ctx);

void cuda_set_decode_paged_layer(void* ctx, int layer_id, void* paged_cache);
void cuda_free_decode_context(void* ctx);

#ifdef __cplusplus
}
#endif

#endif // DECODE_ALL_H
