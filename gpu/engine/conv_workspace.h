// conv_workspace.h — Pre-allocated workspace for conv layer (CUDA Graph safe)
#ifndef CONV_WORKSPACE_H
#define CONV_WORKSPACE_H

#ifdef __cplusplus
extern "C" {
#endif

int cuda_create_conv_workspace(void** ws, int hidden_size, int intermediate_size);
void cuda_free_conv_workspace(void* ws);

// Conv layer forward with pre-allocated workspace (no cudaMalloc during execution)
int cuda_conv_layer_forward_bf16_ws(
    void* output, const void* input, const void* weights,
    void* conv_state, int batch, int seq_len, int position, void* workspace);

#ifdef __cplusplus
}
#endif

#endif
