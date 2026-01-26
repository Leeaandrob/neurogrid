// gpu.h - C header for CUDA bindings
// This file defines the interface between Go (CGO) and CUDA code.

#ifndef GPU_H
#define GPU_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Device info structure
typedef struct {
    char name[256];
    int major;
    int minor;
    size_t total_memory;
} DeviceInfo;

// ============================================================================
// Device Management
// ============================================================================

// Initialize CUDA for the specified device. Returns 0 on success.
int cuda_init(int device_id);

// Shutdown CUDA and release resources.
void cuda_shutdown(void);

// Get the number of available CUDA devices.
int cuda_get_device_count(int* count);

// Set the current CUDA device.
int cuda_set_device(int device_id);

// Get information about the current device.
int cuda_get_device_info(DeviceInfo* info);

// Synchronize the current device.
int cuda_sync_device(void);

// Get current GPU memory usage.
int cuda_get_memory_used(size_t* used);

// ============================================================================
// Memory Management
// ============================================================================

// Allocate GPU memory.
int cuda_malloc(void** ptr, size_t size);

// Free GPU memory.
void cuda_free(void* ptr);

// Copy FP32 data from host to device (with optional FP16 conversion).
// dtype: 0=FP32, 1=FP16, 2=INT8
int cuda_copy_to_device(void* dst, const void* src, size_t num_elements, int dtype);

// Copy data from device to host FP32.
int cuda_copy_to_host(void* dst, const void* src, size_t num_elements, int dtype);

// Copy INT8 data to device.
int cuda_copy_int8_to_device(void* dst, const void* src, size_t num_elements);

// Copy INT8 data from device.
int cuda_copy_int8_to_host(void* dst, const void* src, size_t num_elements);

// ============================================================================
// Basic Kernels
// ============================================================================

// RMSNorm: output = x * rsqrt(mean(x^2) + eps) * weight
int cuda_rmsnorm(
    void* output,
    const void* input,
    const void* weight,
    int num_tokens,
    int hidden_dim,
    float eps
);

// SiLU activation: output = x * sigmoid(x)
int cuda_silu(void* output, const void* input, size_t num_elements);

// Element-wise addition: c = a + b
int cuda_add(void* c, const void* a, const void* b, size_t num_elements);

// Rotary Position Embeddings
int cuda_rope(
    void* output,
    const void* input,
    const int* positions,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim
);

// Rotary Position Embeddings with configurable theta
int cuda_rope_with_theta(
    void* output,
    const void* input,
    const int* positions,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_dim,
    int rope_style,
    float rope_theta
);

// ============================================================================
// Matrix Multiplication
// ============================================================================

// FP16 GEMM: C = A @ B (with optional transposes)
int cuda_gemm_fp16(
    void* c,
    const void* a,
    const void* b,
    int M,
    int K,
    int N,
    bool transpose_a,
    bool transpose_b
);

// INT8 GEMM with dequantization: C = A @ (B * scale)
// When transpose_b=true, B is stored as [N, K] and will be transposed
int cuda_gemm_int8(
    void* c,
    const void* a,        // FP16 [M, K]
    const void* b,        // INT8 [K, N] or [N, K] if transpose_b
    const void* scale,    // FP32 [N]
    int M,
    int K,
    int N,
    bool transpose_b      // If true, B is stored as [N, K]
);

// ============================================================================
// Quantization
// ============================================================================

// Quantize FP16 tensor to INT8 (per-tensor).
int cuda_quantize_per_tensor(
    void* output,         // INT8 output
    void* scale,          // FP32 scale [1]
    const void* input,    // FP16 input
    size_t num_elements
);

// Dequantize INT8 to FP16.
int cuda_dequantize_per_tensor(
    void* output,         // FP16 output
    const void* input,    // INT8 input
    const void* scale,    // FP32 scale [1]
    size_t num_elements
);

// Quantize weights per-column.
int cuda_quantize_weights(
    void* output,         // INT8 [K, N]
    void* scales,         // FP32 [N]
    const void* input,    // FP16 [K, N]
    int K,
    int N
);

// Quantize weights per-row (for transpose_b=true GEMM).
// Use this when weights are stored as [out_dim, in_dim] and will be transposed.
int cuda_quantize_weights_per_row(
    void* output,         // INT8 [rows, cols]
    void* scales,         // FP32 [rows] - one scale per row (per output channel)
    const void* input,    // FP16 [rows, cols]
    int rows,
    int cols
);

// ============================================================================
// Attention
// ============================================================================

// Basic attention (no FlashAttention optimization).
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

// ============================================================================
// KV Cache
// ============================================================================

// Create a new KV cache.
int cuda_kvcache_create(
    void** cache,
    int batch_size,
    int num_heads,
    int head_dim,
    int max_seq_len
);

// Free KV cache.
void cuda_kvcache_free(void* cache);

// Update KV cache with new K, V at position.
int cuda_kvcache_update(
    void* cache,
    const void* k,
    const void* v,
    int position
);

// Get current length of KV cache.
int cuda_kvcache_get_length(void* cache);

// Attention using KV cache.
int cuda_attention_with_kvcache(
    void* output,
    const void* query,
    const void* new_key,
    const void* new_value,
    void* cache,
    int batch_size,
    int num_heads,
    int head_dim,
    int position
);

// ============================================================================
// Multi-Device Management
// ============================================================================

// Maximum number of GPUs supported
#define MAX_GPU_DEVICES 8

// Device context structure for multi-GPU
typedef struct {
    int device_id;
    size_t total_memory;
    size_t used_memory;
    void* compute_stream;
    void* transfer_stream;
    bool peer_access[MAX_GPU_DEVICES];
} DeviceContext;

// Multi-device manager information
typedef struct {
    int num_devices;
    int device_ids[MAX_GPU_DEVICES];
    size_t staging_buffer_size;
    bool initialized;
} MultiDeviceManagerInfo;

// Initialize multi-GPU context with specified devices.
// device_ids: array of device IDs to use
// num_devices: number of devices
// Returns 0 on success.
int cuda_multi_init(int* device_ids, int num_devices);

// Shutdown multi-GPU context and release resources.
void cuda_multi_shutdown(void);

// Get device context for a specific device.
// Returns 0 on success, populates ctx.
int cuda_get_device_context(int device_id, DeviceContext* ctx);

// Get the number of devices in multi-GPU context.
void cuda_multi_get_num_devices(int* count);

// Get the staging buffer size used for cross-device transfers.
void cuda_multi_get_staging_buffer_size(size_t* size);

// Get multi-device manager information.
int cuda_multi_get_info(MultiDeviceManagerInfo* info);

// Check if peer-to-peer access is possible between two devices.
// can_access: output, 1 if accessible, 0 otherwise
int cuda_can_access_peer(int src_device, int dst_device, int* can_access);

// Allocate GPU memory on a specific device.
int cuda_alloc_on_device(void** ptr, size_t size, int device_id);

// Free GPU memory on a specific device.
int cuda_free_on_device(void* ptr, int device_id);

// Copy data between two GPU devices.
// Uses P2P if available, otherwise stages through host.
// stream: optional CUDA stream (can be NULL for synchronous)
int cuda_cross_device_copy(
    void* dst,
    int dst_device,
    void* src,
    int src_device,
    size_t size,
    void* stream
);

// ============================================================================
// Weight Loading
// ============================================================================

// Load layer weights from directory (safetensors format).
int cuda_load_layer_weights(void** weights, const char* path);

// Free layer weights.
void cuda_free_layer_weights(void* weights);

// Create random layer weights (for testing).
int cuda_create_random_layer_weights(
    void** weights,
    int hidden_size,
    int intermediate_size,
    int num_heads,
    int num_kv_heads,
    int head_dim
);

// Create layer weights from host FP16 data.
// All weight pointers are FP16 host memory, will be quantized to INT8 on GPU.
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

// ============================================================================
// Layer Forward
// ============================================================================

// Complete transformer layer forward pass.
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

#ifdef __cplusplus
}
#endif

#endif // GPU_H
