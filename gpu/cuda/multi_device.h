// multi_device.h - Multi-GPU device management header for NeuroGrid
// Provides structures and function declarations for multi-device context management.

#ifndef MULTI_DEVICE_H
#define MULTI_DEVICE_H

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

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

// ============================================================================
// Multi-Device Initialization
// ============================================================================

// Initialize multi-GPU context with specified devices.
// device_ids: array of device IDs to use
// num_devices: number of devices
// Returns 0 on success.
int cuda_multi_init(int* device_ids, int num_devices);

// Shutdown multi-GPU context and release resources.
void cuda_multi_shutdown(void);

// ============================================================================
// Device Context Access
// ============================================================================

// Get device context for a specific device.
// Returns 0 on success, populates ctx.
int cuda_get_device_context(int device_id, DeviceContext* ctx);

// Get the number of devices in multi-GPU context.
void cuda_multi_get_num_devices(int* count);

// Get the staging buffer size used for cross-device transfers.
void cuda_multi_get_staging_buffer_size(size_t* size);

// Get multi-device manager information.
int cuda_multi_get_info(MultiDeviceManagerInfo* info);

// ============================================================================
// Peer-to-Peer Access
// ============================================================================

// Check if peer-to-peer access is possible between two devices.
// can_access: output, 1 if accessible, 0 otherwise
int cuda_can_access_peer(int src_device, int dst_device, int* can_access);

// ============================================================================
// Device-Specific Memory Operations
// ============================================================================

// Allocate GPU memory on a specific device.
int cuda_alloc_on_device(void** ptr, size_t size, int device_id);

// Free GPU memory on a specific device.
int cuda_free_on_device(void* ptr, int device_id);

// ============================================================================
// Cross-Device Memory Copy
// ============================================================================

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
// Raw Memory Copy (Host <-> Device)
// ============================================================================

// Copy raw bytes from host to device.
int cuda_copy_to_device_raw(void* dst, const void* src, size_t size);

// Copy raw bytes from device to host.
int cuda_copy_from_device_raw(void* dst, const void* src, size_t size);

#ifdef __cplusplus
}
#endif

#endif // MULTI_DEVICE_H
