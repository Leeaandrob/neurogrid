// multi_device.cu - Multi-GPU device management for NeuroGrid
// Implements multi-device context management, P2P access, and cross-device memory operations.
// Follows CUDA multi-GPU best practices from the programming guide.

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "multi_device.h"

// Error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        return -1; \
    } \
} while(0)

#define CUDA_CHECK_VOID(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        return; \
    } \
} while(0)

// ============================================================================
// Global Multi-Device State
// ============================================================================

static struct {
    bool initialized;
    int num_devices;
    int device_ids[MAX_GPU_DEVICES];
    DeviceContext contexts[MAX_GPU_DEVICES];
    bool p2p_matrix[MAX_GPU_DEVICES][MAX_GPU_DEVICES];

    // Staging buffer for non-P2P transfers
    void* staging_buffer;
    size_t staging_buffer_size;

} g_multi_device = {
    .initialized = false,
    .num_devices = 0,
    .staging_buffer = nullptr,
    .staging_buffer_size = 0
};

// Default staging buffer size (64 MB)
static const size_t DEFAULT_STAGING_BUFFER_SIZE = 64 * 1024 * 1024;

// ============================================================================
// Multi-Device Initialization
// ============================================================================

extern "C" int cuda_multi_init(int* device_ids, int num_devices) {
    if (g_multi_device.initialized) {
        fprintf(stderr, "Multi-device context already initialized\n");
        return -1;
    }

    if (num_devices <= 0 || num_devices > MAX_GPU_DEVICES) {
        fprintf(stderr, "Invalid number of devices: %d (max: %d)\n",
                num_devices, MAX_GPU_DEVICES);
        return -1;
    }

    // Verify all requested devices exist
    int available_devices;
    CUDA_CHECK(cudaGetDeviceCount(&available_devices));

    for (int i = 0; i < num_devices; i++) {
        if (device_ids[i] < 0 || device_ids[i] >= available_devices) {
            fprintf(stderr, "Invalid device ID %d (available: 0-%d)\n",
                    device_ids[i], available_devices - 1);
            return -1;
        }
        g_multi_device.device_ids[i] = device_ids[i];
    }

    g_multi_device.num_devices = num_devices;

    // Initialize each device context
    for (int i = 0; i < num_devices; i++) {
        int dev_id = device_ids[i];
        DeviceContext* ctx = &g_multi_device.contexts[i];

        CUDA_CHECK(cudaSetDevice(dev_id));

        // Get device properties
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev_id));

        ctx->device_id = dev_id;
        ctx->total_memory = prop.totalGlobalMem;

        // Get current memory usage
        size_t free_mem, total_mem;
        CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
        ctx->used_memory = total_mem - free_mem;

        // Create dedicated streams for compute and transfer
        CUDA_CHECK(cudaStreamCreate((cudaStream_t*)&ctx->compute_stream));
        CUDA_CHECK(cudaStreamCreate((cudaStream_t*)&ctx->transfer_stream));

        // Initialize peer access array
        memset(ctx->peer_access, 0, sizeof(ctx->peer_access));
    }

    // Setup P2P access between devices
    for (int i = 0; i < num_devices; i++) {
        for (int j = 0; j < num_devices; j++) {
            g_multi_device.p2p_matrix[i][j] = false;

            if (i == j) continue; // Skip self

            int src_dev = device_ids[i];
            int dst_dev = device_ids[j];

            int can_access = 0;
            cudaError_t err = cudaDeviceCanAccessPeer(&can_access, src_dev, dst_dev);

            if (err == cudaSuccess && can_access) {
                // Enable P2P access
                CUDA_CHECK(cudaSetDevice(src_dev));
                err = cudaDeviceEnablePeerAccess(dst_dev, 0);

                if (err == cudaSuccess || err == cudaErrorPeerAccessAlreadyEnabled) {
                    g_multi_device.p2p_matrix[i][j] = true;
                    g_multi_device.contexts[i].peer_access[j] = true;
                }
            }
        }
    }

    // Allocate staging buffer on first device for non-P2P transfers
    CUDA_CHECK(cudaSetDevice(device_ids[0]));
    g_multi_device.staging_buffer_size = DEFAULT_STAGING_BUFFER_SIZE;
    CUDA_CHECK(cudaMallocHost(&g_multi_device.staging_buffer,
                              g_multi_device.staging_buffer_size));

    g_multi_device.initialized = true;

    // Reset to first device
    CUDA_CHECK(cudaSetDevice(device_ids[0]));

    return 0;
}

extern "C" void cuda_multi_shutdown() {
    if (!g_multi_device.initialized) {
        return;
    }

    // Disable P2P access
    for (int i = 0; i < g_multi_device.num_devices; i++) {
        int src_dev = g_multi_device.device_ids[i];
        cudaSetDevice(src_dev);

        for (int j = 0; j < g_multi_device.num_devices; j++) {
            if (g_multi_device.p2p_matrix[i][j]) {
                int dst_dev = g_multi_device.device_ids[j];
                cudaDeviceDisablePeerAccess(dst_dev);
            }
        }
    }

    // Destroy streams
    for (int i = 0; i < g_multi_device.num_devices; i++) {
        DeviceContext* ctx = &g_multi_device.contexts[i];
        cudaSetDevice(ctx->device_id);

        if (ctx->compute_stream) {
            cudaStreamDestroy((cudaStream_t)ctx->compute_stream);
            ctx->compute_stream = nullptr;
        }
        if (ctx->transfer_stream) {
            cudaStreamDestroy((cudaStream_t)ctx->transfer_stream);
            ctx->transfer_stream = nullptr;
        }
    }

    // Free staging buffer
    if (g_multi_device.staging_buffer) {
        cudaFreeHost(g_multi_device.staging_buffer);
        g_multi_device.staging_buffer = nullptr;
    }

    // Reset state
    g_multi_device.initialized = false;
    g_multi_device.num_devices = 0;
    g_multi_device.staging_buffer_size = 0;
}

// ============================================================================
// Device Context Access
// ============================================================================

extern "C" int cuda_get_device_context(int device_id, DeviceContext* ctx) {
    if (!g_multi_device.initialized) {
        fprintf(stderr, "Multi-device context not initialized\n");
        return -1;
    }

    // Find device index
    int dev_idx = -1;
    for (int i = 0; i < g_multi_device.num_devices; i++) {
        if (g_multi_device.device_ids[i] == device_id) {
            dev_idx = i;
            break;
        }
    }

    if (dev_idx < 0) {
        fprintf(stderr, "Device %d not in multi-device context\n", device_id);
        return -1;
    }

    // Update memory usage
    DeviceContext* src = &g_multi_device.contexts[dev_idx];
    CUDA_CHECK(cudaSetDevice(device_id));

    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    src->used_memory = total_mem - free_mem;

    // Copy context
    memcpy(ctx, src, sizeof(DeviceContext));

    return 0;
}

extern "C" void cuda_multi_get_num_devices(int* count) {
    *count = g_multi_device.initialized ? g_multi_device.num_devices : 0;
}

extern "C" void cuda_multi_get_staging_buffer_size(size_t* size) {
    *size = g_multi_device.staging_buffer_size;
}

extern "C" int cuda_multi_get_info(MultiDeviceManagerInfo* info) {
    if (!info) {
        return -1;
    }

    info->initialized = g_multi_device.initialized;
    info->num_devices = g_multi_device.num_devices;
    info->staging_buffer_size = g_multi_device.staging_buffer_size;

    memcpy(info->device_ids, g_multi_device.device_ids,
           sizeof(int) * g_multi_device.num_devices);

    return 0;
}

// ============================================================================
// Peer-to-Peer Access
// ============================================================================

extern "C" int cuda_can_access_peer(int src_device, int dst_device, int* can_access) {
    if (!g_multi_device.initialized) {
        fprintf(stderr, "Multi-device context not initialized\n");
        return -1;
    }

    // Find device indices
    int src_idx = -1, dst_idx = -1;
    for (int i = 0; i < g_multi_device.num_devices; i++) {
        if (g_multi_device.device_ids[i] == src_device) src_idx = i;
        if (g_multi_device.device_ids[i] == dst_device) dst_idx = i;
    }

    if (src_idx < 0 || dst_idx < 0) {
        fprintf(stderr, "Device not in multi-device context\n");
        return -1;
    }

    *can_access = g_multi_device.p2p_matrix[src_idx][dst_idx] ? 1 : 0;
    return 0;
}

// ============================================================================
// Device-Specific Memory Operations
// ============================================================================

extern "C" int cuda_alloc_on_device(void** ptr, size_t size, int device_id) {
    if (!g_multi_device.initialized) {
        fprintf(stderr, "Multi-device context not initialized\n");
        return -1;
    }

    if (size == 0) {
        *ptr = nullptr;
        return 0;
    }

    // Verify device is in our context
    bool found = false;
    for (int i = 0; i < g_multi_device.num_devices; i++) {
        if (g_multi_device.device_ids[i] == device_id) {
            found = true;
            break;
        }
    }

    if (!found) {
        fprintf(stderr, "Device %d not in multi-device context\n", device_id);
        return -1;
    }

    CUDA_CHECK(cudaSetDevice(device_id));
    CUDA_CHECK(cudaMalloc(ptr, size));

    return 0;
}

extern "C" int cuda_free_on_device(void* ptr, int device_id) {
    if (ptr == nullptr) {
        return 0;
    }

    if (!g_multi_device.initialized) {
        fprintf(stderr, "Multi-device context not initialized\n");
        return -1;
    }

    CUDA_CHECK(cudaSetDevice(device_id));
    CUDA_CHECK(cudaFree(ptr));

    return 0;
}

// ============================================================================
// Cross-Device Memory Copy
// ============================================================================

extern "C" int cuda_cross_device_copy(
    void* dst,
    int dst_device,
    void* src,
    int src_device,
    size_t size,
    void* stream
) {
    if (!g_multi_device.initialized) {
        fprintf(stderr, "Multi-device context not initialized\n");
        return -1;
    }

    if (size == 0) {
        return 0;
    }

    // Find device indices
    int src_idx = -1, dst_idx = -1;
    for (int i = 0; i < g_multi_device.num_devices; i++) {
        if (g_multi_device.device_ids[i] == src_device) src_idx = i;
        if (g_multi_device.device_ids[i] == dst_device) dst_idx = i;
    }

    if (src_idx < 0 || dst_idx < 0) {
        fprintf(stderr, "Device not in multi-device context\n");
        return -1;
    }

    // Same device - simple copy
    if (src_device == dst_device) {
        CUDA_CHECK(cudaSetDevice(src_device));
        if (stream) {
            CUDA_CHECK(cudaMemcpyAsync(dst, src, size,
                                       cudaMemcpyDeviceToDevice,
                                       (cudaStream_t)stream));
        } else {
            CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
        }
        return 0;
    }

    // P2P available - direct copy
    if (g_multi_device.p2p_matrix[src_idx][dst_idx]) {
        CUDA_CHECK(cudaSetDevice(src_device));
        if (stream) {
            CUDA_CHECK(cudaMemcpyPeerAsync(dst, dst_device, src, src_device,
                                          size, (cudaStream_t)stream));
        } else {
            CUDA_CHECK(cudaMemcpyPeer(dst, dst_device, src, src_device, size));
        }
        return 0;
    }

    // No P2P - stage through host
    // For large transfers, use multiple chunks
    size_t chunk_size = g_multi_device.staging_buffer_size;
    size_t offset = 0;

    while (offset < size) {
        size_t transfer_size = (size - offset < chunk_size) ?
                               (size - offset) : chunk_size;

        // Copy to staging buffer from source
        CUDA_CHECK(cudaSetDevice(src_device));
        CUDA_CHECK(cudaMemcpy(g_multi_device.staging_buffer,
                              (char*)src + offset,
                              transfer_size,
                              cudaMemcpyDeviceToHost));

        // Copy from staging buffer to destination
        CUDA_CHECK(cudaSetDevice(dst_device));
        CUDA_CHECK(cudaMemcpy((char*)dst + offset,
                              g_multi_device.staging_buffer,
                              transfer_size,
                              cudaMemcpyHostToDevice));

        offset += transfer_size;
    }

    return 0;
}

// ============================================================================
// Raw Memory Copy (Host <-> Device)
// ============================================================================

extern "C" int cuda_copy_to_device_raw(void* dst, const void* src, size_t size) {
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
    return 0;
}

extern "C" int cuda_copy_from_device_raw(void* dst, const void* src, size_t size) {
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
    return 0;
}
