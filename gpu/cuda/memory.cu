// memory.cu - CUDA memory management for NeuroGrid
// Provides memory allocation, deallocation, and data transfer utilities.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "memory.h"
#include "matmul.h"

// Error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        return -1; \
    } \
} while(0)

// ============================================================================
// Device Management
// ============================================================================

static bool g_initialized = false;
static int g_current_device = 0;

extern "C" int cuda_init(int device_id) {
    if (g_initialized) {
        return 0;
    }

    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    if (device_id < 0 || device_id >= device_count) {
        fprintf(stderr, "Invalid device ID %d (available: 0-%d)\n",
                device_id, device_count - 1);
        return -1;
    }

    CUDA_CHECK(cudaSetDevice(device_id));
    g_current_device = device_id;
    g_initialized = true;

    // Warm up the GPU context
    void* dummy;
    CUDA_CHECK(cudaMalloc(&dummy, 1024));
    CUDA_CHECK(cudaFree(dummy));

    return 0;
}

extern "C" void cuda_shutdown() {
    if (g_initialized) {
        // Shutdown cuBLAS before device reset
        cublas_shutdown();
        // Sync and clear any pending errors before reset
        cudaDeviceSynchronize();
        cudaGetLastError();  // Clear any pending error
        cudaDeviceReset();
        g_initialized = false;
    }
}

extern "C" int cuda_get_device_count(int* count) {
    CUDA_CHECK(cudaGetDeviceCount(count));
    return 0;
}

extern "C" int cuda_set_device(int device_id) {
    CUDA_CHECK(cudaSetDevice(device_id));
    g_current_device = device_id;
    return 0;
}

extern "C" int cuda_get_device_info(DeviceInfo* info) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, g_current_device));

    strncpy(info->name, prop.name, 255);
    info->name[255] = '\0';
    info->major = prop.major;
    info->minor = prop.minor;
    info->total_memory = prop.totalGlobalMem;

    return 0;
}

extern "C" int cuda_sync_device() {
    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
}

extern "C" int cuda_get_memory_used(size_t* used) {
    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    *used = total_mem - free_mem;
    return 0;
}

// ============================================================================
// Memory Allocation
// ============================================================================

extern "C" int cuda_malloc(void** ptr, size_t size) {
    if (size == 0) {
        *ptr = nullptr;
        return 0;
    }
    CUDA_CHECK(cudaMalloc(ptr, size));
    return 0;
}

extern "C" void cuda_free(void* ptr) {
    if (ptr != nullptr) {
        cudaFree(ptr);
    }
}

// ============================================================================
// Data Transfer
// ============================================================================

// Kernel to convert FP32 to FP16
__global__ void fp32_to_fp16_kernel(half* dst, const float* src, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __float2half(src[idx]);
    }
}

// Kernel to convert FP16 to FP32
__global__ void fp16_to_fp32_kernel(float* dst, const half* src, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = __half2float(src[idx]);
    }
}

extern "C" int cuda_copy_to_device(void* dst, const void* src, size_t num_elements, int dtype) {
    // dtype: 0=FP32, 1=FP16, 2=INT8

    if (dtype == 0) {
        // FP32 -> FP32 direct copy
        CUDA_CHECK(cudaMemcpy(dst, src, num_elements * sizeof(float),
                              cudaMemcpyHostToDevice));
    }
    else if (dtype == 1) {
        // FP32 (host) -> FP16 (device)
        // Allocate temporary device buffer for FP32
        float* d_temp;
        CUDA_CHECK(cudaMalloc(&d_temp, num_elements * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_temp, src, num_elements * sizeof(float),
                              cudaMemcpyHostToDevice));

        // Convert FP32 to FP16 on device
        int block_size = 256;
        int num_blocks = (num_elements + block_size - 1) / block_size;
        fp32_to_fp16_kernel<<<num_blocks, block_size>>>((half*)dst, d_temp, num_elements);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaFree(d_temp));
    }
    else if (dtype == 2) {
        // INT8 direct copy
        CUDA_CHECK(cudaMemcpy(dst, src, num_elements * sizeof(int8_t),
                              cudaMemcpyHostToDevice));
    }
    else {
        return -1;
    }

    return 0;
}

extern "C" int cuda_copy_to_host(void* dst, const void* src, size_t num_elements, int dtype) {
    if (dtype == 0) {
        // FP32 -> FP32 direct copy
        CUDA_CHECK(cudaMemcpy(dst, src, num_elements * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }
    else if (dtype == 1) {
        // FP16 (device) -> FP32 (host)
        // Allocate temporary device buffer for FP32
        float* d_temp;
        CUDA_CHECK(cudaMalloc(&d_temp, num_elements * sizeof(float)));

        // Convert FP16 to FP32 on device
        int block_size = 256;
        int num_blocks = (num_elements + block_size - 1) / block_size;
        fp16_to_fp32_kernel<<<num_blocks, block_size>>>(d_temp, (const half*)src, num_elements);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(dst, d_temp, num_elements * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_temp));
    }
    else if (dtype == 2) {
        // INT8 direct copy
        CUDA_CHECK(cudaMemcpy(dst, src, num_elements * sizeof(int8_t),
                              cudaMemcpyDeviceToHost));
    }
    else {
        return -1;
    }

    return 0;
}

extern "C" int cuda_copy_int8_to_device(void* dst, const void* src, size_t num_elements) {
    CUDA_CHECK(cudaMemcpy(dst, src, num_elements * sizeof(int8_t), cudaMemcpyHostToDevice));
    return 0;
}

extern "C" int cuda_copy_int8_to_host(void* dst, const void* src, size_t num_elements) {
    CUDA_CHECK(cudaMemcpy(dst, src, num_elements * sizeof(int8_t), cudaMemcpyDeviceToHost));
    return 0;
}

// ============================================================================
// Memory Pool (Optional Optimization)
// ============================================================================

// Simple memory pool for frequently allocated sizes
// This reduces cudaMalloc overhead for repeated allocations

struct MemoryPool {
    static constexpr size_t NUM_BUCKETS = 16;
    static constexpr size_t MAX_CACHED = 4;

    struct Bucket {
        void* ptrs[MAX_CACHED];
        size_t count;
        size_t size;
    };

    Bucket buckets[NUM_BUCKETS];
    bool initialized;

    MemoryPool() : initialized(false) {
        memset(buckets, 0, sizeof(buckets));
    }

    ~MemoryPool() {
        clear();
    }

    void clear() {
        for (int i = 0; i < NUM_BUCKETS; i++) {
            for (size_t j = 0; j < buckets[i].count; j++) {
                cudaFree(buckets[i].ptrs[j]);
            }
            buckets[i].count = 0;
        }
    }

    int bucket_index(size_t size) {
        // Find bucket for sizes: 256B, 512B, 1K, 2K, ..., 8M
        size_t bucket_size = 256;
        for (int i = 0; i < NUM_BUCKETS; i++) {
            if (size <= bucket_size) return i;
            bucket_size *= 2;
        }
        return -1; // Too large for pool
    }

    void* allocate(size_t size) {
        int idx = bucket_index(size);
        if (idx < 0) {
            // Too large, use direct allocation
            void* ptr;
            if (cudaMalloc(&ptr, size) != cudaSuccess) return nullptr;
            return ptr;
        }

        Bucket& bucket = buckets[idx];
        if (bucket.count > 0) {
            // Reuse cached pointer
            return bucket.ptrs[--bucket.count];
        }

        // Allocate new
        size_t alloc_size = 256 << idx;
        void* ptr;
        if (cudaMalloc(&ptr, alloc_size) != cudaSuccess) return nullptr;
        return ptr;
    }

    void free(void* ptr, size_t size) {
        int idx = bucket_index(size);
        if (idx < 0 || buckets[idx].count >= MAX_CACHED) {
            // Too large or cache full, direct free
            cudaFree(ptr);
            return;
        }

        // Cache for reuse
        buckets[idx].ptrs[buckets[idx].count++] = ptr;
    }
};

// Global memory pool (disabled by default, can be enabled for optimization)
// static MemoryPool g_pool;
