// memory.h - CUDA memory management header
#ifndef MEMORY_H
#define MEMORY_H

#include <stddef.h>

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

// Device management
int cuda_init(int device_id);
void cuda_shutdown(void);
int cuda_get_device_count(int* count);
int cuda_set_device(int device_id);
int cuda_get_device_info(DeviceInfo* info);
int cuda_sync_device(void);
int cuda_get_memory_used(size_t* used);

// Memory allocation
int cuda_malloc(void** ptr, size_t size);
void cuda_free(void* ptr);

// Data transfer
int cuda_copy_to_device(void* dst, const void* src, size_t num_elements, int dtype);
int cuda_copy_to_host(void* dst, const void* src, size_t num_elements, int dtype);
int cuda_copy_int8_to_device(void* dst, const void* src, size_t num_elements);
int cuda_copy_int8_to_host(void* dst, const void* src, size_t num_elements);

#ifdef __cplusplus
}
#endif

#endif // MEMORY_H
