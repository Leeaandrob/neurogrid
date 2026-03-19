// stream.h — Global CUDA stream for kernel launches
// When CUDA Graph capture is active, all kernels must use the capture stream.
// This global variable is set before capture and cleared after.
#ifndef NEUROGRID_STREAM_H
#define NEUROGRID_STREAM_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

// Get the current execution stream (nullptr = default stream 0)
cudaStream_t ng_get_stream(void);

// Set the execution stream (call before graph capture, clear after)
void ng_set_stream(cudaStream_t stream);

#ifdef __cplusplus
}
#endif

// Macro for kernel launch with the global stream
// Use: NG_LAUNCH(kernel, grid, block, shared)(args...)
// Expands to: kernel<<<grid, block, shared, ng_get_stream()>>>(args...)
#define NG_LAUNCH(kernel, grid, block, shared) kernel<<<grid, block, shared, ng_get_stream()>>>
#define NG_LAUNCH_SIMPLE(kernel, grid, block) kernel<<<grid, block, 0, ng_get_stream()>>>

#endif // NEUROGRID_STREAM_H
