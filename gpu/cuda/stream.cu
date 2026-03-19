// stream.cu — Global stream management for CUDA Graph capture
#include "stream.h"

static cudaStream_t g_ng_stream = nullptr; // nullptr = default stream 0

extern "C" cudaStream_t ng_get_stream(void) {
    return g_ng_stream;
}

extern "C" void ng_set_stream(cudaStream_t stream) {
    g_ng_stream = stream;
}
