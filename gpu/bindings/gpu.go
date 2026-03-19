//go:build cuda

// Package bindings provides CGO bindings for CUDA operations.
// This is the interface between Go and the CUDA backend.
package bindings

/*
#cgo CFLAGS: -I/usr/local/cuda/include
#cgo LDFLAGS: -L/usr/local/cuda/lib64 -lcudart -lcublas -lcublasLt -lgpu_engine

#include <stdlib.h>
#include <cuda_runtime.h>
#include "gpu.h"

// cuda_host_alloc allocates CUDA-registered host (pinned) memory.
// Uses cudaHostAllocPortable for multi-GPU visibility.
static inline cudaError_t cuda_host_alloc(void** ptr, size_t size) {
    return cudaHostAlloc(ptr, size, cudaHostAllocPortable);
}

// cuda_host_free releases CUDA-registered host memory.
static inline cudaError_t cuda_host_free(void* ptr) {
    return cudaFreeHost(ptr);
}

// cuda_host_get_flags retrieves the flags used when allocating pinned memory.
// Returns cudaSuccess if the pointer is registered pinned memory, error otherwise.
static inline cudaError_t cuda_host_get_flags(unsigned int* flags, void* ptr) {
    return cudaHostGetFlags(flags, ptr);
}
*/
import "C"
import (
	"errors"
	"fmt"
	"unsafe"

	"github.com/neurogrid/engine/pkg/types"
)

// DeviceInfo contains GPU device information.
type DeviceInfo struct {
	Name        string
	Major       int
	Minor       int
	TotalMemory uint64
}

// LayerWeights holds all weights for a transformer layer.
type LayerWeights struct {
	ptr unsafe.Pointer
}

// KVCache holds the key-value cache for attention.
type KVCache struct {
	ptr    unsafe.Pointer
	length int
}

// Ptr returns the underlying C pointer.
func (c *KVCache) Ptr() unsafe.Pointer { return c.ptr }

// HasCUDA returns true if CUDA is available (compiled with cuda tag).
func HasCUDA() bool {
	count, err := GetDeviceCount()
	if err != nil {
		return false
	}
	return count > 0
}

// InitGPU initializes the CUDA context for the specified device.
func InitGPU(deviceID int) error {
	result := C.cuda_init(C.int(deviceID))
	if result != 0 {
		return fmt.Errorf("CUDA init failed with code %d", result)
	}
	return nil
}

// ShutdownGPU releases CUDA resources.
func ShutdownGPU() {
	C.cuda_shutdown()
}

// GetDeviceCount returns the number of available CUDA devices.
func GetDeviceCount() (int, error) {
	var count C.int
	result := C.cuda_get_device_count(&count)
	if result != 0 {
		return 0, fmt.Errorf("failed to get device count: %d", result)
	}
	return int(count), nil
}

// SetDevice sets the current CUDA device.
func SetDevice(deviceID int) error {
	result := C.cuda_set_device(C.int(deviceID))
	if result != 0 {
		return fmt.Errorf("failed to set device %d: code %d", deviceID, result)
	}
	return nil
}

// GetDeviceInfo returns information about the current GPU.
func GetDeviceInfo() (*DeviceInfo, error) {
	var info C.DeviceInfo
	result := C.cuda_get_device_info(&info)
	if result != 0 {
		return nil, fmt.Errorf("failed to get device info: %d", result)
	}
	return &DeviceInfo{
		Name:        C.GoString(&info.name[0]),
		Major:       int(info.major),
		Minor:       int(info.minor),
		TotalMemory: uint64(info.total_memory),
	}, nil
}

// SyncDevice synchronizes the current device.
func SyncDevice() error {
	result := C.cuda_sync_device()
	if result != 0 {
		return fmt.Errorf("device sync failed: %d", result)
	}
	return nil
}

// GetMemoryUsed returns the amount of GPU memory currently used.
func GetMemoryUsed() (uint64, error) {
	var used C.size_t
	result := C.cuda_get_memory_used(&used)
	if result != 0 {
		return 0, fmt.Errorf("failed to get memory usage: %d", result)
	}
	return uint64(used), nil
}

// AllocateTensor allocates GPU memory for a tensor.
func AllocateTensor(t *types.Tensor) error {
	if t == nil {
		return errors.New("tensor is nil")
	}
	if err := t.Validate(); err != nil {
		return err
	}
	if t.Device < 0 || t.Device > 7 {
		return fmt.Errorf("invalid device ID: %d", t.Device)
	}

	size := C.size_t(t.ByteSize())
	var ptr unsafe.Pointer
	result := C.cuda_malloc(&ptr, size)
	if result != 0 {
		return fmt.Errorf("CUDA malloc failed: %d (size=%d)", result, size)
	}
	t.Data = ptr
	return nil
}

// FreeTensor frees GPU memory for a tensor.
func FreeTensor(t *types.Tensor) {
	if t == nil || t.Data == nil {
		return
	}
	C.cuda_free(t.Data)
	t.Data = nil
}

// CopyToDevice copies FP32 data from host to GPU tensor (with FP16 conversion if needed).
func CopyToDevice(t *types.Tensor, data []float32) error {
	if t == nil || t.Data == nil {
		return errors.New("tensor not allocated")
	}
	if len(data) < t.NumElements() {
		return fmt.Errorf("data size %d < tensor size %d", len(data), t.NumElements())
	}

	result := C.cuda_copy_to_device(
		t.Data,
		unsafe.Pointer(&data[0]),
		C.size_t(t.NumElements()),
		C.int(t.Dtype),
	)
	if result != 0 {
		return fmt.Errorf("copy to device failed: %d", result)
	}
	return nil
}

// CopyToHost copies GPU tensor data to host FP32 array.
func CopyToHost(data []float32, t *types.Tensor) error {
	if t == nil || t.Data == nil {
		return errors.New("tensor not allocated")
	}
	if len(data) < t.NumElements() {
		return fmt.Errorf("buffer size %d < tensor size %d", len(data), t.NumElements())
	}

	result := C.cuda_copy_to_host(
		unsafe.Pointer(&data[0]),
		t.Data,
		C.size_t(t.NumElements()),
		C.int(t.Dtype),
	)
	if result != 0 {
		return fmt.Errorf("copy to host failed: %d", result)
	}
	return nil
}

// CopyToDeviceINT8 copies INT8 data to GPU.
func CopyToDeviceINT8(t *types.Tensor, data []int8) error {
	if t == nil || t.Data == nil {
		return errors.New("tensor not allocated")
	}
	if t.Dtype != types.DtypeINT8 {
		return errors.New("tensor dtype must be INT8")
	}
	if len(data) < t.NumElements() {
		return fmt.Errorf("data size %d < tensor size %d", len(data), t.NumElements())
	}

	result := C.cuda_copy_int8_to_device(
		t.Data,
		unsafe.Pointer(&data[0]),
		C.size_t(t.NumElements()),
	)
	if result != 0 {
		return fmt.Errorf("copy INT8 to device failed: %d", result)
	}
	return nil
}

// CopyToHostINT8 copies INT8 data from GPU to host.
func CopyToHostINT8(data []int8, t *types.Tensor) error {
	if t == nil || t.Data == nil {
		return errors.New("tensor not allocated")
	}
	if t.Dtype != types.DtypeINT8 {
		return errors.New("tensor dtype must be INT8")
	}

	result := C.cuda_copy_int8_to_host(
		unsafe.Pointer(&data[0]),
		t.Data,
		C.size_t(t.NumElements()),
	)
	if result != 0 {
		return fmt.Errorf("copy INT8 to host failed: %d", result)
	}
	return nil
}

// RMSNorm executes the RMSNorm kernel.
func RMSNorm(output, input, weight *types.Tensor, eps float32) error {
	if output == nil || input == nil || weight == nil {
		return errors.New("nil tensor")
	}
	if output.Data == nil || input.Data == nil || weight.Data == nil {
		return errors.New("tensor not allocated")
	}

	hiddenDim := input.Shape[len(input.Shape)-1]
	numTokens := input.NumElements() / hiddenDim

	result := C.cuda_rmsnorm(
		output.Data,
		input.Data,
		weight.Data,
		C.int(numTokens),
		C.int(hiddenDim),
		C.float(eps),
	)
	if result != 0 {
		return fmt.Errorf("RMSNorm failed: %d", result)
	}
	return nil
}

// SiLU executes the SiLU activation kernel.
func SiLU(output, input *types.Tensor) error {
	if output == nil || input == nil {
		return errors.New("nil tensor")
	}

	result := C.cuda_silu(
		output.Data,
		input.Data,
		C.size_t(input.NumElements()),
	)
	if result != 0 {
		return fmt.Errorf("SiLU failed: %d", result)
	}
	return nil
}

// Add executes element-wise addition c = a + b.
func Add(c, a, b *types.Tensor) error {
	if c == nil || a == nil || b == nil {
		return errors.New("nil tensor")
	}

	result := C.cuda_add(
		c.Data,
		a.Data,
		b.Data,
		C.size_t(a.NumElements()),
	)
	if result != 0 {
		return fmt.Errorf("Add failed: %d", result)
	}
	return nil
}

// RoPE applies Rotary Position Embeddings.
func RoPE(output, input *types.Tensor, positions []int32, headDim int) error {
	if output == nil || input == nil {
		return errors.New("nil tensor")
	}
	if len(input.Shape) < 4 {
		return errors.New("input must have 4 dimensions: [batch, seq, heads, head_dim]")
	}

	batch := input.Shape[0]
	seq := input.Shape[1]
	numHeads := input.Shape[2]

	// Allocate GPU memory for positions array
	var dPositions unsafe.Pointer
	posSize := C.size_t(len(positions) * 4) // int32 = 4 bytes
	if C.cuda_malloc(&dPositions, posSize) != 0 {
		return errors.New("failed to allocate positions on GPU")
	}
	defer C.cuda_free(dPositions)

	// Copy positions to GPU
	if C.cudaMemcpy(dPositions, unsafe.Pointer(&positions[0]), posSize, C.cudaMemcpyHostToDevice) != 0 {
		return errors.New("failed to copy positions to GPU")
	}

	result := C.cuda_rope(
		output.Data,
		input.Data,
		(*C.int)(dPositions),
		C.int(batch),
		C.int(seq),
		C.int(numHeads),
		C.int(headDim),
	)
	if result != 0 {
		return fmt.Errorf("RoPE failed: %d", result)
	}
	return nil
}

// GEMMFP16 executes FP16 matrix multiplication C = A @ B.
func GEMMFP16(c, a, b *types.Tensor, transposeA, transposeB bool) error {
	if c == nil || a == nil || b == nil {
		return errors.New("nil tensor")
	}

	var M, K, N int
	if transposeA {
		M = a.Shape[1]
		K = a.Shape[0]
	} else {
		M = a.Shape[0]
		K = a.Shape[1]
	}
	if transposeB {
		N = b.Shape[0]
	} else {
		N = b.Shape[1]
	}

	result := C.cuda_gemm_fp16(
		c.Data,
		a.Data,
		b.Data,
		C.int(M),
		C.int(K),
		C.int(N),
		C.bool(transposeA),
		C.bool(transposeB),
	)
	if result != 0 {
		return fmt.Errorf("GEMM FP16 failed: %d", result)
	}
	return nil
}

// GEMMINT8 executes INT8 matrix multiplication with dequantization.
// transposeB: if true, B is stored as [N, K] and will be transposed.
func GEMMINT8(c, a, b, scale *types.Tensor, transposeB bool) error {
	if c == nil || a == nil || b == nil || scale == nil {
		return errors.New("nil tensor")
	}

	M := a.Shape[0]
	K := a.Shape[1]
	// N depends on transpose_b: if true, N = b.Shape[0], else N = b.Shape[1]
	var N int
	if transposeB {
		N = b.Shape[0]
	} else {
		N = b.Shape[1]
	}

	result := C.cuda_gemm_int8(
		c.Data,
		a.Data,
		b.Data,
		scale.Data,
		C.int(M),
		C.int(K),
		C.int(N),
		C.bool(transposeB),
	)
	if result != 0 {
		return fmt.Errorf("GEMM INT8 failed: %d", result)
	}
	return nil
}

// QuantizePerTensor quantizes FP16 tensor to INT8.
func QuantizePerTensor(output, scale, input *types.Tensor) error {
	if output == nil || scale == nil || input == nil {
		return errors.New("nil tensor")
	}

	result := C.cuda_quantize_per_tensor(
		output.Data,
		scale.Data,
		input.Data,
		C.size_t(input.NumElements()),
	)
	if result != 0 {
		return fmt.Errorf("quantize failed: %d", result)
	}
	return nil
}

// DequantizePerTensor dequantizes INT8 tensor to FP16.
func DequantizePerTensor(output, input, scale *types.Tensor) error {
	if output == nil || input == nil || scale == nil {
		return errors.New("nil tensor")
	}

	result := C.cuda_dequantize_per_tensor(
		output.Data,
		input.Data,
		scale.Data,
		C.size_t(input.NumElements()),
	)
	if result != 0 {
		return fmt.Errorf("dequantize failed: %d", result)
	}
	return nil
}

// QuantizeWeights performs per-column weight quantization.
func QuantizeWeights(output, scales, input *types.Tensor) error {
	if output == nil || scales == nil || input == nil {
		return errors.New("nil tensor")
	}

	K := input.Shape[0]
	N := input.Shape[1]

	result := C.cuda_quantize_weights(
		output.Data,
		scales.Data,
		input.Data,
		C.int(K),
		C.int(N),
	)
	if result != 0 {
		return fmt.Errorf("quantize weights failed: %d", result)
	}
	return nil
}

// BasicAttention computes attention without KV cache.
func BasicAttention(output, q, k, v *types.Tensor, causal bool) error {
	if output == nil || q == nil || k == nil || v == nil {
		return errors.New("nil tensor")
	}

	batch := q.Shape[0]
	numHeads := q.Shape[1]
	seqLen := q.Shape[2]
	headDim := q.Shape[3]

	result := C.cuda_basic_attention(
		output.Data,
		q.Data,
		k.Data,
		v.Data,
		C.int(batch),
		C.int(numHeads),
		C.int(seqLen),
		C.int(headDim),
		C.bool(causal),
	)
	if result != 0 {
		return fmt.Errorf("basic attention failed: %d", result)
	}
	return nil
}

// NewKVCache creates a new KV cache.
func NewKVCache(batchSize, numHeads, headDim, maxSeqLen int) (*KVCache, error) {
	var ptr unsafe.Pointer
	result := C.cuda_kvcache_create(
		&ptr,
		C.int(batchSize),
		C.int(numHeads),
		C.int(headDim),
		C.int(maxSeqLen),
	)
	if result != 0 {
		return nil, fmt.Errorf("failed to create KV cache: %d", result)
	}
	return &KVCache{ptr: ptr, length: 0}, nil
}

// FreeKVCache releases KV cache memory.
func FreeKVCache(cache *KVCache) {
	if cache == nil || cache.ptr == nil {
		return
	}
	C.cuda_kvcache_free(cache.ptr)
	cache.ptr = nil
}

// UpdateKVCache appends new K and V to the cache.
func UpdateKVCache(cache *KVCache, k, v []float32, position int) error {
	if cache == nil || cache.ptr == nil {
		return errors.New("cache is nil")
	}

	numElements := len(k)
	fp16Size := C.size_t(numElements * 2) // FP16 = 2 bytes

	// Allocate GPU memory for K
	var dK unsafe.Pointer
	if C.cuda_malloc(&dK, fp16Size) != 0 {
		return errors.New("failed to allocate K on GPU")
	}
	defer C.cuda_free(dK)

	// Allocate GPU memory for V
	var dV unsafe.Pointer
	if C.cuda_malloc(&dV, fp16Size) != 0 {
		return errors.New("failed to allocate V on GPU")
	}
	defer C.cuda_free(dV)

	// Copy K to GPU (FP32 -> FP16 conversion)
	if C.cuda_copy_to_device(dK, unsafe.Pointer(&k[0]), C.size_t(numElements), 1) != 0 {
		return errors.New("failed to copy K to GPU")
	}

	// Copy V to GPU (FP32 -> FP16 conversion)
	if C.cuda_copy_to_device(dV, unsafe.Pointer(&v[0]), C.size_t(numElements), 1) != 0 {
		return errors.New("failed to copy V to GPU")
	}

	result := C.cuda_kvcache_update(
		cache.ptr,
		dK,
		dV,
		C.int(position),
	)
	if result != 0 {
		return fmt.Errorf("KV cache update failed: %d", result)
	}
	if position >= cache.length {
		cache.length = position + 1
	}
	return nil
}

// GetKVCacheLength returns the current sequence length in the cache.
func GetKVCacheLength(cache *KVCache) int {
	if cache == nil || cache.ptr == nil {
		return 0
	}
	return int(C.cuda_kvcache_get_length(cache.ptr))
}

// AttentionWithKVCache computes attention using KV cache.
// Allocates a temporary GPU buffer for position (CUDA Graph safe internally).
func AttentionWithKVCache(output, q, k, v *types.Tensor, cache *KVCache, position int) error {
	if output == nil || q == nil || k == nil || v == nil || cache == nil {
		return errors.New("nil argument")
	}

	batch := q.Shape[0]
	numHeads := q.Shape[1]
	numKVHeads := k.Shape[1]
	headDim := q.Shape[3]

	// Allocate GPU buffer for position
	var dPos unsafe.Pointer
	if C.cuda_malloc(&dPos, C.size_t(4)) != 0 {
		return errors.New("failed to allocate position buffer")
	}
	defer C.cuda_free(dPos)
	pos := C.int(position)
	if C.cudaMemcpy(dPos, unsafe.Pointer(&pos), C.size_t(4), C.cudaMemcpyHostToDevice) != 0 {
		return errors.New("failed to copy position to GPU")
	}

	result := C.cuda_attention_with_kvcache(
		output.Data,
		q.Data,
		k.Data,
		v.Data,
		cache.ptr,
		(*C.int)(dPos),
		C.int(batch),
		C.int(numHeads),
		C.int(numKVHeads),
		C.int(headDim),
	)
	if result != 0 {
		return fmt.Errorf("attention with KV cache failed: %d", result)
	}
	return nil
}

// LoadLayerWeights loads weights from a directory.
func LoadLayerWeights(path string) (*LayerWeights, error) {
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	var ptr unsafe.Pointer
	result := C.cuda_load_layer_weights(&ptr, cPath)
	if result != 0 {
		return nil, fmt.Errorf("failed to load weights from %s: %d", path, result)
	}
	return &LayerWeights{ptr: ptr}, nil
}

// FreeLayerWeights releases layer weight memory.
func FreeLayerWeights(w *LayerWeights) {
	if w == nil || w.ptr == nil {
		return
	}
	C.cuda_free_layer_weights(w.ptr)
	w.ptr = nil
}

// CreateRandomLayerWeights creates random weights for testing.
func CreateRandomLayerWeights(config *types.LlamaConfig) (*LayerWeights, error) {
	var ptr unsafe.Pointer
	result := C.cuda_create_random_layer_weights(
		&ptr,
		C.int(config.HiddenSize),
		C.int(config.IntermediateSize),
		C.int(config.NumHeads),
		C.int(config.NumKVHeads),
		C.int(config.HeadDim),
	)
	if result != 0 {
		return nil, fmt.Errorf("failed to create random weights: %d", result)
	}
	return &LayerWeights{ptr: ptr}, nil
}

// CreateLayerWeightsFromHost creates layer weights from FP16 host data.
// All weight slices must be in FP16 format. They will be uploaded to GPU
// and quantized to INT8 for efficient inference.
func CreateLayerWeightsFromHost(
	qProj, kProj, vProj, oProj []byte,
	gateProj, upProj, downProj []byte,
	attnNorm, ffnNorm []byte,
	config *types.LlamaConfig,
) (*LayerWeights, error) {
	if len(qProj) == 0 || len(kProj) == 0 || len(vProj) == 0 || len(oProj) == 0 {
		return nil, errors.New("attention weight slices cannot be empty")
	}
	if len(gateProj) == 0 || len(upProj) == 0 || len(downProj) == 0 {
		return nil, errors.New("FFN weight slices cannot be empty")
	}
	if len(attnNorm) == 0 || len(ffnNorm) == 0 {
		return nil, errors.New("normalization weight slices cannot be empty")
	}

	var ptr unsafe.Pointer
	result := C.cuda_create_layer_weights_from_host(
		&ptr,
		unsafe.Pointer(&qProj[0]),
		unsafe.Pointer(&kProj[0]),
		unsafe.Pointer(&vProj[0]),
		unsafe.Pointer(&oProj[0]),
		unsafe.Pointer(&gateProj[0]),
		unsafe.Pointer(&upProj[0]),
		unsafe.Pointer(&downProj[0]),
		unsafe.Pointer(&attnNorm[0]),
		unsafe.Pointer(&ffnNorm[0]),
		C.int(config.HiddenSize),
		C.int(config.IntermediateSize),
		C.int(config.NumHeads),
		C.int(config.NumKVHeads),
		C.int(config.HeadDim),
	)
	if result != 0 {
		return nil, fmt.Errorf("failed to create layer weights from host: %d", result)
	}
	return &LayerWeights{ptr: ptr}, nil
}

// LayerForward executes a complete transformer layer forward pass.
func LayerForward(output, input *types.Tensor, weights *LayerWeights, cache *KVCache, positions []int32, config *types.LlamaConfig) error {
	if output == nil || input == nil || weights == nil || cache == nil {
		return errors.New("nil argument")
	}

	batch := input.Shape[0]
	seqLen := input.Shape[1]

	// Allocate GPU memory for positions array (required for Turing architecture)
	var dPositions unsafe.Pointer
	posSize := C.size_t(len(positions) * 4) // int32 = 4 bytes
	if C.cuda_malloc(&dPositions, posSize) != 0 {
		return errors.New("failed to allocate positions on GPU")
	}
	defer C.cuda_free(dPositions)

	// Copy positions to GPU
	if C.cudaMemcpy(dPositions, unsafe.Pointer(&positions[0]), posSize, C.cudaMemcpyHostToDevice) != 0 {
		return errors.New("failed to copy positions to GPU")
	}

	// Use RopeTheta from config, default to 10000.0 if not set
	ropeTheta := config.RopeTheta
	if ropeTheta == 0 {
		ropeTheta = 10000.0
	}

	result := C.cuda_layer_forward(
		output.Data,
		input.Data,
		weights.ptr,
		cache.ptr,
		(*C.int)(dPositions),
		C.int(batch),
		C.int(seqLen),
		C.int(config.HiddenSize),
		C.int(config.IntermediateSize),
		C.int(config.NumHeads),
		C.int(config.NumKVHeads),
		C.int(config.HeadDim),
		C.float(config.RMSNormEps),
		C.float(ropeTheta),
	)
	if result != 0 {
		return fmt.Errorf("layer forward failed: %d", result)
	}
	return nil
}

// =============================================================================
// FP16-Pure Layer (no INT8 quantization) — for LFM2 attention layers
// =============================================================================

// LayerWeightsFP16 holds FP16-pure weights (no INT8 quantization).
type LayerWeightsFP16 struct {
	ptr unsafe.Pointer
}

// Ptr returns the underlying C pointer.
func (w *LayerWeightsFP16) Ptr() unsafe.Pointer { return w.ptr }

// CreateLayerWeightsFromHostFP16 creates FP16-pure layer weights (no INT8 quantization).
func CreateLayerWeightsFromHostFP16(
	qProj, kProj, vProj, oProj []byte,
	gateProj, upProj, downProj []byte,
	attnNorm, ffnNorm []byte,
	qLayerNorm, kLayerNorm []byte,
	config *types.LlamaConfig,
) (*LayerWeightsFP16, error) {
	var ptr unsafe.Pointer

	// QK LayerNorm pointers (nil-safe: pass NULL if empty)
	var qLNPtr, kLNPtr unsafe.Pointer
	if len(qLayerNorm) > 0 {
		qLNPtr = unsafe.Pointer(&qLayerNorm[0])
	}
	if len(kLayerNorm) > 0 {
		kLNPtr = unsafe.Pointer(&kLayerNorm[0])
	}

	result := C.cuda_create_layer_weights_from_host_fp16(
		&ptr,
		unsafe.Pointer(&qProj[0]), unsafe.Pointer(&kProj[0]),
		unsafe.Pointer(&vProj[0]), unsafe.Pointer(&oProj[0]),
		unsafe.Pointer(&gateProj[0]), unsafe.Pointer(&upProj[0]),
		unsafe.Pointer(&downProj[0]),
		unsafe.Pointer(&attnNorm[0]), unsafe.Pointer(&ffnNorm[0]),
		qLNPtr, kLNPtr,
		C.int(config.HiddenSize), C.int(config.IntermediateSize),
		C.int(config.NumHeads), C.int(config.NumKVHeads), C.int(config.HeadDim),
	)
	if result != 0 {
		return nil, fmt.Errorf("failed to create FP16 layer weights: %d", result)
	}
	return &LayerWeightsFP16{ptr: ptr}, nil
}

// FreeLayerWeightsFP16 frees FP16-pure layer weights.
func FreeLayerWeightsFP16(w *LayerWeightsFP16) {
	if w == nil || w.ptr == nil {
		return
	}
	C.cuda_free_layer_weights_fp16(w.ptr)
	w.ptr = nil
}

// LayerForwardFP16 executes a FP16-pure transformer layer forward pass.
func LayerForwardFP16(output, input *types.Tensor, weights *LayerWeightsFP16, cache *KVCache,
	positions []int32, config *types.LlamaConfig, ropeStyle int) error {
	if output == nil || input == nil || weights == nil || cache == nil {
		return errors.New("nil argument")
	}

	batch := input.Shape[0]
	seqLen := input.Shape[1]

	var dPositions unsafe.Pointer
	posSize := C.size_t(len(positions) * 4)
	if C.cuda_malloc(&dPositions, posSize) != 0 {
		return errors.New("failed to allocate positions on GPU")
	}
	defer C.cuda_free(dPositions)

	if C.cudaMemcpy(dPositions, unsafe.Pointer(&positions[0]), posSize, C.cudaMemcpyHostToDevice) != 0 {
		return errors.New("failed to copy positions to GPU")
	}

	ropeTheta := config.RopeTheta
	if ropeTheta == 0 {
		ropeTheta = 10000.0
	}

	result := C.cuda_layer_forward_fp16(
		output.Data, input.Data, weights.ptr, cache.ptr,
		(*C.int)(dPositions),
		C.int(batch), C.int(seqLen),
		C.int(config.HiddenSize), C.int(config.IntermediateSize),
		C.int(config.NumHeads), C.int(config.NumKVHeads), C.int(config.HeadDim),
		C.float(config.RMSNormEps), C.float(ropeTheta), C.int(ropeStyle),
	)
	if result != 0 {
		return fmt.Errorf("FP16 layer forward failed: %d", result)
	}
	return nil
}

// LayerWorkspaceFP16 holds pre-allocated GPU buffers for FP16 layer forward passes.
// Eliminates per-call cudaMalloc/cudaFree overhead (~200 calls per token across 16 layers).
type LayerWorkspaceFP16 struct {
	ptr unsafe.Pointer
}

// CreateLayerWorkspaceFP16 pre-allocates all temporary GPU buffers needed by
// cuda_layer_forward_fp16. Call once at init, reuse across all forward calls.
func CreateLayerWorkspaceFP16(maxTokens int, config *types.LlamaConfig) (*LayerWorkspaceFP16, error) {
	var ptr unsafe.Pointer
	result := C.cuda_create_layer_workspace_fp16(
		&ptr,
		C.int(maxTokens),
		C.int(config.HiddenSize),
		C.int(config.IntermediateSize),
		C.int(config.NumKVHeads),
		C.int(config.HeadDim),
	)
	if result != 0 {
		return nil, fmt.Errorf("failed to create FP16 layer workspace: %d", result)
	}
	return &LayerWorkspaceFP16{ptr: ptr}, nil
}

// FreeLayerWorkspaceFP16 releases the pre-allocated workspace buffers.
func FreeLayerWorkspaceFP16(ws *LayerWorkspaceFP16) {
	if ws == nil || ws.ptr == nil {
		return
	}
	C.cuda_free_layer_workspace_fp16(ws.ptr)
	ws.ptr = nil
}

// LayerForwardFP16WithWorkspace executes FP16 layer forward using pre-allocated workspace
// buffers instead of allocating/freeing 8 GPU buffers per call.
func LayerForwardFP16WithWorkspace(output, input *types.Tensor, weights *LayerWeightsFP16, cache *KVCache,
	positions []int32, config *types.LlamaConfig, ropeStyle int, workspace *LayerWorkspaceFP16) error {
	if output == nil || input == nil || weights == nil || cache == nil || workspace == nil {
		return errors.New("nil argument")
	}

	batch := input.Shape[0]
	seqLen := input.Shape[1]

	var dPositions unsafe.Pointer
	posSize := C.size_t(len(positions) * 4)
	if C.cuda_malloc(&dPositions, posSize) != 0 {
		return errors.New("failed to allocate positions on GPU")
	}
	defer C.cuda_free(dPositions)

	if C.cudaMemcpy(dPositions, unsafe.Pointer(&positions[0]), posSize, C.cudaMemcpyHostToDevice) != 0 {
		return errors.New("failed to copy positions to GPU")
	}

	ropeTheta := config.RopeTheta
	if ropeTheta == 0 {
		ropeTheta = 10000.0
	}

	result := C.cuda_layer_forward_fp16_with_workspace(
		output.Data, input.Data, weights.ptr, cache.ptr,
		(*C.int)(dPositions),
		C.int(batch), C.int(seqLen),
		C.int(config.HiddenSize), C.int(config.IntermediateSize),
		C.int(config.NumHeads), C.int(config.NumKVHeads), C.int(config.HeadDim),
		C.float(config.RMSNormEps), C.float(ropeTheta), C.int(ropeStyle),
		workspace.ptr,
	)
	if result != 0 {
		return fmt.Errorf("FP16 layer forward with workspace failed: %d", result)
	}
	return nil
}

// =============================================================================
// Paged KV Cache (PagedAttention)
// =============================================================================

// PagedKVCache holds a block-based KV cache on GPU.
type PagedKVCache struct {
	ptr unsafe.Pointer
}

// CreatePagedKVCache creates a paged KV cache with the given number of blocks.
func CreatePagedKVCache(numBlocks, numKVHeads, headDim, blockSize int) (*PagedKVCache, error) {
	var ptr unsafe.Pointer
	result := C.cuda_paged_kvcache_create(&ptr, C.int(numBlocks), C.int(numKVHeads), C.int(headDim), C.int(blockSize))
	if result != 0 {
		return nil, fmt.Errorf("failed to create paged KV cache: %d", result)
	}
	return &PagedKVCache{ptr: ptr}, nil
}

// FreePagedKVCache releases the paged KV cache.
func FreePagedKVCache(cache *PagedKVCache) {
	if cache != nil && cache.ptr != nil {
		C.cuda_paged_kvcache_free(cache.ptr)
		cache.ptr = nil
	}
}

// PagedAttention runs paged attention with KV cache update.
// Allocates temporary GPU buffers for position and seq_len (CUDA Graph safe internally).
func PagedAttention(output, query, newKey, newValue unsafe.Pointer, cache *PagedKVCache,
	dBlockTable unsafe.Pointer, numHeads, numKVHeads, headDim, position int) error {
	// Allocate GPU buffers for seq_lens and position
	var dSeqLens, dPos unsafe.Pointer
	if C.cuda_malloc(&dSeqLens, C.size_t(4)) != 0 {
		return errors.New("failed to allocate seq_lens buffer")
	}
	defer C.cuda_free(dSeqLens)
	if C.cuda_malloc(&dPos, C.size_t(4)) != 0 {
		return errors.New("failed to allocate position buffer")
	}
	defer C.cuda_free(dPos)

	seqLen := C.int(position + 1)
	pos := C.int(position)
	if C.cudaMemcpy(dSeqLens, unsafe.Pointer(&seqLen), C.size_t(4), C.cudaMemcpyHostToDevice) != 0 {
		return errors.New("failed to copy seq_len to GPU")
	}
	if C.cudaMemcpy(dPos, unsafe.Pointer(&pos), C.size_t(4), C.cudaMemcpyHostToDevice) != 0 {
		return errors.New("failed to copy position to GPU")
	}

	result := C.cuda_paged_attention(output, query, newKey, newValue, cache.ptr,
		(*C.int)(dBlockTable), (*C.int)(dSeqLens), (*C.int)(dPos),
		C.int(numHeads), C.int(numKVHeads), C.int(headDim))
	if result != 0 {
		return fmt.Errorf("paged attention failed: %d", result)
	}
	return nil
}

// Ptr returns the underlying cache pointer.
func (c *PagedKVCache) Ptr() unsafe.Pointer { return c.ptr }

// LayerForwardFP16Paged executes FP16 layer forward with paged attention (block-based KV cache).
func LayerForwardFP16Paged(output, input *types.Tensor, weights *LayerWeightsFP16,
	pagedCache *PagedKVCache, dBlockTable unsafe.Pointer,
	positions []int32, config *types.LlamaConfig, ropeStyle int,
	workspace *LayerWorkspaceFP16) error {
	if output == nil || input == nil || weights == nil || pagedCache == nil || workspace == nil {
		return errors.New("nil argument to LayerForwardFP16Paged")
	}

	batch := input.Shape[0]
	seqLen := input.Shape[1]

	var dPositions unsafe.Pointer
	posSize := C.size_t(len(positions) * 4)
	if C.cuda_malloc(&dPositions, posSize) != 0 {
		return errors.New("failed to allocate positions on GPU")
	}
	defer C.cuda_free(dPositions)

	if C.cudaMemcpy(dPositions, unsafe.Pointer(&positions[0]), posSize, C.cudaMemcpyHostToDevice) != 0 {
		return errors.New("failed to copy positions to GPU")
	}

	// Allocate GPU buffer for seq_lens (= position + 1)
	var dSeqLens unsafe.Pointer
	if C.cuda_malloc(&dSeqLens, C.size_t(4)) != 0 {
		return errors.New("failed to allocate seq_lens buffer")
	}
	defer C.cuda_free(dSeqLens)
	seqLenVal := C.int(positions[0] + 1)
	if C.cudaMemcpy(dSeqLens, unsafe.Pointer(&seqLenVal), C.size_t(4), C.cudaMemcpyHostToDevice) != 0 {
		return errors.New("failed to copy seq_lens to GPU")
	}

	ropeTheta := config.RopeTheta
	if ropeTheta == 0 {
		ropeTheta = 10000.0
	}

	result := C.cuda_layer_forward_fp16_paged(
		output.Data, input.Data, weights.ptr,
		pagedCache.ptr,
		(*C.int)(dBlockTable),
		(*C.int)(dPositions),
		(*C.int)(dSeqLens),
		C.int(batch), C.int(seqLen),
		C.int(config.HiddenSize), C.int(config.IntermediateSize),
		C.int(config.NumHeads), C.int(config.NumKVHeads), C.int(config.HeadDim),
		C.float(config.RMSNormEps), C.float(ropeTheta), C.int(ropeStyle),
		workspace.ptr,
	)
	if result != 0 {
		return fmt.Errorf("FP16 paged layer forward failed: %d", result)
	}
	return nil
}

// =============================================================================
// Full Decode Context — all layers in a single CUDA call
// =============================================================================

// DecodeContext runs all layers in a single CUDA call (eliminates Go↔C round-trips).
type DecodeContext struct {
	ptr unsafe.Pointer
}

// CreateDecodeContext creates a context for full-model decode steps.
func CreateDecodeContext(config *types.LlamaConfig) (*DecodeContext, error) {
	var ptr unsafe.Pointer
	ropeTheta := config.RopeTheta
	if ropeTheta == 0 {
		ropeTheta = 10000.0
	}
	result := C.cuda_create_decode_context(&ptr,
		C.int(config.NumLayers), C.int(config.HiddenSize), C.int(config.IntermediateSize),
		C.int(config.NumHeads), C.int(config.NumKVHeads), C.int(config.HeadDim),
		C.float(config.RMSNormEps), C.float(ropeTheta), C.int(1), // rope_style=1 (interleaved for LFM2)
		C.int(config.ConvKernelSize))
	if result != 0 {
		return nil, fmt.Errorf("failed to create decode context: %d", result)
	}
	return &DecodeContext{ptr: ptr}, nil
}

// SetDecodeLayer registers a layer's weights and cache in the decode context.
func SetDecodeLayer(ctx *DecodeContext, layerID, layerType int, weights, cache unsafe.Pointer) {
	C.cuda_set_decode_layer(ctx.ptr, C.int(layerID), C.int(layerType), weights, cache)
}

// SetDecodeWorkspace sets the shared attention workspace.
func SetDecodeWorkspace(ctx *DecodeContext, workspace *LayerWorkspaceFP16) {
	if workspace != nil {
		C.cuda_set_decode_workspace(ctx.ptr, workspace.ptr)
	}
}

// ConvWorkspace holds pre-allocated buffers for conv layer forward (CUDA Graph safe).
type ConvWorkspace struct {
	ptr unsafe.Pointer
}

// CreateConvWorkspace creates a pre-allocated workspace for conv layer decode.
func CreateConvWorkspace(hiddenSize, intermediateSize int) (*ConvWorkspace, error) {
	var ptr unsafe.Pointer
	result := C.cuda_create_conv_workspace(&ptr, C.int(hiddenSize), C.int(intermediateSize))
	if result != 0 {
		return nil, fmt.Errorf("create conv workspace failed: %d", result)
	}
	return &ConvWorkspace{ptr: ptr}, nil
}

// FreeConvWorkspace releases the conv workspace.
func FreeConvWorkspace(ws *ConvWorkspace) {
	if ws != nil && ws.ptr != nil {
		C.cuda_free_conv_workspace(ws.ptr)
		ws.ptr = nil
	}
}

// SetDecodeConvWorkspace sets the conv workspace on the decode context.
func SetDecodeConvWorkspace(ctx *DecodeContext, ws *ConvWorkspace) {
	if ws != nil {
		C.cuda_set_decode_conv_workspace(ctx.ptr, ws.ptr)
	}
}

// SetDecodePagedCache sets the paged KV cache on the decode context.
func SetDecodePagedCache(ctx *DecodeContext, pagedCache *PagedKVCache, dBlockTable unsafe.Pointer, maxBlocksPerSeq int) {
	if pagedCache != nil {
		C.cuda_set_decode_paged_cache(ctx.ptr, pagedCache.ptr, (*C.int)(dBlockTable), C.int(maxBlocksPerSeq))
	}
}

// SetDecodePagedLayer sets a per-layer paged cache on the decode context.
func SetDecodePagedLayer(ctx *DecodeContext, layerID int, pagedCache *PagedKVCache) {
	if pagedCache != nil {
		C.cuda_set_decode_paged_layer(ctx.ptr, C.int(layerID), pagedCache.ptr)
	}
}

// DecodeStep runs all layers for a single token decode step.
// Input/output are HOST FP16 byte slices.
func DecodeStep(ctx *DecodeContext, output, input []byte, position int) error {
	result := C.cuda_decode_step(ctx.ptr,
		unsafe.Pointer(&output[0]),
		unsafe.Pointer(&input[0]),
		C.int(position))
	if result != 0 {
		return fmt.Errorf("decode step failed: %d", result)
	}
	return nil
}

// DecodeSetHiddenFromGPU copies hidden from another GPU buffer (GPU→GPU, zero-copy).
func DecodeSetHiddenFromGPU(ctx *DecodeContext, gpuPtr unsafe.Pointer) error {
	result := C.cuda_decode_set_hidden_from_gpu(ctx.ptr, gpuPtr)
	if result != 0 {
		return fmt.Errorf("decode set hidden from GPU failed: %d", result)
	}
	return nil
}

// DecodeSetHidden copies initial hidden state from host to GPU.
func DecodeSetHidden(ctx *DecodeContext, hidden []byte) error {
	result := C.cuda_decode_set_hidden(ctx.ptr, unsafe.Pointer(&hidden[0]))
	if result != 0 {
		return fmt.Errorf("decode set hidden failed: %d", result)
	}
	return nil
}

// DecodeGetHidden copies hidden state from GPU to host.
func DecodeGetHidden(ctx *DecodeContext, hidden []byte) error {
	result := C.cuda_decode_get_hidden(ctx.ptr, unsafe.Pointer(&hidden[0]))
	if result != 0 {
		return fmt.Errorf("decode get hidden failed: %d", result)
	}
	return nil
}

// DecodeStepGPU runs all layers with hidden state staying on GPU.
func DecodeStepGPU(ctx *DecodeContext, position int) error {
	result := C.cuda_decode_step_gpu(ctx.ptr, C.int(position))
	if result != 0 {
		return fmt.Errorf("decode step GPU failed: %d", result)
	}
	return nil
}

// DecodeGetHiddenGPUPtr returns GPU pointer to current hidden state (for LM head).
func DecodeGetHiddenGPUPtr(ctx *DecodeContext) unsafe.Pointer {
	return C.cuda_decode_get_hidden_gpu_ptr(ctx.ptr)
}

// FreeDecodeContext releases the decode context.
func FreeDecodeContext(ctx *DecodeContext) {
	if ctx != nil && ctx.ptr != nil {
		C.cuda_free_decode_context(ctx.ptr)
		ctx.ptr = nil
	}
}

// =============================================================================
// LFM2 / BF16 Operations
// =============================================================================

// ConvLayerWeights holds weights for an LFM2 conv layer.
type ConvLayerWeights struct {
	ptr unsafe.Pointer
}

// Ptr returns the underlying C pointer.
func (w *ConvLayerWeights) Ptr() unsafe.Pointer { return w.ptr }

// CheckBF16Support returns true if the current GPU supports BF16 (compute >= 8.0).
func CheckBF16Support() (bool, error) {
	var supported C.int
	result := C.cuda_check_bf16_support(&supported)
	if result != 0 {
		return false, fmt.Errorf("failed to check BF16 support: %d", result)
	}
	return supported != 0, nil
}

// FP16ToBF16 converts FP16 data to BF16 on GPU.
func FP16ToBF16(output, input unsafe.Pointer, numElements int) error {
	result := C.cuda_fp16_to_bf16(output, input, C.size_t(numElements))
	if result != 0 {
		return fmt.Errorf("FP16→BF16 failed: %d", result)
	}
	return nil
}

// BF16ToFP16 converts BF16 data to FP16 on GPU.
func BF16ToFP16(output, input unsafe.Pointer, numElements int) error {
	result := C.cuda_bf16_to_fp16(output, input, C.size_t(numElements))
	if result != 0 {
		return fmt.Errorf("BF16→FP16 failed: %d", result)
	}
	return nil
}

// GEMMBF16 executes BF16 matrix multiplication C = A @ B.
func GEMMBF16(cPtr, aPtr, bPtr unsafe.Pointer, M, K, N int, transposeA, transposeB bool) error {
	result := C.cuda_gemm_bf16(cPtr, aPtr, bPtr, C.int(M), C.int(K), C.int(N),
		C.bool(transposeA), C.bool(transposeB))
	if result != 0 {
		return fmt.Errorf("GEMM BF16 failed: %d", result)
	}
	return nil
}

// ConvStateCreate creates a conv state buffer on GPU.
func ConvStateCreate(batch, dim, width int) (unsafe.Pointer, error) {
	ptr := C.cuda_conv_state_create(C.int(batch), C.int(dim), C.int(width))
	if ptr == nil {
		return nil, errors.New("failed to create conv state")
	}
	return ptr, nil
}

// ConvStateReset zeros a conv state buffer.
func ConvStateReset(state unsafe.Pointer, batch, dim, width int) error {
	result := C.cuda_conv_state_reset(state, C.int(batch), C.int(dim), C.int(width))
	if result != 0 {
		return fmt.Errorf("conv state reset failed: %d", result)
	}
	return nil
}

// ConvStateFree frees a conv state buffer.
func ConvStateFree(state unsafe.Pointer) {
	if state != nil {
		C.cuda_conv_state_free(state)
	}
}

// CreateConvLayerWeightsBF16 creates conv layer weights from BF16 host data.
func CreateConvLayerWeightsBF16(
	inProj, conv, outProj []byte,
	opNorm, ffnNorm []byte,
	gate, up, down []byte,
	hidden, intermediate, kernelSize int, normEps float32,
) (*ConvLayerWeights, error) {
	var ptr unsafe.Pointer
	result := C.cuda_create_conv_layer_weights_bf16(
		&ptr,
		unsafe.Pointer(&inProj[0]), unsafe.Pointer(&conv[0]), unsafe.Pointer(&outProj[0]),
		unsafe.Pointer(&opNorm[0]), unsafe.Pointer(&ffnNorm[0]),
		unsafe.Pointer(&gate[0]), unsafe.Pointer(&up[0]), unsafe.Pointer(&down[0]),
		C.int(hidden), C.int(intermediate), C.int(kernelSize), C.float(normEps),
	)
	if result != 0 {
		return nil, fmt.Errorf("failed to create conv layer weights: %d", result)
	}
	return &ConvLayerWeights{ptr: ptr}, nil
}

// ConvLayerForwardBF16 executes a conv layer forward pass.
func ConvLayerForwardBF16(output, input unsafe.Pointer, weights *ConvLayerWeights, convState unsafe.Pointer, batch, seqLen, position int) error {
	if weights == nil || weights.ptr == nil {
		return errors.New("nil conv layer weights")
	}
	result := C.cuda_conv_layer_forward_bf16(output, input, weights.ptr, convState,
		C.int(batch), C.int(seqLen), C.int(position))
	if result != 0 {
		return fmt.Errorf("conv layer forward failed: %d", result)
	}
	return nil
}

// FreeConvLayerWeights releases conv layer weight memory.
func FreeConvLayerWeights(w *ConvLayerWeights) {
	if w == nil || w.ptr == nil {
		return
	}
	C.cuda_free_conv_layer_weights(w.ptr)
	w.ptr = nil
}

// =============================================================================
// Multi-GPU Operations (TASK-001 through TASK-005)
// =============================================================================

// DeviceContext represents a single GPU device context for multi-GPU operations.
type DeviceContext struct {
	DeviceID       int
	TotalMemory    uint64
	UsedMemory     uint64
	ComputeStream  unsafe.Pointer
	TransferStream unsafe.Pointer
	PeerAccess     []bool
}

// MultiDeviceManagerInfo contains information about the multi-device manager.
type MultiDeviceManagerInfo struct {
	NumDevices        int
	StagingBufferSize uint64
}

var multiGPUInitialized = false

// InitMultiGPU initializes multiple GPU devices for distributed operations.
// This creates a DeviceContext for each device with dedicated streams and P2P access matrix.
func InitMultiGPU(deviceIDs []int) error {
	if len(deviceIDs) == 0 {
		return errors.New("no devices specified")
	}

	// TODO: Implement CUDA multi-device initialization
	// This should call cuda_multi_init with the device IDs array
	result := C.cuda_multi_init(
		(*C.int)(unsafe.Pointer(&deviceIDs[0])),
		C.int(len(deviceIDs)),
	)
	if result != 0 {
		return fmt.Errorf("multi-GPU init failed: %d", result)
	}

	multiGPUInitialized = true
	return nil
}

// ShutdownMultiGPU releases all multi-GPU resources.
func ShutdownMultiGPU() error {
	if !multiGPUInitialized {
		return errors.New("multi-GPU not initialized")
	}

	C.cuda_multi_shutdown()
	multiGPUInitialized = false
	return nil
}

// GetDeviceContext returns the context for a specific device.
func GetDeviceContext(deviceID int) (*DeviceContext, error) {
	if !multiGPUInitialized {
		return nil, errors.New("multi-GPU not initialized")
	}

	var ctx C.DeviceContext
	result := C.cuda_get_device_context(C.int(deviceID), &ctx)
	if result != 0 {
		return nil, fmt.Errorf("failed to get device context for device %d: %d", deviceID, result)
	}

	// Convert C struct to Go struct
	peerAccess := make([]bool, 8)
	for i := 0; i < 8; i++ {
		peerAccess[i] = bool(ctx.peer_access[i])
	}

	return &DeviceContext{
		DeviceID:       int(ctx.device_id),
		TotalMemory:    uint64(ctx.total_memory),
		UsedMemory:     uint64(ctx.used_memory),
		ComputeStream:  unsafe.Pointer(ctx.compute_stream),
		TransferStream: unsafe.Pointer(ctx.transfer_stream),
		PeerAccess:     peerAccess,
	}, nil
}

// GetP2PAccessMatrix returns the P2P access matrix for all initialized devices.
func GetP2PAccessMatrix() ([][]bool, error) {
	if !multiGPUInitialized {
		return nil, errors.New("multi-GPU not initialized")
	}

	var numDevices C.int
	C.cuda_multi_get_num_devices(&numDevices)

	matrix := make([][]bool, int(numDevices))
	for i := 0; i < int(numDevices); i++ {
		matrix[i] = make([]bool, int(numDevices))
		for j := 0; j < int(numDevices); j++ {
			var canAccess C.int
			C.cuda_can_access_peer(C.int(i), C.int(j), &canAccess)
			matrix[i][j] = canAccess != 0
		}
	}

	return matrix, nil
}

// GetMultiDeviceManagerInfo returns information about the multi-device manager.
func GetMultiDeviceManagerInfo() (*MultiDeviceManagerInfo, error) {
	if !multiGPUInitialized {
		return nil, errors.New("multi-GPU not initialized")
	}

	var numDevices C.int
	var stagingSize C.size_t
	C.cuda_multi_get_num_devices(&numDevices)
	C.cuda_multi_get_staging_buffer_size(&stagingSize)

	return &MultiDeviceManagerInfo{
		NumDevices:        int(numDevices),
		StagingBufferSize: uint64(stagingSize),
	}, nil
}

// CanAccessPeer checks if device srcDevice can access memory on dstDevice via P2P.
func CanAccessPeer(srcDevice, dstDevice int) (bool, error) {
	if !multiGPUInitialized {
		return false, errors.New("multi-GPU not initialized")
	}

	var canAccess C.int
	result := C.cuda_can_access_peer(C.int(srcDevice), C.int(dstDevice), &canAccess)
	if result != 0 {
		return false, fmt.Errorf("failed to check P2P access: %d", result)
	}

	return canAccess != 0, nil
}

// AllocOnDevice allocates memory on a specific GPU device.
func AllocOnDevice(size uint64, deviceID int) (unsafe.Pointer, error) {
	if !multiGPUInitialized {
		return nil, errors.New("multi-GPU not initialized")
	}

	var ptr unsafe.Pointer
	result := C.cuda_alloc_on_device(&ptr, C.size_t(size), C.int(deviceID))
	if result != 0 {
		return nil, fmt.Errorf("allocation failed on device %d: %d", deviceID, result)
	}

	return ptr, nil
}

// FreeOnDevice frees memory on a specific GPU device.
func FreeOnDevice(ptr unsafe.Pointer, deviceID int) error {
	if !multiGPUInitialized {
		return errors.New("multi-GPU not initialized")
	}

	result := C.cuda_free_on_device(ptr, C.int(deviceID))
	if result != 0 {
		return fmt.Errorf("free failed on device %d: %d", deviceID, result)
	}

	return nil
}

// CrossDeviceCopy copies data between GPUs (uses P2P if available, staged copy otherwise).
func CrossDeviceCopy(dst unsafe.Pointer, dstDevice int, src unsafe.Pointer, srcDevice int, size uint64) error {
	if !multiGPUInitialized {
		return errors.New("multi-GPU not initialized")
	}

	result := C.cuda_cross_device_copy(dst, C.int(dstDevice), src, C.int(srcDevice), C.size_t(size), nil)
	if result != 0 {
		return fmt.Errorf("cross-device copy failed: %d", result)
	}

	return nil
}

// CopyToDeviceRaw copies raw bytes to a device pointer.
func CopyToDeviceRaw(dst unsafe.Pointer, src unsafe.Pointer, size uint64) error {
	result := C.cudaMemcpy(dst, src, C.size_t(size), C.cudaMemcpyHostToDevice)
	if result != 0 {
		return fmt.Errorf("copy to device failed: %d", result)
	}
	return nil
}

// CopyFromDeviceRaw copies raw bytes from a device pointer.
func CopyFromDeviceRaw(dst unsafe.Pointer, src unsafe.Pointer, size uint64) error {
	result := C.cudaMemcpy(dst, src, C.size_t(size), C.cudaMemcpyDeviceToHost)
	if result != 0 {
		return fmt.Errorf("copy from device failed: %d", result)
	}
	return nil
}

// =============================================================================
// CUDA Pinned Memory Operations (for DMA transfers)
// =============================================================================

// AllocPinnedMemory allocates CUDA-registered host memory for DMA transfers.
// Uses cudaHostAllocPortable flag for multi-GPU visibility.
// The allocated memory is page-locked and suitable for asynchronous transfers.
//
// CRITICAL: Uses cudaHostAllocPortable so pinned memory is visible from all GPUs.
//
// Parameters:
//   - size: number of bytes to allocate (must be > 0)
//
// Returns:
//   - pointer to allocated pinned memory
//   - error if allocation fails or size is zero
func AllocPinnedMemory(size uint64) (unsafe.Pointer, error) {
	if size == 0 {
		return nil, errors.New("cannot allocate zero bytes of pinned memory")
	}

	var ptr unsafe.Pointer
	result := C.cuda_host_alloc(&ptr, C.size_t(size))
	if result != C.cudaSuccess {
		return nil, fmt.Errorf("cudaHostAlloc failed: error=%d, requested_size=%d bytes", result, size)
	}
	return ptr, nil
}

// FreePinnedMemory releases CUDA-registered host memory.
// Safe to call with nil pointer (no-op).
//
// Parameters:
//   - ptr: pointer to pinned memory to free (nil is safe)
//
// Returns:
//   - error if CUDA deallocation fails
func FreePinnedMemory(ptr unsafe.Pointer) error {
	if ptr == nil {
		return nil
	}
	result := C.cuda_host_free(ptr)
	if result != C.cudaSuccess {
		return fmt.Errorf("cudaFreeHost failed: error=%d, ptr=%p", result, ptr)
	}
	return nil
}

// IsPinnedMemory checks if a pointer points to CUDA-registered pinned memory.
// Returns true if the memory was allocated with cudaHostAlloc/cudaMallocHost.
//
// Parameters:
//   - ptr: pointer to check (nil returns false, nil)
//
// Returns:
//   - isPinned: true if memory is CUDA-registered pinned memory
//   - error: nil (errors are suppressed as non-pinned memory returns cudaErrorInvalidValue)
func IsPinnedMemory(ptr unsafe.Pointer) (bool, error) {
	if ptr == nil {
		return false, nil
	}
	var flags C.uint
	result := C.cuda_host_get_flags(&flags, ptr)
	if result != C.cudaSuccess {
		// Not pinned memory or error - return false without error for non-pinned
		// CUDA returns cudaErrorInvalidValue for non-pinned memory
		return false, nil
	}
	return true, nil
}
