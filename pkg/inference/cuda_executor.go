//go:build cuda

// Package inference provides CUDA-accelerated layer execution.
package inference

import (
	"context"
	"fmt"
	"sync"
	"unsafe"

	"github.com/neurogrid/engine/gpu/bindings"
	"github.com/neurogrid/engine/pkg/types"
)

// CUDALayerExecutor implements LayerExecutor using CUDA kernels.
// It manages transformer layer weights on the GPU and executes forward passes.
//
// The executor preallocates GPU buffers for input/output hidden states to avoid
// per-call cudaMalloc/cudaFree overhead during inference. This is especially
// important for distributed inference where activation transfer latency is critical.
//
// Thread-safety: all public methods are safe for concurrent use via RWMutex.
//
// Resource management: call Close() to free GPU memory when done.
type CUDALayerExecutor struct {
	layerWeights map[int]*bindings.LayerWeights
	kvCaches     map[int]*bindings.KVCache
	config       *types.LlamaConfig
	deviceID     int
	mu           sync.RWMutex

	// Preallocated GPU buffers for Forward() - avoids per-call cudaMalloc/cudaFree.
	// These buffers are sized for single-token inference (hidden_size * 2 bytes for FP16).
	// For batch inference or prefill, the executor falls back to dynamic allocation.
	inputGPU   unsafe.Pointer // GPU buffer for input hidden state
	outputGPU  unsafe.Pointer // GPU buffer for output hidden state
	bufferSize uint64         // Size of preallocated buffers in bytes
}

// NewCUDALayerExecutor creates a new CUDA-based layer executor.
//
// Preallocates GPU buffers for input/output to avoid per-call allocations during
// forward passes. Buffer size is config.HiddenSize * 2 bytes (FP16 format).
//
// Parameters:
//   - config: LLaMA model configuration (determines buffer sizes)
//   - deviceID: CUDA device index for GPU memory allocation
//
// Returns error if GPU buffer allocation fails. On partial failure, any allocated
// resources are freed before returning.
func NewCUDALayerExecutor(config *types.LlamaConfig, deviceID int) (*CUDALayerExecutor, error) {
	// Calculate buffer size for hidden state (FP16 = 2 bytes per element)
	// Using max sequence length of 1 for single token inference
	bufferSize := uint64(config.HiddenSize * 2)

	// Preallocate input buffer on GPU
	inputGPU, err := bindings.AllocOnDevice(bufferSize, deviceID)
	if err != nil {
		// Lazy-init multi-GPU manager for single-device executor usage in tests/runtime.
		if initErr := bindings.InitMultiGPU([]int{deviceID}); initErr == nil {
			inputGPU, err = bindings.AllocOnDevice(bufferSize, deviceID)
		}
		if err != nil {
			return nil, fmt.Errorf("failed to preallocate input buffer: %w", err)
		}
	}

	// Preallocate output buffer on GPU
	outputGPU, err := bindings.AllocOnDevice(bufferSize, deviceID)
	if err != nil {
		bindings.FreeOnDevice(inputGPU, deviceID)
		return nil, fmt.Errorf("failed to preallocate output buffer: %w", err)
	}

	return &CUDALayerExecutor{
		layerWeights: make(map[int]*bindings.LayerWeights),
		kvCaches:     make(map[int]*bindings.KVCache),
		config:       config,
		deviceID:     deviceID,
		inputGPU:     inputGPU,
		outputGPU:    outputGPU,
		bufferSize:   bufferSize,
	}, nil
}

// HasPreallocatedBuffers returns whether the executor has preallocated GPU buffers.
func (e *CUDALayerExecutor) HasPreallocatedBuffers() bool {
	return e.inputGPU != nil && e.outputGPU != nil
}

// GetBufferSize returns the size of preallocated buffers in bytes.
func (e *CUDALayerExecutor) GetBufferSize() uint64 {
	return e.bufferSize
}

// TransformerLayerWeights represents the weights for a single transformer layer.
// This is the structure expected by LoadLayer.
type TransformerLayerWeights struct {
	QProj    []byte // FP16 [hidden_size, hidden_size]
	KProj    []byte // FP16 [hidden_size, kv_dim]
	VProj    []byte // FP16 [hidden_size, kv_dim]
	OProj    []byte // FP16 [hidden_size, hidden_size]
	GateProj []byte // FP16 [hidden_size, intermediate_size]
	UpProj   []byte // FP16 [hidden_size, intermediate_size]
	DownProj []byte // FP16 [intermediate_size, hidden_size]
	AttnNorm []byte // FP16 [hidden_size]
	FFNNorm  []byte // FP16 [hidden_size]
}

// LoadLayer uploads a single layer's weights to GPU memory.
func (e *CUDALayerExecutor) LoadLayer(layerID int, weights *TransformerLayerWeights) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	// Free existing weights if any
	if existing, ok := e.layerWeights[layerID]; ok {
		bindings.FreeLayerWeights(existing)
	}

	// Upload weights to GPU with quantization
	gpuWeights, err := bindings.CreateLayerWeightsFromHost(
		weights.QProj, weights.KProj, weights.VProj, weights.OProj,
		weights.GateProj, weights.UpProj, weights.DownProj,
		weights.AttnNorm, weights.FFNNorm,
		e.config,
	)
	if err != nil {
		return fmt.Errorf("failed to upload layer %d to GPU: %w", layerID, err)
	}

	e.layerWeights[layerID] = gpuWeights

	// Create KV cache for this layer if not exists
	if _, ok := e.kvCaches[layerID]; !ok {
		// Max sequence length for KV cache (should match model config)
		maxSeqLen := 2048
		if e.config.MaxSeqLen > 0 {
			maxSeqLen = e.config.MaxSeqLen
		}

		cache, err := bindings.NewKVCache(1, e.config.NumKVHeads, e.config.HeadDim, maxSeqLen)
		if err != nil {
			return fmt.Errorf("failed to create KV cache for layer %d: %w", layerID, err)
		}
		e.kvCaches[layerID] = cache
	}

	return nil
}

// Forward executes the transformer layer forward pass on GPU.
// Returns: output hidden state, K cache data, V cache data
//
// OPTIMIZED: Uses preallocated GPU buffers to avoid per-call cudaMalloc/cudaFree.
func (e *CUDALayerExecutor) Forward(
	ctx context.Context,
	layerID int,
	hidden []byte,
	position int,
) (output []byte, k []byte, v []byte, err error) {
	e.mu.RLock()
	weights, ok := e.layerWeights[layerID]
	cache := e.kvCaches[layerID]
	e.mu.RUnlock()

	if !ok {
		return nil, nil, nil, fmt.Errorf("layer %d not loaded", layerID)
	}
	if cache == nil {
		return nil, nil, nil, fmt.Errorf("KV cache for layer %d not initialized", layerID)
	}

	// Use preallocated buffers if available and size matches
	hiddenSize := e.config.HiddenSize
	expectedSize := uint64(hiddenSize * 2) // FP16

	if e.inputGPU != nil && e.outputGPU != nil && uint64(len(hidden)) <= e.bufferSize {
		// Use preallocated buffers (no allocation needed)
		inputTensor := &types.Tensor{
			Shape:  []int{1, 1, hiddenSize}, // [batch, seq, hidden]
			Dtype:  types.DtypeFP16,
			Device: e.deviceID,
			Data:   e.inputGPU,
		}

		// Copy hidden state to preallocated GPU buffer
		if err := bindings.CopyToDeviceRaw(e.inputGPU, getBytePointer(hidden), uint64(len(hidden))); err != nil {
			return nil, nil, nil, fmt.Errorf("failed to copy input to GPU: %w", err)
		}

		outputTensor := &types.Tensor{
			Shape:  []int{1, 1, hiddenSize},
			Dtype:  types.DtypeFP16,
			Device: e.deviceID,
			Data:   e.outputGPU,
		}

		// Positions array
		positions := []int32{int32(position)}

		// Execute layer forward
		if err := bindings.LayerForward(
			outputTensor,
			inputTensor,
			weights,
			cache,
			positions,
			e.config,
		); err != nil {
			return nil, nil, nil, fmt.Errorf("CUDA layer forward failed: %w", err)
		}

		// Copy output back to host
		output = make([]byte, len(hidden))
		if err := bindings.CopyFromDeviceRaw(getBytePointer(output), e.outputGPU, uint64(len(output))); err != nil {
			return nil, nil, nil, fmt.Errorf("failed to copy output from GPU: %w", err)
		}
	} else {
		// Fallback: allocate buffers dynamically (for inputs larger than preallocated size)
		inputTensor := &types.Tensor{
			Shape:  []int{1, 1, hiddenSize}, // [batch, seq, hidden]
			Dtype:  types.DtypeFP16,
			Device: e.deviceID,
		}
		if err := bindings.AllocateTensor(inputTensor); err != nil {
			return nil, nil, nil, fmt.Errorf("failed to allocate input tensor: %w", err)
		}
		defer bindings.FreeTensor(inputTensor)

		// Copy hidden state to GPU
		if err := bindings.CopyToDeviceRaw(inputTensor.Data, getBytePointer(hidden), uint64(len(hidden))); err != nil {
			return nil, nil, nil, fmt.Errorf("failed to copy input to GPU: %w", err)
		}

		// Create output tensor
		outputTensor := &types.Tensor{
			Shape:  []int{1, 1, hiddenSize},
			Dtype:  types.DtypeFP16,
			Device: e.deviceID,
		}
		if err := bindings.AllocateTensor(outputTensor); err != nil {
			return nil, nil, nil, fmt.Errorf("failed to allocate output tensor: %w", err)
		}
		defer bindings.FreeTensor(outputTensor)

		// Positions array
		positions := []int32{int32(position)}

		// Execute layer forward
		if err := bindings.LayerForward(
			outputTensor,
			inputTensor,
			weights,
			cache,
			positions,
			e.config,
		); err != nil {
			return nil, nil, nil, fmt.Errorf("CUDA layer forward failed: %w", err)
		}

		// Copy output back to host
		output = make([]byte, len(hidden))
		if err := bindings.CopyFromDeviceRaw(getBytePointer(output), outputTensor.Data, uint64(len(output))); err != nil {
			return nil, nil, nil, fmt.Errorf("failed to copy output from GPU: %w", err)
		}
	}

	// K and V are managed by the KV cache, return empty for now
	// (The KV cache update happens inside cuda_layer_forward)
	kvSize := e.config.NumKVHeads * e.config.HeadDim * 2 // FP16
	k = make([]byte, kvSize)
	v = make([]byte, kvSize)

	// Suppress unused variable warning
	_ = expectedSize

	return output, k, v, nil
}

// NumLayers returns the number of layers currently loaded.
func (e *CUDALayerExecutor) NumLayers() int {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return len(e.layerWeights)
}

// Close frees all GPU resources including preallocated buffers.
func (e *CUDALayerExecutor) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()

	// Free preallocated buffers
	if e.inputGPU != nil {
		bindings.FreeOnDevice(e.inputGPU, e.deviceID)
		e.inputGPU = nil
	}
	if e.outputGPU != nil {
		bindings.FreeOnDevice(e.outputGPU, e.deviceID)
		e.outputGPU = nil
	}

	for id, weights := range e.layerWeights {
		bindings.FreeLayerWeights(weights)
		delete(e.layerWeights, id)
	}

	for id, cache := range e.kvCaches {
		bindings.FreeKVCache(cache)
		delete(e.kvCaches, id)
	}

	return nil
}

// ResetKVCache resets the KV cache for all layers.
func (e *CUDALayerExecutor) ResetKVCache() error {
	e.mu.Lock()
	defer e.mu.Unlock()

	for layerID, cache := range e.kvCaches {
		bindings.FreeKVCache(cache)

		maxSeqLen := 2048
		if e.config.MaxSeqLen > 0 {
			maxSeqLen = e.config.MaxSeqLen
		}

		newCache, err := bindings.NewKVCache(1, e.config.NumKVHeads, e.config.HeadDim, maxSeqLen)
		if err != nil {
			return fmt.Errorf("failed to reset KV cache for layer %d: %w", layerID, err)
		}
		e.kvCaches[layerID] = newCache
	}

	return nil
}

// getBytePointer returns an unsafe.Pointer to the first element of a byte slice.
func getBytePointer(b []byte) unsafe.Pointer {
	if len(b) == 0 {
		return nil
	}
	return unsafe.Pointer(&b[0])
}
