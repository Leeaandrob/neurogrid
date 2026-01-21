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
type CUDALayerExecutor struct {
	layerWeights map[int]*bindings.LayerWeights
	kvCaches     map[int]*bindings.KVCache
	config       *types.LlamaConfig
	deviceID     int
	mu           sync.RWMutex
}

// NewCUDALayerExecutor creates a new CUDA-based layer executor.
func NewCUDALayerExecutor(config *types.LlamaConfig, deviceID int) *CUDALayerExecutor {
	return &CUDALayerExecutor{
		layerWeights: make(map[int]*bindings.LayerWeights),
		kvCaches:     make(map[int]*bindings.KVCache),
		config:       config,
		deviceID:     deviceID,
	}
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

	// Create input tensor (FP16 on GPU)
	hiddenSize := e.config.HiddenSize
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

	// K and V are managed by the KV cache, return empty for now
	// (The KV cache update happens inside cuda_layer_forward)
	kvSize := e.config.NumKVHeads * e.config.HeadDim * 2 // FP16
	k = make([]byte, kvSize)
	v = make([]byte, kvSize)

	return output, k, v, nil
}

// NumLayers returns the number of layers currently loaded.
func (e *CUDALayerExecutor) NumLayers() int {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return len(e.layerWeights)
}

// Close frees all GPU resources.
func (e *CUDALayerExecutor) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()

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
