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

	// LFM2 conv layer support
	convLayerWeights map[int]*bindings.ConvLayerWeights
	convStates       map[int]unsafe.Pointer // GPU conv state pointers

	// FP16-pure attention layers (LFM2 — no INT8 quantization)
	fp16LayerWeights map[int]*bindings.LayerWeightsFP16

	// Full decode context — runs all layers in single CUDA call
	decodeCtx *bindings.DecodeContext

	// Preallocated GPU buffers for Forward() - avoids per-call cudaMalloc/cudaFree.
	// These buffers are sized for single-token inference (hidden_size * 2 bytes for FP16).
	// For batch inference or prefill, the executor falls back to dynamic allocation.
	inputGPU   unsafe.Pointer // GPU buffer for input hidden state
	outputGPU  unsafe.Pointer // GPU buffer for output hidden state
	bufferSize uint64         // Size of preallocated buffers in bytes

	// FP16 layer workspace — pre-allocated temp buffers for zero-alloc forward passes.
	// Eliminates ~200 cudaMalloc/cudaFree calls per token across 16 layers (~2-3ms saved).
	fp16Workspace *bindings.LayerWorkspaceFP16

	// Paged KV Cache — one cache per attention layer (replaces contiguous kvCaches for LFM2)
	pagedCaches     map[int]*bindings.PagedKVCache // layerID -> paged cache
	pagedManager    *PagedKVCacheManager
	blockTableGPU   unsafe.Pointer // GPU buffer for block table
	maxBlocksPerSeq int
	usePagedCache   bool

	// Preallocated conv layer buffers — avoids 4 cudaMalloc/cudaFree per ForwardConv call.
	convInputFP16  unsafe.Pointer // GPU buffer for FP16 input
	convInputBF16  unsafe.Pointer // GPU buffer for BF16 input (converted from FP16)
	convOutputBF16 unsafe.Pointer // GPU buffer for BF16 output
	convOutputFP16 unsafe.Pointer // GPU buffer for FP16 output (converted from BF16)
	convBufSize    uint64         // Size of each conv buffer in bytes
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
		return nil, fmt.Errorf("failed to preallocate input buffer: %w", err)
	}

	// Preallocate output buffer on GPU
	outputGPU, err := bindings.AllocOnDevice(bufferSize, deviceID)
	if err != nil {
		bindings.FreeOnDevice(inputGPU, deviceID)
		return nil, fmt.Errorf("failed to preallocate output buffer: %w", err)
	}

	// Preallocate FP16 layer workspace (8 temp buffers used by every forward call)
	fp16Workspace, err := bindings.CreateLayerWorkspaceFP16(1, config)
	if err != nil {
		// Non-fatal: fall back to per-call allocation via LayerForwardFP16
		fp16Workspace = nil
	}

	// Preallocate conv layer buffers (4 buffers used by every ForwardConv call)
	convBufSize := bufferSize // same as hidden_size * 2 bytes (FP16/BF16)
	var convInputFP16, convInputBF16, convOutputBF16, convOutputFP16 unsafe.Pointer

	convInputFP16, err = bindings.AllocOnDevice(convBufSize, deviceID)
	if err != nil {
		convInputFP16 = nil // non-fatal
	}
	if convInputFP16 != nil {
		convInputBF16, err = bindings.AllocOnDevice(convBufSize, deviceID)
		if err != nil {
			bindings.FreeOnDevice(convInputFP16, deviceID)
			convInputFP16 = nil
			convInputBF16 = nil
		}
	}
	if convInputBF16 != nil {
		convOutputBF16, err = bindings.AllocOnDevice(convBufSize, deviceID)
		if err != nil {
			bindings.FreeOnDevice(convInputFP16, deviceID)
			bindings.FreeOnDevice(convInputBF16, deviceID)
			convInputFP16 = nil
			convInputBF16 = nil
			convOutputBF16 = nil
		}
	}
	if convOutputBF16 != nil {
		convOutputFP16, err = bindings.AllocOnDevice(convBufSize, deviceID)
		if err != nil {
			bindings.FreeOnDevice(convInputFP16, deviceID)
			bindings.FreeOnDevice(convInputBF16, deviceID)
			bindings.FreeOnDevice(convOutputBF16, deviceID)
			convInputFP16 = nil
			convInputBF16 = nil
			convOutputBF16 = nil
			convOutputFP16 = nil
		}
	}

	return &CUDALayerExecutor{
		layerWeights:     make(map[int]*bindings.LayerWeights),
		kvCaches:         make(map[int]*bindings.KVCache),
		convLayerWeights: make(map[int]*bindings.ConvLayerWeights),
		convStates:       make(map[int]unsafe.Pointer),
		fp16LayerWeights: make(map[int]*bindings.LayerWeightsFP16),
		config:           config,
		deviceID:         deviceID,
		inputGPU:         inputGPU,
		outputGPU:        outputGPU,
		bufferSize:       bufferSize,
		fp16Workspace:    fp16Workspace,
		convInputFP16:    convInputFP16,
		convInputBF16:    convInputBF16,
		convOutputBF16:   convOutputBF16,
		convOutputFP16:   convOutputFP16,
		convBufSize:      convBufSize,
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

// InitPagedKVCache initializes the paged KV cache for block-based attention.
// numBlocks is the total number of KV cache blocks to allocate on GPU.
func (e *CUDALayerExecutor) InitPagedKVCache(numBlocks, numKVHeads, headDim int) error {
	// Count attention layers
	numAttnLayers := 0
	for i := 0; i < e.config.NumLayers; i++ {
		if !e.config.IsConvLayer(i) {
			numAttnLayers++
		}
	}
	if numAttnLayers == 0 {
		return nil
	}

	// Divide blocks among attention layers
	blocksPerLayer := numBlocks / numAttnLayers
	if blocksPerLayer < 16 {
		return fmt.Errorf("not enough blocks: %d for %d layers", numBlocks, numAttnLayers)
	}

	e.pagedCaches = make(map[int]*bindings.PagedKVCache)

	// Create one paged cache per attention layer
	for i := 0; i < e.config.NumLayers; i++ {
		if e.config.IsConvLayer(i) {
			continue
		}
		cache, err := bindings.CreatePagedKVCache(blocksPerLayer, numKVHeads, headDim, BlockSize)
		if err != nil {
			// Cleanup already created caches
			for _, c := range e.pagedCaches {
				bindings.FreePagedKVCache(c)
			}
			e.pagedCaches = nil
			return fmt.Errorf("create paged cache for layer %d: %w", i, err)
		}
		e.pagedCaches[i] = cache
	}

	e.pagedManager = NewPagedKVCacheManager(blocksPerLayer, 1, numKVHeads, headDim) // 1 layer per manager (blocks shared per layer)
	e.usePagedCache = true

	// Pre-allocate GPU block table buffer
	e.maxBlocksPerSeq = 256
	tableSize := uint64(e.maxBlocksPerSeq * 4)
	ptr, err := bindings.AllocOnDevice(tableSize, e.deviceID)
	if err != nil {
		return fmt.Errorf("alloc block table GPU buffer: %w", err)
	}
	e.blockTableGPU = ptr
	return nil
}

// HasPagedCache returns whether paged KV cache is initialized.
func (e *CUDALayerExecutor) HasPagedCache() bool {
	return e.usePagedCache && len(e.pagedCaches) > 0
}

// PagedManager returns the paged KV cache manager.
func (e *CUDALayerExecutor) PagedManager() *PagedKVCacheManager {
	return e.pagedManager
}

// AllocateSequence allocates paged KV cache blocks for a new sequence.
func (e *CUDALayerExecutor) AllocateSequence(seqID uint64, maxTokens int) error {
	if e.pagedManager == nil {
		return nil // No paged cache, silently succeed
	}
	return e.pagedManager.AllocateForSequence(seqID, maxTokens)
}

// AppendToken records that a token was added to the sequence's KV cache.
func (e *CUDALayerExecutor) AppendToken(seqID uint64) error {
	if e.pagedManager == nil {
		return nil
	}
	return e.pagedManager.AppendToken(seqID)
}

// FreeSequence releases all paged KV cache blocks for a sequence.
func (e *CUDALayerExecutor) FreeSequence(seqID uint64) {
	if e.pagedManager == nil {
		return
	}
	e.pagedManager.FreeSequence(seqID)
}

// GetBlockTableGPU returns the GPU pointer to the block table and sequence length.
func (e *CUDALayerExecutor) GetBlockTableGPU(seqID uint64) (unsafe.Pointer, int, error) {
	if e.pagedManager == nil {
		return nil, 0, fmt.Errorf("paged cache not initialized")
	}

	blockTable, seqLen, err := e.pagedManager.GetBlockTable(seqID, e.maxBlocksPerSeq)
	if err != nil {
		return nil, 0, err
	}

	// Copy block table to GPU
	tableBytes := uint64(len(blockTable) * 4)
	if err := bindings.CopyToDeviceRaw(e.blockTableGPU, unsafe.Pointer(&blockTable[0]), tableBytes); err != nil {
		return nil, 0, fmt.Errorf("copy block table to GPU: %w", err)
	}

	return e.blockTableGPU, seqLen, nil
}

// TransformerLayerWeights represents the weights for a single transformer layer.
// This is the structure expected by LoadLayer.
type TransformerLayerWeights struct {
	QProj      []byte // FP16 [hidden_size, hidden_size]
	KProj      []byte // FP16 [hidden_size, kv_dim]
	VProj      []byte // FP16 [hidden_size, kv_dim]
	OProj      []byte // FP16 [hidden_size, hidden_size]
	GateProj   []byte // FP16 [hidden_size, intermediate_size]
	UpProj     []byte // FP16 [hidden_size, intermediate_size]
	DownProj   []byte // FP16 [intermediate_size, hidden_size]
	AttnNorm   []byte // FP16 [hidden_size]
	FFNNorm    []byte // FP16 [hidden_size]
	QLayerNorm []byte // FP16 [head_dim] — per-head QK LayerNorm (nil if not used)
	KLayerNorm []byte // FP16 [head_dim] — per-head QK LayerNorm (nil if not used)
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

// LoadLayerFP16 uploads attention layer weights as FP16 (no INT8 quantization).
// Used for LFM2 where INT8 quantization destroys precision.
func (e *CUDALayerExecutor) LoadLayerFP16(layerID int, weights *TransformerLayerWeights) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if existing, ok := e.fp16LayerWeights[layerID]; ok {
		bindings.FreeLayerWeightsFP16(existing)
	}

	gpuWeights, err := bindings.CreateLayerWeightsFromHostFP16(
		weights.QProj, weights.KProj, weights.VProj, weights.OProj,
		weights.GateProj, weights.UpProj, weights.DownProj,
		weights.AttnNorm, weights.FFNNorm,
		weights.QLayerNorm, weights.KLayerNorm,
		e.config,
	)
	if err != nil {
		return fmt.Errorf("failed to upload FP16 layer %d: %w", layerID, err)
	}
	e.fp16LayerWeights[layerID] = gpuWeights

	// Create KV cache
	if _, ok := e.kvCaches[layerID]; !ok {
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
	fp16Weights, fp16Ok := e.fp16LayerWeights[layerID]
	cache := e.kvCaches[layerID]
	e.mu.RUnlock()

	// Use FP16-pure path if available (LFM2 attention layers)
	if fp16Ok && fp16Weights != nil {
		return e.forwardFP16(ctx, layerID, hidden, position, fp16Weights, cache)
	}

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

// BuildDecodeContext creates the full-model decode context for LFM2.
// Must be called after all layers are loaded. Enables DecodeAll() fast path.
func (e *CUDALayerExecutor) BuildDecodeContext() error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.config.ModelType != "lfm2" {
		return nil // Only for LFM2 hybrid models
	}

	ctx, err := bindings.CreateDecodeContext(e.config)
	if err != nil {
		return fmt.Errorf("create decode context: %w", err)
	}

	// Register all layers
	for i := 0; i < e.config.NumLayers; i++ {
		if e.config.IsConvLayer(i) {
			if w, ok := e.convLayerWeights[i]; ok {
				state := e.convStates[i]
				bindings.SetDecodeLayer(ctx, i, 0, w.Ptr(), state) // 0=conv
			}
		} else {
			if w, ok := e.fp16LayerWeights[i]; ok {
				cache := e.kvCaches[i]
				var cachePtr unsafe.Pointer
				if cache != nil {
					cachePtr = cache.Ptr()
				}
				bindings.SetDecodeLayer(ctx, i, 1, w.Ptr(), cachePtr) // 1=attention
			}
		}
	}

	// Set workspace
	if e.fp16Workspace != nil {
		bindings.SetDecodeWorkspace(ctx, e.fp16Workspace)
	}

	e.decodeCtx = ctx
	return nil
}

// DecodeAll runs all layers for a single decode step in one CUDA call.
func (e *CUDALayerExecutor) DecodeAll(hidden []byte, position int) ([]byte, error) {
	if e.decodeCtx == nil {
		return nil, fmt.Errorf("decode context not initialized")
	}
	// When paged cache is active, update the block table on GPU before decode.
	// The decode context's paged attention reads from the same persistent block table buffer.
	if e.usePagedCache && e.pagedManager != nil {
		activeSeqID := e.pagedManager.FirstActiveSequenceID()
		if activeSeqID == 0 {
			return nil, fmt.Errorf("no active sequence for paged decode")
		}
		blockTable, _, err := e.pagedManager.GetBlockTable(activeSeqID, e.maxBlocksPerSeq)
		if err != nil {
			return nil, fmt.Errorf("get block table: %w", err)
		}
		tableBytes := uint64(len(blockTable) * 4)
		if err := bindings.CopyToDeviceRaw(e.blockTableGPU, getBytePointer(blockTableToBytes(blockTable)), tableBytes); err != nil {
			return nil, fmt.Errorf("copy block table: %w", err)
		}
	}

	output := make([]byte, len(hidden))
	if err := bindings.DecodeStep(e.decodeCtx, output, hidden, position); err != nil {
		return nil, err
	}
	return output, nil
}

// DecodeStepGPUResident runs all layers with hidden state on GPU.
// Call SetHiddenGPU first, then this for each token. Hidden stays on GPU.
func (e *CUDALayerExecutor) DecodeStepGPUResident(position int) error {
	if e.decodeCtx == nil {
		return fmt.Errorf("decode context not initialized")
	}
	// When paged cache is active, update block table before decode
	if e.usePagedCache && e.pagedManager != nil {
		activeSeqID := e.pagedManager.FirstActiveSequenceID()
		if activeSeqID > 0 {
			blockTable, _, err := e.pagedManager.GetBlockTable(activeSeqID, e.maxBlocksPerSeq)
			if err == nil {
				tableBytes := uint64(len(blockTable) * 4)
				bindings.CopyToDeviceRaw(e.blockTableGPU, getBytePointer(blockTableToBytes(blockTable)), tableBytes)
			}
		}
	}
	return bindings.DecodeStepGPU(e.decodeCtx, position)
}

// SetHiddenFromGPU copies hidden from another GPU buffer (zero-copy).
func (e *CUDALayerExecutor) SetHiddenFromGPU(gpuPtr unsafe.Pointer) error {
	if e.decodeCtx == nil {
		return fmt.Errorf("decode context not initialized")
	}
	return bindings.DecodeSetHiddenFromGPU(e.decodeCtx, gpuPtr)
}

// SetHiddenGPU copies hidden state from host to GPU decode context.
func (e *CUDALayerExecutor) SetHiddenGPU(hidden []byte) error {
	if e.decodeCtx == nil {
		return fmt.Errorf("decode context not initialized")
	}
	return bindings.DecodeSetHidden(e.decodeCtx, hidden)
}

// GetHiddenGPU copies hidden state from GPU to host.
func (e *CUDALayerExecutor) GetHiddenGPU(hidden []byte) error {
	if e.decodeCtx == nil {
		return fmt.Errorf("decode context not initialized")
	}
	return bindings.DecodeGetHidden(e.decodeCtx, hidden)
}

// GetHiddenGPUPtr returns GPU pointer to current hidden (for LM head).
func (e *CUDALayerExecutor) GetHiddenGPUPtr() unsafe.Pointer {
	if e.decodeCtx == nil {
		return nil
	}
	return bindings.DecodeGetHiddenGPUPtr(e.decodeCtx)
}

func (e *CUDALayerExecutor) forwardFP16(ctx context.Context, layerID int, hidden []byte, position int,
	weights *bindings.LayerWeightsFP16, cache *bindings.KVCache) ([]byte, []byte, []byte, error) {

	hiddenSize := e.config.HiddenSize

	inputTensor := &types.Tensor{
		Shape:  []int{1, 1, hiddenSize},
		Dtype:  types.DtypeFP16,
		Device: e.deviceID,
		Data:   e.inputGPU,
	}

	if err := bindings.CopyToDeviceRaw(e.inputGPU, getBytePointer(hidden), uint64(len(hidden))); err != nil {
		return nil, nil, nil, fmt.Errorf("copy input: %w", err)
	}

	outputTensor := &types.Tensor{
		Shape:  []int{1, 1, hiddenSize},
		Dtype:  types.DtypeFP16,
		Device: e.deviceID,
		Data:   e.outputGPU,
	}

	positions := []int32{int32(position)}

	// Use interleaved RoPE for LFM2
	ropeStyle := 1 // RoPEStyleInterleaved

	// Choose attention path: paged (per-layer) or contiguous
	usedPaged := false
	if e.usePagedCache && e.pagedManager != nil && e.fp16Workspace != nil {
		// Get per-layer paged cache
		layerCache, hasLayerCache := e.pagedCaches[layerID]
		if hasLayerCache {
			// Get block table from paged manager
			activeSeqID := e.pagedManager.FirstActiveSequenceID()
			blockTable, _, btErr := e.pagedManager.GetBlockTable(activeSeqID, e.maxBlocksPerSeq)
			if btErr == nil {
				tableBytes := uint64(len(blockTable) * 4)
				if cpErr := bindings.CopyToDeviceRaw(e.blockTableGPU, unsafe.Pointer(&blockTable[0]), tableBytes); cpErr != nil {
					return nil, nil, nil, fmt.Errorf("copy block table to GPU: %w", cpErr)
				}

				if fwdErr := bindings.LayerForwardFP16Paged(outputTensor, inputTensor, weights,
					layerCache, e.blockTableGPU,
					positions, e.config, ropeStyle, e.fp16Workspace); fwdErr != nil {
					return nil, nil, nil, fmt.Errorf("FP16 paged layer forward failed: %w", fwdErr)
				}
				usedPaged = true
			}
		}
	}

	if !usedPaged {
		// Use workspace path if available (eliminates ~8 cudaMalloc/cudaFree per layer call)
		if e.fp16Workspace != nil {
			if err := bindings.LayerForwardFP16WithWorkspace(outputTensor, inputTensor, weights, cache, positions, e.config, ropeStyle, e.fp16Workspace); err != nil {
				return nil, nil, nil, fmt.Errorf("FP16 layer forward (workspace) failed: %w", err)
			}
		} else {
			// Fallback to allocating path
			if err := bindings.LayerForwardFP16(outputTensor, inputTensor, weights, cache, positions, e.config, ropeStyle); err != nil {
				return nil, nil, nil, fmt.Errorf("FP16 layer forward failed: %w", err)
			}
		}
	}
	output := make([]byte, len(hidden))
	if err := bindings.CopyFromDeviceRaw(getBytePointer(output), e.outputGPU, uint64(len(output))); err != nil {
		return nil, nil, nil, fmt.Errorf("copy output: %w", err)
	}

	kvSize := e.config.NumKVHeads * e.config.HeadDim * 2
	k := make([]byte, kvSize)
	v := make([]byte, kvSize)

	return output, k, v, nil
}

// LoadConvLayer uploads conv layer weights to GPU.
func (e *CUDALayerExecutor) LoadConvLayer(layerID int, inProj, conv, outProj, opNorm, ffnNorm, gate, up, down []byte,
	hidden, intermediate, kernelSize int, normEps float32) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	// Free existing weights
	if existing, ok := e.convLayerWeights[layerID]; ok {
		bindings.FreeConvLayerWeights(existing)
	}

	weights, err := bindings.CreateConvLayerWeightsBF16(
		inProj, conv, outProj, opNorm, ffnNorm, gate, up, down,
		hidden, intermediate, kernelSize, normEps,
	)
	if err != nil {
		return fmt.Errorf("failed to upload conv layer %d: %w", layerID, err)
	}
	e.convLayerWeights[layerID] = weights

	// Create conv state
	if _, ok := e.convStates[layerID]; !ok {
		state, err := bindings.ConvStateCreate(1, hidden, kernelSize)
		if err != nil {
			return fmt.Errorf("failed to create conv state for layer %d: %w", layerID, err)
		}
		e.convStates[layerID] = state
	}

	return nil
}

// ForwardConv executes a conv layer forward pass.
//
// OPTIMIZED: Uses preallocated GPU buffers when available to eliminate 4
// cudaMalloc/cudaFree calls per invocation. Falls back to dynamic allocation
// if preallocated buffers are not available or input exceeds buffer size.
func (e *CUDALayerExecutor) ForwardConv(ctx context.Context, layerID int, hidden []byte, position int) ([]byte, error) {
	e.mu.RLock()
	weights, ok := e.convLayerWeights[layerID]
	state := e.convStates[layerID]
	e.mu.RUnlock()

	if !ok || weights == nil {
		return nil, fmt.Errorf("conv layer %d not loaded", layerID)
	}
	if state == nil {
		return nil, fmt.Errorf("conv state for layer %d not initialized", layerID)
	}

	hiddenSize := e.config.HiddenSize
	numElements := hiddenSize // single token
	seqLen := 1

	bufSize := uint64(numElements * 2) // 2 bytes per element (FP16 or BF16)

	// Use preallocated buffers if available and size fits
	if e.convInputFP16 != nil && bufSize <= e.convBufSize {
		// Copy FP16 hidden state to preallocated GPU buffer
		if err := bindings.CopyToDeviceRaw(e.convInputFP16, getBytePointer(hidden), uint64(len(hidden))); err != nil {
			return nil, fmt.Errorf("copy input: %w", err)
		}

		// Convert FP16 → BF16
		if err := bindings.FP16ToBF16(e.convInputBF16, e.convInputFP16, numElements); err != nil {
			return nil, fmt.Errorf("fp16→bf16 input: %w", err)
		}

		// Execute conv layer forward (BF16 in, BF16 out)
		if err := bindings.ConvLayerForwardBF16(e.convOutputBF16, e.convInputBF16, weights, state, 1, seqLen, position); err != nil {
			return nil, fmt.Errorf("conv forward: %w", err)
		}

		// Convert BF16 → FP16
		if err := bindings.BF16ToFP16(e.convOutputFP16, e.convOutputBF16, numElements); err != nil {
			return nil, fmt.Errorf("bf16→fp16 output: %w", err)
		}

		// Copy FP16 output back to host
		output := make([]byte, len(hidden))
		if err := bindings.CopyFromDeviceRaw(getBytePointer(output), e.convOutputFP16, uint64(len(output))); err != nil {
			return nil, fmt.Errorf("copy output: %w", err)
		}

		return output, nil
	}

	// Fallback: allocate buffers dynamically
	fp16InputGPU, err := bindings.AllocOnDevice(bufSize, e.deviceID)
	if err != nil {
		return nil, fmt.Errorf("alloc fp16 input: %w", err)
	}
	defer bindings.FreeOnDevice(fp16InputGPU, e.deviceID)

	if err := bindings.CopyToDeviceRaw(fp16InputGPU, getBytePointer(hidden), uint64(len(hidden))); err != nil {
		return nil, fmt.Errorf("copy input: %w", err)
	}

	bf16InputGPU, err := bindings.AllocOnDevice(bufSize, e.deviceID)
	if err != nil {
		return nil, fmt.Errorf("alloc bf16 input: %w", err)
	}
	defer bindings.FreeOnDevice(bf16InputGPU, e.deviceID)

	if err := bindings.FP16ToBF16(bf16InputGPU, fp16InputGPU, numElements); err != nil {
		return nil, fmt.Errorf("fp16→bf16 input: %w", err)
	}

	bf16OutputGPU, err := bindings.AllocOnDevice(bufSize, e.deviceID)
	if err != nil {
		return nil, fmt.Errorf("alloc bf16 output: %w", err)
	}
	defer bindings.FreeOnDevice(bf16OutputGPU, e.deviceID)

	if err := bindings.ConvLayerForwardBF16(bf16OutputGPU, bf16InputGPU, weights, state, 1, seqLen, position); err != nil {
		return nil, fmt.Errorf("conv forward: %w", err)
	}

	fp16OutputGPU, err := bindings.AllocOnDevice(bufSize, e.deviceID)
	if err != nil {
		return nil, fmt.Errorf("alloc fp16 output: %w", err)
	}
	defer bindings.FreeOnDevice(fp16OutputGPU, e.deviceID)

	if err := bindings.BF16ToFP16(fp16OutputGPU, bf16OutputGPU, numElements); err != nil {
		return nil, fmt.Errorf("bf16→fp16 output: %w", err)
	}

	output := make([]byte, len(hidden))
	if err := bindings.CopyFromDeviceRaw(getBytePointer(output), fp16OutputGPU, uint64(len(output))); err != nil {
		return nil, fmt.Errorf("copy output: %w", err)
	}

	return output, nil
}

// Close frees all GPU resources including preallocated buffers.
func (e *CUDALayerExecutor) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()

	// Free decode context
	if e.decodeCtx != nil {
		bindings.FreeDecodeContext(e.decodeCtx)
		e.decodeCtx = nil
	}

	// Free paged KV cache resources
	for id, cache := range e.pagedCaches {
		bindings.FreePagedKVCache(cache)
		delete(e.pagedCaches, id)
	}
	if e.blockTableGPU != nil {
		bindings.FreeOnDevice(e.blockTableGPU, e.deviceID)
		e.blockTableGPU = nil
	}
	e.pagedManager = nil

	// Free preallocated buffers
	if e.inputGPU != nil {
		bindings.FreeOnDevice(e.inputGPU, e.deviceID)
		e.inputGPU = nil
	}
	if e.outputGPU != nil {
		bindings.FreeOnDevice(e.outputGPU, e.deviceID)
		e.outputGPU = nil
	}

	// Free FP16 layer workspace
	if e.fp16Workspace != nil {
		bindings.FreeLayerWorkspaceFP16(e.fp16Workspace)
		e.fp16Workspace = nil
	}

	// Free preallocated conv buffers
	if e.convInputFP16 != nil {
		bindings.FreeOnDevice(e.convInputFP16, e.deviceID)
		e.convInputFP16 = nil
	}
	if e.convInputBF16 != nil {
		bindings.FreeOnDevice(e.convInputBF16, e.deviceID)
		e.convInputBF16 = nil
	}
	if e.convOutputBF16 != nil {
		bindings.FreeOnDevice(e.convOutputBF16, e.deviceID)
		e.convOutputBF16 = nil
	}
	if e.convOutputFP16 != nil {
		bindings.FreeOnDevice(e.convOutputFP16, e.deviceID)
		e.convOutputFP16 = nil
	}

	for id, weights := range e.layerWeights {
		bindings.FreeLayerWeights(weights)
		delete(e.layerWeights, id)
	}

	for id, cache := range e.kvCaches {
		bindings.FreeKVCache(cache)
		delete(e.kvCaches, id)
	}

	for id, w := range e.fp16LayerWeights {
		bindings.FreeLayerWeightsFP16(w)
		delete(e.fp16LayerWeights, id)
	}

	for id, weights := range e.convLayerWeights {
		bindings.FreeConvLayerWeights(weights)
		delete(e.convLayerWeights, id)
	}
	for id, state := range e.convStates {
		bindings.ConvStateFree(state)
		delete(e.convStates, id)
	}

	return nil
}

// ResetKVCache resets the KV cache for all layers.
func (e *CUDALayerExecutor) ResetKVCache() error {
	e.mu.Lock()
	defer e.mu.Unlock()

	// Free all sequences in paged manager
	if e.pagedManager != nil {
		e.pagedManager.FreeAll()
	}

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

	for layerID, state := range e.convStates {
		if err := bindings.ConvStateReset(state, 1, e.config.HiddenSize, e.config.ConvKernelSize); err != nil {
			return fmt.Errorf("failed to reset conv state for layer %d: %w", layerID, err)
		}
	}

	return nil
}

// blockTableToBytes converts []int32 to []byte for GPU copy.
func blockTableToBytes(table []int32) []byte {
	if len(table) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&table[0])), len(table)*4)
}

// getBytePointer returns an unsafe.Pointer to the first element of a byte slice.
func getBytePointer(b []byte) unsafe.Pointer {
	if len(b) == 0 {
		return nil
	}
	return unsafe.Pointer(&b[0])
}
