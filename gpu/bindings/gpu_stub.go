//go:build !cuda

// Package bindings provides stub implementations for testing without CUDA.
// Use build tag 'cuda' for real CUDA bindings.
package bindings

import (
	"errors"
	"sync"
	"unsafe"

	"github.com/neurogrid/engine/pkg/types"
)

var (
	// ErrNotImplemented is returned for unimplemented stub functions
	ErrNotImplemented = errors.New("CUDA not available: stub implementation")

	// ErrMultiDeviceNotInitialized is returned when multi-device is not initialized
	ErrMultiDeviceNotInitialized = errors.New("multi-device context not initialized")

	// Stub state for multi-device management
	stubState = &stubMultiDeviceState{}
)

// stubAllocation tracks a simulated memory allocation
type stubAllocation struct {
	deviceID int
	data     []byte
	size     uint64
}

// stubMultiDeviceState holds stub state for testing
type stubMultiDeviceState struct {
	mu               sync.RWMutex
	initialized      bool
	deviceIDs        []int
	deviceContexts   map[int]*DeviceContext
	p2pMatrix        [][]bool
	allocations      map[uintptr]*stubAllocation // ptr address -> allocation
	nextAllocAddr    uintptr
	stagingBufSize   uint64
}

// DeviceInfo contains GPU device information.
type DeviceInfo struct {
	Name        string
	Major       int
	Minor       int
	TotalMemory uint64
}

// DeviceContext holds per-GPU state for multi-device operations.
type DeviceContext struct {
	DeviceID       int
	TotalMemory    uint64
	UsedMemory     uint64
	ComputeStream  unsafe.Pointer
	TransferStream unsafe.Pointer
	PeerAccess     []bool
}

// MultiDeviceManagerInfo provides information about the multi-device manager.
type MultiDeviceManagerInfo struct {
	NumDevices        int
	DeviceIDs         []int
	StagingBufferSize uint64
	Initialized       bool
}

// findDeviceIndex returns the index of deviceID in the device list, or -1 if not found.
func (s *stubMultiDeviceState) findDeviceIndex(deviceID int) int {
	for i, id := range s.deviceIDs {
		if id == deviceID {
			return i
		}
	}
	return -1
}

// =============================================================================
// Device Management (Stubs)
// =============================================================================

// HasCUDA returns false for stub builds (no CUDA available).
func HasCUDA() bool {
	return false
}

// InitGPU initializes CUDA for the specified device (stub).
func InitGPU(deviceID int) error {
	return nil // Stub: always succeeds
}

// Init initializes CUDA for the specified device (stub).
func Init(deviceID int) error {
	return nil // Stub: always succeeds
}

// Shutdown releases CUDA resources (stub).
func Shutdown() {
	// Stub: no-op
}

// GetDeviceCount returns the number of CUDA devices (stub).
func GetDeviceCount() (int, error) {
	return 2, nil // Stub: return 2 GPUs for testing
}

// SetDevice sets the current CUDA device (stub).
func SetDevice(deviceID int) error {
	return nil // Stub: always succeeds
}

// GetDeviceInfo returns information about the current device (stub).
func GetDeviceInfo() (*DeviceInfo, error) {
	return &DeviceInfo{
		Name:        "Stub GPU",
		Major:       8,
		Minor:       0,
		TotalMemory: 8 * 1024 * 1024 * 1024, // 8 GB
	}, nil
}

// SyncDevice synchronizes the current device (stub).
func SyncDevice() error {
	return nil
}

// GetMemoryUsed returns current GPU memory usage (stub).
func GetMemoryUsed() (uint64, error) {
	return 0, nil
}

// =============================================================================
// Multi-GPU Operations
// =============================================================================

// InitMultiGPU initializes multi-GPU context with specified devices.
func InitMultiGPU(deviceIDs []int) error {
	stubState.mu.Lock()
	defer stubState.mu.Unlock()

	if len(deviceIDs) == 0 {
		return errors.New("no devices specified")
	}

	// Initialize state
	stubState.initialized = true
	stubState.deviceIDs = make([]int, len(deviceIDs))
	copy(stubState.deviceIDs, deviceIDs)
	stubState.deviceContexts = make(map[int]*DeviceContext)
	stubState.allocations = make(map[uintptr]*stubAllocation)
	stubState.nextAllocAddr = 0x10000000 // Start address for simulated allocations
	stubState.stagingBufSize = 64 * 1024 * 1024 // 64 MB

	// Create P2P matrix (all devices can access each other in stub)
	numDevices := len(deviceIDs)
	stubState.p2pMatrix = make([][]bool, numDevices)
	for i := 0; i < numDevices; i++ {
		stubState.p2pMatrix[i] = make([]bool, numDevices)
		for j := 0; j < numDevices; j++ {
			stubState.p2pMatrix[i][j] = (i != j) // P2P available for different devices
		}
	}

	// Create device contexts
	for idx, devID := range deviceIDs {
		peerAccess := make([]bool, numDevices)
		for j := 0; j < numDevices; j++ {
			peerAccess[j] = stubState.p2pMatrix[idx][j]
		}
		stubState.deviceContexts[devID] = &DeviceContext{
			DeviceID:       devID,
			TotalMemory:    8 * 1024 * 1024 * 1024, // 8 GB each
			UsedMemory:     0,
			ComputeStream:  unsafe.Pointer(uintptr(0x1000 + idx*0x100)),
			TransferStream: unsafe.Pointer(uintptr(0x2000 + idx*0x100)),
			PeerAccess:     peerAccess,
		}
	}

	return nil
}

// ShutdownMultiGPU releases multi-GPU resources.
func ShutdownMultiGPU() error {
	stubState.mu.Lock()
	defer stubState.mu.Unlock()

	stubState.initialized = false
	stubState.deviceIDs = nil
	stubState.deviceContexts = nil
	stubState.p2pMatrix = nil
	stubState.allocations = nil

	return nil
}

// GetDeviceContext returns the context for a specific device.
func GetDeviceContext(deviceID int) (*DeviceContext, error) {
	stubState.mu.RLock()
	defer stubState.mu.RUnlock()

	if !stubState.initialized {
		return nil, ErrMultiDeviceNotInitialized
	}

	ctx, ok := stubState.deviceContexts[deviceID]
	if !ok {
		return nil, errors.New("device not found in multi-device context")
	}

	// Return a copy
	result := &DeviceContext{
		DeviceID:       ctx.DeviceID,
		TotalMemory:    ctx.TotalMemory,
		UsedMemory:     ctx.UsedMemory,
		ComputeStream:  ctx.ComputeStream,
		TransferStream: ctx.TransferStream,
		PeerAccess:     make([]bool, len(ctx.PeerAccess)),
	}
	copy(result.PeerAccess, ctx.PeerAccess)

	return result, nil
}

// GetP2PAccessMatrix returns the peer-to-peer access matrix.
func GetP2PAccessMatrix() ([][]bool, error) {
	stubState.mu.RLock()
	defer stubState.mu.RUnlock()

	if !stubState.initialized {
		return nil, ErrMultiDeviceNotInitialized
	}

	// Return a copy
	result := make([][]bool, len(stubState.p2pMatrix))
	for i := range stubState.p2pMatrix {
		result[i] = make([]bool, len(stubState.p2pMatrix[i]))
		copy(result[i], stubState.p2pMatrix[i])
	}

	return result, nil
}

// AllocOnDevice allocates memory on a specific device.
func AllocOnDevice(size uint64, deviceID int) (unsafe.Pointer, error) {
	stubState.mu.Lock()
	defer stubState.mu.Unlock()

	if !stubState.initialized {
		return nil, ErrMultiDeviceNotInitialized
	}

	ctx, ok := stubState.deviceContexts[deviceID]
	if !ok {
		return nil, errors.New("device not found")
	}

	// Actually allocate memory for the stub
	addr := stubState.nextAllocAddr
	stubState.nextAllocAddr += uintptr(size) + 0x1000 // Add padding

	alloc := &stubAllocation{
		deviceID: deviceID,
		data:     make([]byte, size),
		size:     size,
	}
	stubState.allocations[addr] = alloc
	ctx.UsedMemory += size

	return unsafe.Pointer(addr), nil
}

// FreeOnDevice frees memory on a specific device.
func FreeOnDevice(ptr unsafe.Pointer, deviceID int) error {
	stubState.mu.Lock()
	defer stubState.mu.Unlock()

	if !stubState.initialized {
		return ErrMultiDeviceNotInitialized
	}

	addr := uintptr(ptr)
	if alloc, ok := stubState.allocations[addr]; ok {
		if alloc.deviceID != deviceID {
			return errors.New("device mismatch for allocation")
		}
		delete(stubState.allocations, addr)
	}

	return nil
}

// CrossDeviceCopy copies data between devices.
func CrossDeviceCopy(dst unsafe.Pointer, dstDevice int, src unsafe.Pointer, srcDevice int, size uint64) error {
	stubState.mu.Lock()
	defer stubState.mu.Unlock()

	if !stubState.initialized {
		return ErrMultiDeviceNotInitialized
	}

	// Validate devices exist
	if _, ok := stubState.deviceContexts[srcDevice]; !ok {
		return errors.New("source device not found")
	}
	if _, ok := stubState.deviceContexts[dstDevice]; !ok {
		return errors.New("destination device not found")
	}

	// Find source and destination allocations
	srcAlloc, srcOk := stubState.allocations[uintptr(src)]
	dstAlloc, dstOk := stubState.allocations[uintptr(dst)]

	if !srcOk || !dstOk {
		return errors.New("invalid allocation pointer")
	}

	// Verify sizes
	if size > srcAlloc.size || size > dstAlloc.size {
		return errors.New("copy size exceeds allocation")
	}

	// Actually copy the data
	copy(dstAlloc.data[:size], srcAlloc.data[:size])

	return nil
}

// GetMultiDeviceManagerInfo returns information about the multi-device manager.
func GetMultiDeviceManagerInfo() (*MultiDeviceManagerInfo, error) {
	stubState.mu.RLock()
	defer stubState.mu.RUnlock()

	if !stubState.initialized {
		return &MultiDeviceManagerInfo{
			Initialized: false,
		}, nil
	}

	return &MultiDeviceManagerInfo{
		NumDevices:        len(stubState.deviceIDs),
		DeviceIDs:         append([]int{}, stubState.deviceIDs...),
		StagingBufferSize: stubState.stagingBufSize,
		Initialized:       true,
	}, nil
}

// CanAccessPeer checks if one device can access another via P2P.
func CanAccessPeer(srcDevice, dstDevice int) (bool, error) {
	stubState.mu.RLock()
	defer stubState.mu.RUnlock()

	if !stubState.initialized {
		return false, ErrMultiDeviceNotInitialized
	}

	srcIdx := stubState.findDeviceIndex(srcDevice)
	dstIdx := stubState.findDeviceIndex(dstDevice)

	if srcIdx == -1 || dstIdx == -1 {
		return false, errors.New("device not found in multi-device context")
	}

	return stubState.p2pMatrix[srcIdx][dstIdx], nil
}

// =============================================================================
// Memory Management (Stubs)
// =============================================================================

// Malloc allocates GPU memory (stub).
func Malloc(size uint64) (unsafe.Pointer, error) {
	return unsafe.Pointer(uintptr(0xDEADBEEF)), nil
}

// Free releases GPU memory (stub).
func Free(ptr unsafe.Pointer) {
	// Stub: no-op
}

// CopyToDevice copies data from host to device (stub).
func CopyToDevice(dst, src unsafe.Pointer, numElements int, dtype types.Dtype) error {
	return nil
}

// CopyToHost copies data from device to host (stub).
func CopyToHost(dst, src unsafe.Pointer, numElements int, dtype types.Dtype) error {
	return nil
}

// CopyToDeviceRaw copies raw bytes from host to device (stub).
func CopyToDeviceRaw(dst, src unsafe.Pointer, size uint64) error {
	stubState.mu.Lock()
	defer stubState.mu.Unlock()

	// Find the destination allocation
	dstAlloc, ok := stubState.allocations[uintptr(dst)]
	if !ok {
		return errors.New("destination not a valid device allocation")
	}

	if size > dstAlloc.size {
		return errors.New("copy size exceeds allocation")
	}

	// Copy from host memory (src) to our simulated device memory
	srcSlice := unsafe.Slice((*byte)(src), size)
	copy(dstAlloc.data[:size], srcSlice)

	return nil
}

// CopyFromDeviceRaw copies raw bytes from device to host (stub).
func CopyFromDeviceRaw(dst, src unsafe.Pointer, size uint64) error {
	stubState.mu.RLock()
	defer stubState.mu.RUnlock()

	// Find the source allocation
	srcAlloc, ok := stubState.allocations[uintptr(src)]
	if !ok {
		return errors.New("source not a valid device allocation")
	}

	if size > srcAlloc.size {
		return errors.New("copy size exceeds allocation")
	}

	// Copy from our simulated device memory to host memory (dst)
	dstSlice := unsafe.Slice((*byte)(dst), size)
	copy(dstSlice, srcAlloc.data[:size])

	return nil
}

// CopyInt8ToDevice copies INT8 data to device (stub).
func CopyInt8ToDevice(dst, src unsafe.Pointer, numElements int) error {
	return nil
}

// CopyInt8ToHost copies INT8 data from device (stub).
func CopyInt8ToHost(dst, src unsafe.Pointer, numElements int) error {
	return nil
}

// =============================================================================
// Kernel Operations (Stubs)
// =============================================================================

// RMSNorm performs RMSNorm operation (stub).
func RMSNorm(output, input, weight unsafe.Pointer, numTokens, hiddenDim int, eps float32) error {
	return nil
}

// SiLU performs SiLU activation (stub).
func SiLU(output, input unsafe.Pointer, numElements int) error {
	return nil
}

// Add performs element-wise addition (stub).
func Add(c, a, b unsafe.Pointer, numElements int) error {
	return nil
}

// RoPE applies rotary position embeddings (stub).
func RoPE(output, input unsafe.Pointer, positions []int32, batchSize, seqLen, numHeads, headDim int) error {
	return nil
}

// GEMMFP16 performs FP16 matrix multiplication (stub).
// Takes Tensor arguments to match the CUDA bindings signature.
func GEMMFP16(c, a, b *types.Tensor, transposeA, transposeB bool) error {
	return nil
}

// GEMMINT8 performs INT8 matrix multiplication (stub).
// transposeB: if true, B is stored as [N, K] and will be transposed.
func GEMMINT8(c, a, b, scale *types.Tensor, transposeB bool) error {
	return nil
}

// QuantizePerTensor quantizes FP16 to INT8 (stub).
func QuantizePerTensor(output, scale, input unsafe.Pointer, numElements int) error {
	return nil
}

// DequantizePerTensor dequantizes INT8 to FP16 (stub).
func DequantizePerTensor(output, input, scale unsafe.Pointer, numElements int) error {
	return nil
}

// QuantizeWeights quantizes weights per-column (stub).
func QuantizeWeights(output, scales, input unsafe.Pointer, K, N int) error {
	return nil
}

// BasicAttention performs basic attention (stub).
func BasicAttention(output, query, key, value unsafe.Pointer, batchSize, numHeads, seqLen, headDim int, causal bool) error {
	return nil
}

// =============================================================================
// KV Cache (Stubs)
// =============================================================================

// KVCache holds the key-value cache for attention (stub).
type KVCache struct {
	ptr    unsafe.Pointer
	length int
}
func (c *KVCache) Ptr() unsafe.Pointer { return c.ptr }

// NewKVCache creates a new KV cache (stub).
func NewKVCache(batchSize, numHeads, headDim, maxSeqLen int) (*KVCache, error) {
	return &KVCache{ptr: unsafe.Pointer(uintptr(0xCAFEBABE)), length: 0}, nil
}

// FreeKVCache frees KV cache (stub).
func FreeKVCache(cache *KVCache) {
	if cache != nil {
		cache.ptr = nil
	}
}

// UpdateKVCache updates KV cache (stub).
func UpdateKVCache(cache *KVCache, k, v []float32, position int) error {
	if cache != nil && position >= cache.length {
		cache.length = position + 1
	}
	return nil
}

// GetKVCacheLength returns cache length (stub).
func GetKVCacheLength(cache *KVCache) int {
	if cache == nil {
		return 0
	}
	return cache.length
}

// AttentionWithKVCache performs attention with KV cache (stub).
func AttentionWithKVCache(output, q, k, v *types.Tensor, cache *KVCache, position int) error {
	return nil
}

// =============================================================================
// Layer Operations (Stubs)
// =============================================================================

// LayerWeights holds all weights for a transformer layer (stub).
type LayerWeights struct {
	ptr unsafe.Pointer
}

// LoadLayerWeights loads layer weights (stub).
func LoadLayerWeights(path string) (*LayerWeights, error) {
	return &LayerWeights{ptr: unsafe.Pointer(uintptr(0xBAADF00D))}, nil
}

// FreeLayerWeights frees layer weights (stub).
func FreeLayerWeights(weights *LayerWeights) {
	if weights != nil {
		weights.ptr = nil
	}
}

// CreateRandomLayerWeights creates random weights for testing (stub).
func CreateRandomLayerWeights(config *types.LlamaConfig) (*LayerWeights, error) {
	return &LayerWeights{ptr: unsafe.Pointer(uintptr(0xFEEDFACE))}, nil
}

// CreateLayerWeightsFromHost creates layer weights from FP16 host data (stub).
func CreateLayerWeightsFromHost(
	qProj, kProj, vProj, oProj []byte,
	gateProj, upProj, downProj []byte,
	attnNorm, ffnNorm []byte,
	config *types.LlamaConfig,
) (*LayerWeights, error) {
	return &LayerWeights{ptr: unsafe.Pointer(uintptr(0xFEEDFACE))}, nil
}

// LayerForward performs a complete layer forward pass (stub).
func LayerForward(output, input *types.Tensor, weights *LayerWeights, cache *KVCache, positions []int32, config *types.LlamaConfig) error {
	return nil
}

// =============================================================================
// FP16-Pure Layer (Stubs) — for LFM2 attention layers
// =============================================================================

// LayerWeightsFP16 holds FP16-pure weights (stub).
type LayerWeightsFP16 struct {
	ptr unsafe.Pointer
}
func (w *LayerWeightsFP16) Ptr() unsafe.Pointer { return w.ptr }

// CreateLayerWeightsFromHostFP16 creates FP16-pure layer weights (stub).
func CreateLayerWeightsFromHostFP16(
	qProj, kProj, vProj, oProj []byte,
	gateProj, upProj, downProj []byte,
	attnNorm, ffnNorm []byte,
	qLayerNorm, kLayerNorm []byte,
	config *types.LlamaConfig,
) (*LayerWeightsFP16, error) {
	return nil, ErrNotImplemented
}

// FreeLayerWeightsFP16 frees FP16-pure layer weights (stub).
func FreeLayerWeightsFP16(w *LayerWeightsFP16) {}

// LayerForwardFP16 executes FP16-pure layer forward (stub).
func LayerForwardFP16(output, input *types.Tensor, weights *LayerWeightsFP16, cache *KVCache,
	positions []int32, config *types.LlamaConfig, ropeStyle int) error {
	return ErrNotImplemented
}

// LayerWorkspaceFP16 holds pre-allocated GPU buffers (stub).
type LayerWorkspaceFP16 struct {
	ptr unsafe.Pointer
}

// CreateLayerWorkspaceFP16 pre-allocates workspace buffers (stub).
func CreateLayerWorkspaceFP16(maxTokens int, config *types.LlamaConfig) (*LayerWorkspaceFP16, error) {
	return nil, ErrNotImplemented
}

// FreeLayerWorkspaceFP16 frees workspace buffers (stub).
func FreeLayerWorkspaceFP16(ws *LayerWorkspaceFP16) {}

// LayerForwardFP16WithWorkspace executes FP16 layer forward with workspace (stub).
func LayerForwardFP16WithWorkspace(output, input *types.Tensor, weights *LayerWeightsFP16, cache *KVCache,
	positions []int32, config *types.LlamaConfig, ropeStyle int, workspace *LayerWorkspaceFP16) error {
	return ErrNotImplemented
}

// =============================================================================
// Paged KV Cache (Stubs)
// =============================================================================

type PagedKVCache struct{ ptr unsafe.Pointer }

func CreatePagedKVCache(numBlocks, numKVHeads, headDim, blockSize int) (*PagedKVCache, error) {
	return nil, ErrNotImplemented
}
func FreePagedKVCache(cache *PagedKVCache) {}
func PagedAttention(output, query, newKey, newValue unsafe.Pointer, cache *PagedKVCache,
	dBlockTable unsafe.Pointer, numHeads, numKVHeads, headDim, position int) error {
	return ErrNotImplemented
}
func (c *PagedKVCache) Ptr() unsafe.Pointer { return nil }

// LayerForwardFP16Paged executes FP16 layer forward with paged attention (stub).
func LayerForwardFP16Paged(output, input *types.Tensor, weights *LayerWeightsFP16,
	pagedCache *PagedKVCache, dBlockTable unsafe.Pointer,
	positions []int32, config *types.LlamaConfig, ropeStyle int,
	workspace *LayerWorkspaceFP16) error {
	return ErrNotImplemented
}

// =============================================================================
// Full Decode Context (Stubs)
// =============================================================================

type DecodeContext struct{ ptr unsafe.Pointer }

func CreateDecodeContext(config *types.LlamaConfig) (*DecodeContext, error) {
	return nil, ErrNotImplemented
}
func SetDecodeLayer(ctx *DecodeContext, layerID, layerType int, weights, cache unsafe.Pointer) {}
func SetDecodeWorkspace(ctx *DecodeContext, workspace *LayerWorkspaceFP16) {}

type ConvWorkspace struct{ ptr unsafe.Pointer }

func CreateConvWorkspace(hiddenSize, intermediateSize int) (*ConvWorkspace, error) {
	return nil, ErrNotImplemented
}
func FreeConvWorkspace(ws *ConvWorkspace) {}
func SetDecodeConvWorkspace(ctx *DecodeContext, ws *ConvWorkspace) {}
func SetDecodePagedCache(ctx *DecodeContext, pagedCache *PagedKVCache, dBlockTable unsafe.Pointer, maxBlocksPerSeq int) {}
func SetDecodePagedLayer(ctx *DecodeContext, layerID int, pagedCache *PagedKVCache) {}
func DecodeStep(ctx *DecodeContext, output, input []byte, position int) error {
	return ErrNotImplemented
}
func DecodeSetHiddenFromGPU(ctx *DecodeContext, gpuPtr unsafe.Pointer) error { return ErrNotImplemented }
func DecodeSetHidden(ctx *DecodeContext, hidden []byte) error                 { return ErrNotImplemented }
func DecodeGetHidden(ctx *DecodeContext, hidden []byte) error { return ErrNotImplemented }
func DecodeStepGPU(ctx *DecodeContext, position int) error     { return ErrNotImplemented }
func DecodeGetHiddenGPUPtr(ctx *DecodeContext) unsafe.Pointer  { return nil }
func SetDecodeBF16Native(ctx *DecodeContext, workspace *LayerWorkspaceBF16) error {
	return ErrNotImplemented
}
func DecodeSetHiddenBF16(ctx *DecodeContext, hidden []byte) error       { return ErrNotImplemented }
func DecodeGetHiddenBF16(ctx *DecodeContext, hidden []byte) error       { return ErrNotImplemented }
func DecodeSetHiddenBF16FromGPU(ctx *DecodeContext, gpuPtr unsafe.Pointer) error {
	return ErrNotImplemented
}
func DecodeGetHiddenBF16GPUPtr(ctx *DecodeContext) unsafe.Pointer { return nil }
func DecodeConvertFP16ToBF16(ctx *DecodeContext) error            { return ErrNotImplemented }
func DecodeConvertBF16ToFP16(ctx *DecodeContext) error            { return ErrNotImplemented }
func PrefillBatch(ctx *DecodeContext, dInput, dOutput unsafe.Pointer, dPositions, dSlotMapping unsafe.Pointer, numTokens int) error {
	return ErrNotImplemented
}
func GatherEmbeddings(output, embedTable unsafe.Pointer, dTokenIDs unsafe.Pointer, hiddenSize, numTokens int) error {
	return ErrNotImplemented
}
func DecodeStepBatched(ctx *DecodeContext, dEmbeddings, dOutput unsafe.Pointer, dPositions, dSeqLens, dBlockTables unsafe.Pointer, dConvStates unsafe.Pointer, batchSize int) error {
	return ErrNotImplemented
}
func DecodeInvalidateGraph(ctx *DecodeContext) {}
func FreeDecodeContext(ctx *DecodeContext)     {}

// =============================================================================
// LFM2 / BF16 Operations (Stubs)
// =============================================================================

// ConvLayerWeights holds weights for an LFM2 conv layer (stub).
type ConvLayerWeights struct {
	ptr unsafe.Pointer
}
func (w *ConvLayerWeights) Ptr() unsafe.Pointer { return w.ptr }

// CheckBF16Support returns false for stub builds.
func CheckBF16Support() (bool, error) {
	return false, ErrNotImplemented
}

// FP16ToBF16 converts FP16 to BF16 (stub).
func FP16ToBF16(output, input unsafe.Pointer, numElements int) error {
	return ErrNotImplemented
}

// BF16ToFP16 converts BF16 to FP16 (stub).
func BF16ToFP16(output, input unsafe.Pointer, numElements int) error {
	return ErrNotImplemented
}

// GEMMBF16 performs BF16 matrix multiplication (stub).
func GEMMBF16(cPtr, aPtr, bPtr unsafe.Pointer, M, K, N int, transposeA, transposeB bool) error {
	return ErrNotImplemented
}

// ConvStateCreate creates a conv state buffer (stub).
func ConvStateCreate(batch, dim, width int) (unsafe.Pointer, error) {
	return nil, ErrNotImplemented
}

// ConvStateReset zeros a conv state buffer (stub).
func ConvStateReset(state unsafe.Pointer, batch, dim, width int) error {
	return ErrNotImplemented
}

// ConvStateFree frees a conv state buffer (stub).
func ConvStateFree(state unsafe.Pointer) {}

// CreateConvLayerWeightsBF16 creates conv layer weights (stub).
func CreateConvLayerWeightsBF16(
	inProj, conv, outProj []byte,
	opNorm, ffnNorm []byte,
	gate, up, down []byte,
	hidden, intermediate, kernelSize int, normEps float32,
) (*ConvLayerWeights, error) {
	return nil, ErrNotImplemented
}

// ConvLayerForwardBF16 executes a conv layer forward pass (stub).
func ConvLayerForwardBF16(output, input unsafe.Pointer, weights *ConvLayerWeights, convState unsafe.Pointer, batch, seqLen, position int) error {
	return ErrNotImplemented
}

// FreeConvLayerWeights releases conv layer weights (stub).
func FreeConvLayerWeights(w *ConvLayerWeights) {
	if w != nil {
		w.ptr = nil
	}
}

// =============================================================================
// BF16-Native Attention Layer (Stubs)
// =============================================================================

// LayerWeightsBF16 holds BF16-native weights (stub).
type LayerWeightsBF16 struct {
	ptr unsafe.Pointer
}

func (w *LayerWeightsBF16) Ptr() unsafe.Pointer { return w.ptr }

// CreateLayerWeightsBF16Native creates BF16-native layer weights (stub).
func CreateLayerWeightsBF16Native(
	qProj, kProj, vProj, oProj []byte,
	gateProj, upProj, downProj []byte,
	attnNorm, ffnNorm []byte,
	qLayerNorm, kLayerNorm []byte,
	config *types.LlamaConfig,
) (*LayerWeightsBF16, error) {
	return nil, ErrNotImplemented
}

// FreeLayerWeightsBF16Native frees BF16-native layer weights (stub).
func FreeLayerWeightsBF16Native(w *LayerWeightsBF16) {}

// LayerWorkspaceBF16 holds pre-allocated BF16 workspace (stub).
type LayerWorkspaceBF16 struct {
	ptr unsafe.Pointer
}

// CreateLayerWorkspaceBF16 pre-allocates BF16 workspace buffers (stub).
func CreateLayerWorkspaceBF16(maxTokens int, config *types.LlamaConfig) (*LayerWorkspaceBF16, error) {
	return nil, ErrNotImplemented
}

// FreeLayerWorkspaceBF16 frees BF16 workspace buffers (stub).
func FreeLayerWorkspaceBF16(ws *LayerWorkspaceBF16) {}

// LayerForwardBF16Native executes BF16-native layer forward (stub).
func LayerForwardBF16Native(output, input *types.Tensor, weights *LayerWeightsBF16, cache *KVCache,
	positions []int32, config *types.LlamaConfig, ropeStyle int, workspace *LayerWorkspaceBF16) error {
	return ErrNotImplemented
}

// LayerForwardBF16Paged executes BF16-native layer forward with paged attention (stub).
func LayerForwardBF16Paged(output, input *types.Tensor, weights *LayerWeightsBF16,
	pagedCache *PagedKVCache, dBlockTable unsafe.Pointer,
	positions []int32, config *types.LlamaConfig, ropeStyle int,
	workspace *LayerWorkspaceBF16) error {
	return ErrNotImplemented
}

// =============================================================================
// Tensor Operations (Stubs)
// =============================================================================

// AllocateTensor allocates GPU memory for a tensor (stub).
func AllocateTensor(t *types.Tensor) error {
	if t == nil {
		return errors.New("tensor is nil")
	}
	t.Data = unsafe.Pointer(uintptr(0xDEADBEEF))
	return nil
}

// FreeTensor frees GPU memory for a tensor (stub).
func FreeTensor(t *types.Tensor) {
	if t != nil {
		t.Data = nil
	}
}
