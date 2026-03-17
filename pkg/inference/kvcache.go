// Package inference provides distributed inference capabilities for LLM generation.
package inference

import (
	"context"
	"fmt"
	"sync"
	"unsafe"
)

// KVCacheConfig holds configuration for a KV cache.
type KVCacheConfig struct {
	LayerID    int
	NumKVHeads int
	HeadDim    int
	MaxSeqLen  int
}

// DistributedKVCache manages KV cache for a single layer, which may be local or remote.
type DistributedKVCache struct {
	config     KVCacheConfig
	peerID     string
	deviceID   int
	currentLen int
	isLocal    bool

	// Local cache data (nil if remote)
	keys   []byte // [maxSeqLen, numKVHeads, headDim] in FP16
	values []byte // [maxSeqLen, numKVHeads, headDim] in FP16

	mu sync.RWMutex
}

// NewDistributedKVCache creates a new distributed KV cache.
func NewDistributedKVCache(config KVCacheConfig, peerID string, deviceID int, isLocal bool) *DistributedKVCache {
	cache := &DistributedKVCache{
		config:   config,
		peerID:   peerID,
		deviceID: deviceID,
		isLocal:  isLocal,
	}

	if isLocal {
		// Allocate cache memory for local cache
		// Each position stores numKVHeads * headDim * 2 bytes (FP16) for K and V
		cacheSize := config.MaxSeqLen * config.NumKVHeads * config.HeadDim * 2 // 2 bytes per FP16
		cache.keys = make([]byte, cacheSize)
		cache.values = make([]byte, cacheSize)
	}

	return cache
}

// Update updates the KV cache at the given position.
// For local caches, this writes directly. For remote caches, this is a no-op
// as the remote worker handles its own cache updates during forward pass.
func (c *DistributedKVCache) Update(ctx context.Context, k, v []byte, position int) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if !c.isLocal {
		// Remote cache updates are handled by the remote worker
		return nil
	}

	// Validate position
	if position < 0 || position >= c.config.MaxSeqLen {
		return fmt.Errorf("position %d out of range [0, %d)", position, c.config.MaxSeqLen)
	}

	// Calculate byte offset for this position
	posSize := c.config.NumKVHeads * c.config.HeadDim * 2 // bytes per position
	offset := position * posSize

	// Validate input sizes
	if len(k) != posSize {
		return fmt.Errorf("key size mismatch: got %d, expected %d", len(k), posSize)
	}
	if len(v) != posSize {
		return fmt.Errorf("value size mismatch: got %d, expected %d", len(v), posSize)
	}

	// Copy K and V to cache
	copy(c.keys[offset:offset+posSize], k)
	copy(c.values[offset:offset+posSize], v)

	// Update current length if extending
	if position >= c.currentLen {
		c.currentLen = position + 1
	}

	return nil
}

// Get retrieves cached K and V values up to the specified length.
func (c *DistributedKVCache) Get(ctx context.Context, upToLen int) (keys, values []byte, err error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	if !c.isLocal {
		return nil, nil, fmt.Errorf("cannot get remote cache locally")
	}

	if upToLen > c.currentLen {
		upToLen = c.currentLen
	}

	posSize := c.config.NumKVHeads * c.config.HeadDim * 2
	totalSize := upToLen * posSize

	return c.keys[:totalSize], c.values[:totalSize], nil
}

// CurrentLength returns the current number of cached positions.
func (c *DistributedKVCache) CurrentLength() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.currentLen
}

// Clear resets the cache for a new sequence.
func (c *DistributedKVCache) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.currentLen = 0
	// Note: we don't zero the backing arrays for performance,
	// the currentLen ensures we don't read stale data
}

// LayerID returns the layer ID this cache belongs to.
func (c *DistributedKVCache) LayerID() int {
	return c.config.LayerID
}

// PeerID returns the peer ID where this cache resides.
func (c *DistributedKVCache) PeerID() string {
	return c.peerID
}

// IsLocal returns whether this cache is local or remote.
func (c *DistributedKVCache) IsLocal() bool {
	return c.isLocal
}

// Config returns the cache configuration.
func (c *DistributedKVCache) Config() KVCacheConfig {
	return c.config
}

// KVCacheManager manages KV caches across all layers.
type KVCacheManager struct {
	caches map[int]*DistributedKVCache // layerID -> cache
	mu     sync.RWMutex
}

// NewKVCacheManager creates a new KV cache manager.
func NewKVCacheManager() *KVCacheManager {
	return &KVCacheManager{
		caches: make(map[int]*DistributedKVCache),
	}
}

// RegisterCache registers a KV cache for a layer.
func (m *KVCacheManager) RegisterCache(cache *DistributedKVCache) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.caches[cache.LayerID()] = cache
}

// GetCache returns the KV cache for a layer.
func (m *KVCacheManager) GetCache(layerID int) (*DistributedKVCache, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	cache, ok := m.caches[layerID]
	return cache, ok
}

// ClearAll clears all caches for a new sequence.
func (m *KVCacheManager) ClearAll() {
	m.mu.RLock()
	defer m.mu.RUnlock()

	for _, cache := range m.caches {
		cache.Clear()
	}
}

// LocalCaches returns all local KV caches.
func (m *KVCacheManager) LocalCaches() []*DistributedKVCache {
	m.mu.RLock()
	defer m.mu.RUnlock()

	var local []*DistributedKVCache
	for _, cache := range m.caches {
		if cache.IsLocal() {
			local = append(local, cache)
		}
	}
	return local
}

// CacheCount returns the total number of registered caches.
func (m *KVCacheManager) CacheCount() int {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return len(m.caches)
}

// ConvStateCache manages conv state for a single LFM2 conv layer.
type ConvStateCache struct {
	layerID    int
	hiddenSize int
	kernelSize int
	gpuPtr     unsafe.Pointer // GPU FP32 buffer [batch, hidden, kernel]
}

// NewConvStateCache creates a new conv state cache for a conv layer.
func NewConvStateCache(layerID, hiddenSize, kernelSize int) *ConvStateCache {
	return &ConvStateCache{
		layerID:    layerID,
		hiddenSize: hiddenSize,
		kernelSize: kernelSize,
	}
}

// GPUPtr returns the GPU pointer for the conv state.
func (c *ConvStateCache) GPUPtr() unsafe.Pointer {
	return c.gpuPtr
}

// SetGPUPtr sets the GPU pointer (created by bindings.ConvStateCreate).
func (c *ConvStateCache) SetGPUPtr(ptr unsafe.Pointer) {
	c.gpuPtr = ptr
}

// HybridCacheManager manages both KV caches (attention) and conv state caches (conv).
type HybridCacheManager struct {
	kvManager   *KVCacheManager
	convCaches  map[int]*ConvStateCache // layerID -> conv state
	mu          sync.RWMutex
}

// NewHybridCacheManager creates a cache manager that handles both cache types.
func NewHybridCacheManager() *HybridCacheManager {
	return &HybridCacheManager{
		kvManager:  NewKVCacheManager(),
		convCaches: make(map[int]*ConvStateCache),
	}
}

// KVManager returns the underlying KV cache manager.
func (h *HybridCacheManager) KVManager() *KVCacheManager {
	return h.kvManager
}

// RegisterConvCache registers a conv state cache for a layer.
func (h *HybridCacheManager) RegisterConvCache(cache *ConvStateCache) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.convCaches[cache.layerID] = cache
}

// GetConvCache returns the conv state cache for a layer.
func (h *HybridCacheManager) GetConvCache(layerID int) (*ConvStateCache, bool) {
	h.mu.RLock()
	defer h.mu.RUnlock()
	cache, ok := h.convCaches[layerID]
	return cache, ok
}

// ClearAll clears all caches (both KV and conv state).
func (h *HybridCacheManager) ClearAll() {
	h.kvManager.ClearAll()
	// Conv state reset is handled by the CUDA executor (needs GPU calls)
}
