//go:build cuda

// Package transport — CUDA pinned memory buffer pool implementation.
// Requires CUDA runtime and gpu/bindings package.
package transport

import (
	"sync"
	"unsafe"

	"github.com/neurogrid/engine/gpu/bindings"
)

// =============================================================================
// PinnedTransportBufferPool - CUDA pinned memory wrapper for transport layer
// =============================================================================

// PinnedTransportBufferPool wraps bindings.PinnedBufferPool for transport use.
// Provides the BufferPool interface using CUDA pinned memory for DMA optimization.
//
// Thread-safe: all methods are safe for concurrent use.
type PinnedTransportBufferPool struct {
	pool    *bindings.PinnedBufferPool
	bufSize int
	mu      sync.RWMutex
	bufMap  map[uintptr]*bindings.PinnedBuffer // Maps slice data ptr to PinnedBuffer
	closed  bool
	stats   struct {
		misses int64
	}
}

// isClosed checks if the pool is closed (read-lock safe).
func (p *PinnedTransportBufferPool) isClosed() bool {
	p.mu.RLock()
	closed := p.closed
	p.mu.RUnlock()
	return closed
}

// NewPinnedBufferPool creates a transport buffer pool backed by CUDA pinned memory.
// Returns error if CUDA pinned memory allocation fails.
func NewPinnedBufferPool(bufferSize, poolSize int) (*PinnedTransportBufferPool, error) {
	pool, err := bindings.NewPinnedBufferPool(uint64(bufferSize), poolSize)
	if err != nil {
		return nil, err
	}

	return &PinnedTransportBufferPool{
		pool:    pool,
		bufSize: bufferSize,
		bufMap:  make(map[uintptr]*bindings.PinnedBuffer),
	}, nil
}

// Get returns a buffer from the pool.
// If the pool is exhausted, allocates a new pinned buffer (fallback).
// Returns nil only if the pool is closed or allocation completely fails.
func (p *PinnedTransportBufferPool) Get(size int) []byte {
	if p == nil || p.isClosed() {
		return nil
	}

	buf := p.pool.Get()
	if buf == nil {
		// Fallback to regular allocation
		return make([]byte, size)
	}

	slice := buf.AsSlice()
	if len(slice) < size {
		// Buffer too small - return to pool and allocate regular memory
		p.pool.Put(buf)
		return make([]byte, size)
	}

	// Track the mapping from slice pointer to PinnedBuffer for Put()
	p.mu.Lock()
	p.bufMap[uintptr(unsafe.Pointer(&slice[0]))] = buf
	p.mu.Unlock()

	return slice[:size]
}

// Put returns a buffer to the pool.
// Only returns pinned buffers to the pool; regular buffers are discarded.
// Safe to call with nil or empty buffer.
func (p *PinnedTransportBufferPool) Put(buf []byte) {
	if p == nil || len(buf) == 0 || p.isClosed() {
		return
	}

	// Look up the PinnedBuffer by the slice's data pointer
	ptr := uintptr(unsafe.Pointer(&buf[0]))

	p.mu.Lock()
	pinnedBuf, ok := p.bufMap[ptr]
	if ok {
		delete(p.bufMap, ptr)
	}
	p.mu.Unlock()

	if ok {
		p.pool.Put(pinnedBuf)
	}
	// Not a pinned buffer - was allocated as fallback, just let GC handle it
}

// Close releases all pool resources.
func (p *PinnedTransportBufferPool) Close() error {
	if p == nil {
		return nil
	}

	p.mu.Lock()
	if p.closed {
		p.mu.Unlock()
		return nil
	}
	p.closed = true
	// Clear the buffer map
	p.bufMap = make(map[uintptr]*bindings.PinnedBuffer)
	p.mu.Unlock()

	return p.pool.Close()
}

// Stats returns pool statistics.
func (p *PinnedTransportBufferPool) Stats() BufferPoolStats {
	if p == nil {
		return BufferPoolStats{}
	}

	p.mu.RLock()
	defer p.mu.RUnlock()

	available, total := p.pool.Stats()
	return BufferPoolStats{
		Capacity:  total,
		Available: available,
		Allocated: total - available,
		IsPinned:  true,
	}
}

// =============================================================================
// Worker Pool Initialization
// =============================================================================

// InitializeWorkerPool initializes a buffer pool for worker use.
// Attempts to use CUDA pinned memory if configured, falls back to regular memory.
func InitializeWorkerPool(config WorkerPoolConfig) (*PinnedTransportBufferPool, error) {
	if !config.UsePinnedMemory {
		return nil, nil // No pool requested
	}

	pool, err := NewPinnedBufferPool(config.BufferSize, config.PoolSize)
	if err != nil {
		if config.FallbackEnabled {
			// Return nil without error - caller should use DefaultBufferPool
			return nil, nil
		}
		return nil, err
	}

	return pool, nil
}
