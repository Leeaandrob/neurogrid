//go:build cuda

// Package transport provides abstractions for activation transfer between peers.
// This file implements buffer pool abstractions for transport layer optimization.
//
// Two implementations are provided:
//   - PinnedTransportBufferPool: uses CUDA pinned memory for DMA optimization
//   - DefaultBufferPool: uses sync.Pool for systems without CUDA
package transport

import (
	"sync"
	"unsafe"

	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/neurogrid/engine/gpu/bindings"
)

// Buffer pool configuration constants.
const (
	// DefaultTransportBufferSize is the default buffer size for transport (16KB).
	DefaultTransportBufferSize = 16 * 1024

	// DefaultTransportPoolSize is the default number of buffers in the pool.
	DefaultTransportPoolSize = 32
)

// BufferPool provides reusable byte buffers for transport operations.
// Implementations may use regular memory (DefaultBufferPool) or
// CUDA pinned memory (PinnedTransportBufferPool) for DMA optimization.
type BufferPool interface {
	// Get returns a buffer of at least the specified size.
	// The returned buffer may be larger than requested.
	Get(size int) []byte

	// Put returns a buffer to the pool for reuse.
	// The buffer may be reused by subsequent Get calls.
	Put(buf []byte)

	// Close releases all pooled resources.
	Close() error
}

// BufferPoolStats contains pool statistics.
type BufferPoolStats struct {
	Capacity  int  // Total number of buffers in pool
	Available int  // Number of buffers currently available
	Allocated int  // Number of buffers allocated (in use)
	Misses    int  // Number of times pool was exhausted
	IsPinned  bool // True if using CUDA pinned memory
}

// WorkerPoolConfig configures worker buffer pool initialization.
type WorkerPoolConfig struct {
	PoolSize        int  // Number of buffers to preallocate
	BufferSize      int  // Size of each buffer in bytes
	UsePinnedMemory bool // Use CUDA pinned memory if available
	FallbackEnabled bool // Fall back to regular memory on failure
}

// =============================================================================
// DefaultBufferPool - sync.Pool based fallback for non-CUDA systems
// =============================================================================

// DefaultBufferPool uses sync.Pool for regular memory buffers.
// Used as fallback when CUDA is not available or pinned allocation fails.
type DefaultBufferPool struct {
	pool sync.Pool
	size int
}

// NewDefaultBufferPool creates a sync.Pool-based buffer pool.
// This is a fallback for systems without CUDA support.
func NewDefaultBufferPool(size int) *DefaultBufferPool {
	return &DefaultBufferPool{
		size: size,
		pool: sync.Pool{
			New: func() interface{} {
				return make([]byte, size)
			},
		},
	}
}

// Get returns a buffer from the pool.
func (p *DefaultBufferPool) Get(size int) []byte {
	buf := p.pool.Get().([]byte)
	if len(buf) < size {
		// Buffer too small, allocate new one
		return make([]byte, size)
	}
	return buf[:size]
}

// Put returns a buffer to the pool.
func (p *DefaultBufferPool) Put(buf []byte) {
	if cap(buf) >= p.size {
		// Reset slice to full capacity before returning to pool
		p.pool.Put(buf[:cap(buf)])
	}
	// Small buffers are discarded
}

// Close releases all pool resources (no-op for sync.Pool).
func (p *DefaultBufferPool) Close() error {
	return nil
}

// =============================================================================
// PinnedTransportBufferPool - CUDA pinned memory wrapper for transport layer
// =============================================================================

// PinnedTransportBufferPool wraps bindings.PinnedBufferPool for transport use.
// Provides the BufferPool interface using CUDA pinned memory for DMA optimization.
//
// Thread-safe: all methods are safe for concurrent use.
type PinnedTransportBufferPool struct {
	pool     *bindings.PinnedBufferPool
	bufSize  int
	mu       sync.RWMutex
	bufMap   map[uintptr]*bindings.PinnedBuffer // Maps slice data ptr to PinnedBuffer
	closed   bool
	stats    struct {
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
// P2PTransport Buffer Pool Integration
// =============================================================================

// P2PTransportWithPool extends P2PTransport to include buffer pool support.
// This is used by the transport tests.

// HasBufferPool returns whether the transport has a buffer pool configured.
func (t *P2PTransport) HasBufferPool() bool {
	return t.bufferPool != nil
}

// NewP2PTransportWithPool creates a P2P transport with buffer pool.
// This is a convenience function that combines NewP2PTransport with WithBufferPool option.
func NewP2PTransportWithPool(h host.Host, peerID peer.ID, pool BufferPool) *P2PTransport {
	return NewP2PTransport(h, peerID, WithBufferPool(pool))
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
