// Package transport provides abstractions for activation transfer between peers.
// This file defines buffer pool interfaces and the default (non-CUDA) implementation.
package transport

import (
	"sync"

	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/peer"
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
// P2PTransport Buffer Pool Integration
// =============================================================================

// HasBufferPool returns whether the transport has a buffer pool configured.
func (t *P2PTransport) HasBufferPool() bool {
	return t.bufferPool != nil
}

// NewP2PTransportWithPool creates a P2P transport with buffer pool.
// This is a convenience function that combines NewP2PTransport with WithBufferPool option.
func NewP2PTransportWithPool(h host.Host, peerID peer.ID, pool BufferPool) *P2PTransport {
	return NewP2PTransport(h, peerID, WithBufferPool(pool))
}
