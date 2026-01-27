//go:build cuda

// Package bindings provides CGO bindings for CUDA operations.
// This file implements pinned memory buffer pool for DMA transfers.
//
// Pinned memory (page-locked host memory) enables Direct Memory Access (DMA)
// between host and GPU, significantly improving transfer throughput for
// activation data in distributed inference workloads.
package bindings

import (
	"fmt"
	"sync"
	"unsafe"
)

// Default buffer pool configuration constants.
const (
	// DefaultPinnedBufferSize is the default size for pinned buffers (16KB).
	// Suitable for typical activation sizes: Llama 7B (8KB), Llama 13B (10KB).
	DefaultPinnedBufferSize = 16 * 1024

	// DefaultPinnedPoolCount is the default number of buffers in the pool.
	// 32 buffers allow concurrent handling of multiple requests.
	DefaultPinnedPoolCount = 32
)

// PinnedBuffer represents a CUDA-registered host memory buffer.
// The buffer is page-locked and suitable for DMA transfers to/from GPU.
type PinnedBuffer struct {
	Ptr  unsafe.Pointer // Pointer to pinned memory
	Size uint64         // Size of the buffer in bytes
}

// AsSlice returns the buffer as a Go byte slice.
// WARNING: The slice is backed by pinned memory - do not use after Put().
// The slice is valid only while the buffer is held (between Get and Put).
func (b *PinnedBuffer) AsSlice() []byte {
	if b == nil || b.Ptr == nil {
		return nil
	}
	// Create slice header pointing to pinned memory
	return unsafe.Slice((*byte)(b.Ptr), b.Size)
}

// PinnedBufferPool manages reusable CUDA pinned memory buffers.
// Uses a channel-based pool for lock-free Get/Put operations.
//
// Usage pattern:
//
//	pool, _ := NewPinnedBufferPool(16*1024, 32) // 32 buffers of 16KB
//	defer pool.Close()
//
//	buf := pool.Get()
//	defer pool.Put(buf)
//	slice := buf.AsSlice()
//	// Use slice for DMA transfer...
type PinnedBufferPool struct {
	buffers chan *PinnedBuffer // Channel-based pool for lock-free access
	bufSize uint64             // Size of each buffer in bytes
	count   int                // Number of preallocated buffers
	mu      sync.Mutex         // Protects closed flag and stats
	closed  bool               // True if pool has been closed
}

// NewPinnedBufferPoolWithDefaults creates a pool using default configuration.
// Uses DefaultPinnedBufferSize (16KB) and DefaultPinnedPoolCount (32).
//
// Returns error if CUDA pinned memory allocation fails.
func NewPinnedBufferPoolWithDefaults() (*PinnedBufferPool, error) {
	return NewPinnedBufferPool(DefaultPinnedBufferSize, DefaultPinnedPoolCount)
}

// NewPinnedBufferPool creates a pool with preallocated CUDA pinned memory buffers.
//
// Parameters:
//   - bufSize: size of each buffer in bytes (e.g., 16*1024 for 16KB)
//   - count: number of buffers to preallocate (e.g., 32)
//
// Returns error if CUDA pinned memory allocation fails.
// On partial failure, all successfully allocated buffers are freed.
//
// Recommended sizes for LLM inference:
//   - 8KB (8192) for Llama 7B hidden states
//   - 10KB (10240) for Llama 13B hidden states
//   - 16KB (16384) for general use with headroom
//
// Example:
//
//	pool, err := NewPinnedBufferPool(16*1024, 32) // 32 buffers of 16KB
//	if err != nil {
//	    return err
//	}
//	defer pool.Close()
func NewPinnedBufferPool(bufSize uint64, count int) (*PinnedBufferPool, error) {
	if bufSize == 0 {
		return nil, fmt.Errorf("pinned buffer pool: buffer size cannot be zero")
	}
	if count <= 0 {
		return nil, fmt.Errorf("pinned buffer pool: count must be positive, got %d", count)
	}

	pool := &PinnedBufferPool{
		buffers: make(chan *PinnedBuffer, count),
		bufSize: bufSize,
		count:   count,
	}

	// Preallocate all buffers
	for i := 0; i < count; i++ {
		ptr, err := AllocPinnedMemory(bufSize)
		if err != nil {
			// Cleanup on failure - close pool to free already allocated buffers
			pool.Close()
			return nil, fmt.Errorf("pinned buffer pool: failed to allocate buffer %d of %d (size: %d bytes): %w", i+1, count, bufSize, err)
		}
		pool.buffers <- &PinnedBuffer{Ptr: ptr, Size: bufSize}
	}

	return pool, nil
}

// Get returns a buffer from the pool, or allocates a new one if exhausted.
// Returns nil only if allocation fails after pool exhaustion.
//
// The returned buffer must be returned via Put() when done.
// Do not use the buffer's AsSlice() after Put().
func (p *PinnedBufferPool) Get() *PinnedBuffer {
	if p == nil {
		return nil
	}

	p.mu.Lock()
	if p.closed {
		p.mu.Unlock()
		return nil
	}
	p.mu.Unlock()

	select {
	case buf := <-p.buffers:
		return buf
	default:
		// Pool exhausted - allocate new buffer (fallback)
		ptr, err := AllocPinnedMemory(p.bufSize)
		if err != nil {
			return nil // Caller must handle nil
		}
		return &PinnedBuffer{Ptr: ptr, Size: p.bufSize}
	}
}

// Put returns a buffer to the pool for reuse.
// If the pool is full, the buffer is freed to prevent memory accumulation.
// Safe to call with nil buffer.
func (p *PinnedBufferPool) Put(buf *PinnedBuffer) {
	if p == nil || buf == nil || buf.Ptr == nil {
		return
	}

	p.mu.Lock()
	if p.closed {
		p.mu.Unlock()
		// Pool closed - free the buffer
		FreePinnedMemory(buf.Ptr)
		return
	}
	p.mu.Unlock()

	select {
	case p.buffers <- buf:
		// Returned to pool successfully
	default:
		// Pool full - free the overflow buffer
		FreePinnedMemory(buf.Ptr)
	}
}

// Close frees all buffers in the pool and prevents further use.
// Safe to call multiple times (idempotent).
func (p *PinnedBufferPool) Close() error {
	if p == nil {
		return nil
	}

	p.mu.Lock()
	if p.closed {
		p.mu.Unlock()
		return nil
	}
	p.closed = true
	p.mu.Unlock()

	// Drain and free all buffers in the channel
	close(p.buffers)
	for buf := range p.buffers {
		if buf != nil && buf.Ptr != nil {
			FreePinnedMemory(buf.Ptr)
		}
	}

	return nil
}

// Stats returns pool statistics: (available, total).
func (p *PinnedBufferPool) Stats() (available, total int) {
	if p == nil {
		return 0, 0
	}
	p.mu.Lock()
	defer p.mu.Unlock()

	if p.closed {
		return 0, p.count
	}
	return len(p.buffers), p.count
}

// BufSize returns the size of each buffer in the pool.
func (p *PinnedBufferPool) BufSize() uint64 {
	if p == nil {
		return 0
	}
	return p.bufSize
}

// Count returns the number of buffers in the pool (capacity).
func (p *PinnedBufferPool) Count() int {
	if p == nil {
		return 0
	}
	return p.count
}
