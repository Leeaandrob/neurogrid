# API Documentation: Buffer Pool

## Overview

The Buffer Pool API provides reusable byte buffers for transport operations in distributed inference. Two implementations are available:

- **PinnedTransportBufferPool**: CUDA pinned memory for DMA optimization
- **DefaultBufferPool**: sync.Pool fallback for non-CUDA systems

This document covers both Phase 1 (core implementation) and Phase 2 (integration) APIs.

---

## Package: `pkg/transport`

### BufferPool Interface

```go
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
```

### BufferPoolStats

```go
// BufferPoolStats contains pool statistics.
type BufferPoolStats struct {
    Capacity  int  // Total number of buffers in pool
    Available int  // Number of buffers currently available
    Allocated int  // Number of buffers allocated (in use)
    Misses    int  // Number of times pool was exhausted
    IsPinned  bool // True if using CUDA pinned memory
}
```

### Constants

```go
const (
    // DefaultTransportBufferSize is the default buffer size (16KB).
    DefaultTransportBufferSize = 16 * 1024

    // DefaultTransportPoolSize is the default pool capacity (32).
    DefaultTransportPoolSize = 32
)
```

---

## P2PTransport (Phase 2)

The P2PTransport now supports buffer pool configuration for zero-allocation message handling.

### P2PTransportOption

```go
// P2PTransportOption is a functional option for configuring P2PTransport.
type P2PTransportOption func(*P2PTransport)
```

### WithBufferPool

```go
// WithBufferPool configures a buffer pool for zero-allocation message handling.
// When set, the transport will use pooled buffers instead of allocating new ones.
//
// For optimal DMA performance with CUDA, use a pinned memory buffer pool.
// The pool should have buffers sized for typical activation data (8KB-16KB).
//
// Example:
//
//     pool, _ := NewPinnedBufferPool(16*1024, 32)
//     transport := NewP2PTransport(host, peerID, WithBufferPool(pool))
func WithBufferPool(pool BufferPool) P2PTransportOption
```

**Parameters:**
- `pool`: BufferPool implementation to use

**Example:**

```go
// Create transport with pinned buffer pool
pool, err := transport.NewPinnedBufferPool(16*1024, 32)
if err != nil {
    log.Fatal(err)
}
defer pool.Close()

t := transport.NewP2PTransport(host, peerID, transport.WithBufferPool(pool))
```

### SetBufferPool

```go
// SetBufferPool sets the buffer pool for the transport.
// Can be called after transport creation to add or replace the buffer pool.
// Thread-safe: acquires write lock during update.
//
// Passing nil disables pooled buffers (transport reverts to per-message allocation).
func (t *P2PTransport) SetBufferPool(pool BufferPool)
```

**Parameters:**
- `pool`: BufferPool implementation, or nil to disable pooling

**Example:**

```go
// Dynamically configure buffer pool after creation
t := transport.NewP2PTransport(host, peerID)

// Later, when pool is ready
pool, _ := transport.NewPinnedBufferPool(16*1024, 32)
t.SetBufferPool(pool)

// Disable pooling
t.SetBufferPool(nil)
```

### GetBufferPool

```go
// GetBufferPool returns the configured buffer pool, or nil if none is set.
// Use HasBufferPool() to check if a pool is configured without getting it.
func (t *P2PTransport) GetBufferPool() BufferPool
```

**Returns:**
- `BufferPool`: Current pool, or nil if not configured

**Example:**

```go
pool := t.GetBufferPool()
if pool != nil {
    stats := pool.Stats()
    log.Printf("Pool: %d/%d available", stats.Available, stats.Capacity)
}
```

---

## Package: `p2p`

### BufferPool Interface (p2p package)

```go
// BufferPool provides reusable byte buffers for transport operations.
// This mirrors the transport.BufferPool interface for protocol layer use.
//
// Implementations should be thread-safe for concurrent Get/Put calls.
// Buffers returned by Get may be larger than requested size.
// Put accepts nil buffers gracefully (no-op).
//
// Note: This interface is intentionally duplicated from transport.BufferPool
// to avoid circular import between p2p and transport packages.
type BufferPool interface {
    // Get returns a buffer of at least the requested size.
    // Returns nil if allocation fails and pool is exhausted.
    Get(size int) []byte

    // Put returns a buffer to the pool for reuse.
    // Safe to call with nil buffer.
    Put(buf []byte)

    // Close releases all pooled resources.
    Close() error
}
```

### Protocol.SetBufferPool

```go
// SetBufferPool sets the buffer pool for zero-allocation message handling.
// When set, incoming messages will use pooled buffers instead of allocating new ones.
//
// The pool should have buffers sized for typical activation data (8KB-16KB).
// For CUDA-optimized transfers, use a pinned memory pool implementation.
//
// Thread-safe: can be called while protocol is handling messages.
func (p *Protocol) SetBufferPool(pool BufferPool)
```

**Parameters:**
- `pool`: BufferPool implementation (e.g., pinnedPoolAdapter)

**Example:**

```go
protocol := p2p.NewProtocol(host)

// Configure with pinned pool adapter
adapter := &pinnedPoolAdapter{pool: pinnedBufferPool}
protocol.SetBufferPool(adapter)
```

### Protocol.GetBufferPool

```go
// GetBufferPool returns the configured buffer pool, or nil if none is set.
// Thread-safe.
func (p *Protocol) GetBufferPool() BufferPool
```

**Returns:**
- `BufferPool`: Current pool, or nil

### Protocol.HasBufferPool

```go
// HasBufferPool returns whether the protocol has a buffer pool configured.
// Thread-safe.
func (p *Protocol) HasBufferPool() bool
```

**Returns:**
- `bool`: true if pool is configured

**Example:**

```go
if protocol.HasBufferPool() {
    log.Println("Protocol using pooled buffers")
}
```

---

## Package: `pkg/inference`

### CUDALayerExecutor (Phase 2)

The CUDALayerExecutor now preallocates GPU buffers for input/output to avoid per-call allocations.

### CUDALayerExecutor Fields

```go
type CUDALayerExecutor struct {
    layerWeights map[int]*bindings.LayerWeights
    kvCaches     map[int]*bindings.KVCache
    config       *types.LlamaConfig
    deviceID     int
    mu           sync.RWMutex

    // Preallocated GPU buffers (Phase 2)
    inputGPU   unsafe.Pointer // GPU buffer for input hidden state
    outputGPU  unsafe.Pointer // GPU buffer for output hidden state
    bufferSize uint64         // Size of preallocated buffers in bytes
}
```

### NewCUDALayerExecutor

```go
// NewCUDALayerExecutor creates a new CUDA-based layer executor.
//
// Preallocates GPU buffers for input/output to avoid per-call allocations.
// Buffer size is config.HiddenSize * 2 bytes (FP16 format).
//
// Parameters:
//   - config: LLaMA model configuration (determines buffer sizes)
//   - deviceID: CUDA device index for GPU memory allocation
//
// Returns error if GPU buffer allocation fails.
func NewCUDALayerExecutor(config *types.LlamaConfig, deviceID int) (*CUDALayerExecutor, error)
```

**Example:**

```go
config := types.Llama7BConfig()
executor, err := inference.NewCUDALayerExecutor(config, 0)
if err != nil {
    log.Fatalf("Failed to create executor: %v", err)
}
defer executor.Close()
```

### HasPreallocatedBuffers

```go
// HasPreallocatedBuffers returns whether the executor has preallocated GPU buffers.
func (e *CUDALayerExecutor) HasPreallocatedBuffers() bool
```

**Returns:**
- `bool`: true if both inputGPU and outputGPU are allocated

**Example:**

```go
if executor.HasPreallocatedBuffers() {
    log.Printf("Executor using preallocated buffers (size: %d)", executor.GetBufferSize())
}
```

### GetBufferSize

```go
// GetBufferSize returns the size of preallocated buffers in bytes.
func (e *CUDALayerExecutor) GetBufferSize() uint64
```

**Returns:**
- `uint64`: Buffer size (hiddenSize * 2 for FP16)

**Example:**

```go
size := executor.GetBufferSize()
log.Printf("Preallocated buffer size: %d bytes (%.2f KB)", size, float64(size)/1024)
```

### Close

```go
// Close frees all GPU resources including preallocated buffers.
func (e *CUDALayerExecutor) Close() error
```

**Behavior:**
- Frees inputGPU and outputGPU preallocated buffers
- Frees all layer weights
- Frees all KV caches
- Sets pointers to nil

---

## Package: `cmd/worker`

### Worker Pinned Pool Methods (Phase 2)

### GetPinnedPool

```go
// GetPinnedPool returns the pinned buffer pool, or nil if not initialized.
// Use HasPinnedPool() to check availability without getting the pool.
func (w *Worker) GetPinnedPool() *bindings.PinnedBufferPool
```

**Returns:**
- `*bindings.PinnedBufferPool`: The pool, or nil

**Example:**

```go
if pool := worker.GetPinnedPool(); pool != nil {
    log.Printf("Pool stats: %d buffers x %d bytes", pool.Count(), pool.BufSize())
}
```

### HasPinnedPool

```go
// HasPinnedPool returns whether the worker has a pinned buffer pool.
// Returns false if pool initialization failed or worker was created without GPU.
func (w *Worker) HasPinnedPool() bool
```

**Returns:**
- `bool`: true if pool is available

**Example:**

```go
if worker.HasPinnedPool() {
    log.Println("Worker using CUDA pinned memory for DMA optimization")
} else {
    log.Println("Worker using regular memory buffers")
}
```

---

## PinnedTransportBufferPool

CUDA pinned memory wrapper for transport use. Provides the BufferPool interface using page-locked memory for DMA optimization.

### Constructor

```go
// NewPinnedBufferPool creates a transport buffer pool backed by CUDA pinned memory.
// Returns error if CUDA pinned memory allocation fails.
func NewPinnedBufferPool(bufferSize, poolSize int) (*PinnedTransportBufferPool, error)
```

**Parameters:**
- `bufferSize`: Size of each buffer in bytes (e.g., 16384 for 16KB)
- `poolSize`: Number of buffers to preallocate (e.g., 32)

**Returns:**
- `*PinnedTransportBufferPool`: The created pool
- `error`: Non-nil if allocation fails

**Example:**

```go
pool, err := transport.NewPinnedBufferPool(16*1024, 32)
if err != nil {
    log.Fatalf("Failed to create pinned pool: %v", err)
}
defer pool.Close()
```

### Methods

#### Get

```go
func (p *PinnedTransportBufferPool) Get(size int) []byte
```

Returns a buffer from the pool. If the pool is exhausted, allocates a new pinned buffer (fallback). Returns nil only if the pool is closed or allocation completely fails.

**Parameters:**
- `size`: Minimum buffer size required

**Returns:**
- `[]byte`: Buffer of at least the requested size, or nil on failure

**Thread Safety:** Safe for concurrent use.

**Example:**

```go
buf := pool.Get(10240) // Request 10KB
if buf == nil {
    return errors.New("buffer allocation failed")
}
defer pool.Put(buf)

// Use buffer...
copy(buf, activationData)
```

#### Put

```go
func (p *PinnedTransportBufferPool) Put(buf []byte)
```

Returns a buffer to the pool for reuse. Only returns pinned buffers to the pool; regular buffers (fallback allocations) are discarded. Safe to call with nil or empty buffer.

**Parameters:**
- `buf`: Buffer to return (may be nil)

**Thread Safety:** Safe for concurrent use.

**Example:**

```go
buf := pool.Get(4096)
defer pool.Put(buf) // Always return buffer

// Process data...
```

#### Close

```go
func (p *PinnedTransportBufferPool) Close() error
```

Releases all pool resources. Safe to call multiple times (idempotent). After Close(), Get() returns nil.

**Returns:**
- `error`: Always nil (errors are suppressed during cleanup)

**Example:**

```go
pool, _ := transport.NewPinnedBufferPool(16*1024, 32)
defer pool.Close() // Cleanup on exit

// Use pool...
```

#### Stats

```go
func (p *PinnedTransportBufferPool) Stats() BufferPoolStats
```

Returns current pool statistics.

**Returns:**
- `BufferPoolStats`: Current pool state

**Example:**

```go
stats := pool.Stats()
log.Printf("Pool: %d/%d available, pinned=%v",
    stats.Available, stats.Capacity, stats.IsPinned)
```

---

## DefaultBufferPool

sync.Pool-based buffer pool for non-CUDA systems. Used as fallback when pinned memory is unavailable.

### Constructor

```go
// NewDefaultBufferPool creates a sync.Pool-based buffer pool.
// This is a fallback for systems without CUDA support.
func NewDefaultBufferPool(size int) *DefaultBufferPool
```

**Parameters:**
- `size`: Default buffer size

**Example:**

```go
pool := transport.NewDefaultBufferPool(16 * 1024)
```

### Methods

Same interface as PinnedTransportBufferPool:
- `Get(size int) []byte`
- `Put(buf []byte)`
- `Close() error` (no-op for sync.Pool)

---

## Package: `gpu/bindings`

### PinnedBuffer

```go
// PinnedBuffer represents a CUDA-registered host memory buffer.
// The buffer is page-locked and suitable for DMA transfers to/from GPU.
type PinnedBuffer struct {
    Ptr  unsafe.Pointer // Pointer to pinned memory
    Size uint64         // Size of the buffer in bytes
}
```

#### AsSlice

```go
// AsSlice returns the buffer as a Go byte slice.
// WARNING: The slice is backed by pinned memory - do not use after Put().
func (b *PinnedBuffer) AsSlice() []byte
```

### PinnedBufferPool (bindings)

Low-level CUDA pinned memory pool.

### Constants

```go
const (
    // DefaultPinnedBufferSize is 16KB (covers Llama 7B/13B hidden states).
    DefaultPinnedBufferSize = 16 * 1024

    // DefaultPinnedPoolCount is 32 buffers for concurrent requests.
    DefaultPinnedPoolCount = 32
)
```

### Constructors

```go
// NewPinnedBufferPool creates a pool with preallocated CUDA pinned memory.
func NewPinnedBufferPool(bufSize uint64, count int) (*PinnedBufferPool, error)

// NewPinnedBufferPoolWithDefaults creates a pool using default configuration.
func NewPinnedBufferPoolWithDefaults() (*PinnedBufferPool, error)
```

### Methods

```go
// Get returns a buffer from the pool, or allocates new if exhausted.
func (p *PinnedBufferPool) Get() *PinnedBuffer

// Put returns a buffer to the pool.
func (p *PinnedBufferPool) Put(buf *PinnedBuffer)

// Close frees all buffers.
func (p *PinnedBufferPool) Close() error

// Count returns the pool capacity.
func (p *PinnedBufferPool) Count() int

// BufSize returns the buffer size.
func (p *PinnedBufferPool) BufSize() uint64
```

### CUDA Memory Functions

```go
// AllocPinnedMemory allocates CUDA-registered host memory for DMA transfers.
// Uses cudaHostAllocPortable flag for multi-GPU visibility.
func AllocPinnedMemory(size uint64) (unsafe.Pointer, error)

// FreePinnedMemory releases CUDA-registered host memory.
// Safe to call with nil pointer (no-op).
func FreePinnedMemory(ptr unsafe.Pointer) error

// IsPinnedMemory checks if a pointer points to CUDA-registered pinned memory.
func IsPinnedMemory(ptr unsafe.Pointer) (bool, error)
```

---

## Usage Examples

### Basic Usage

```go
package main

import (
    "github.com/neurogrid/engine/pkg/transport"
)

func main() {
    // Create pinned buffer pool
    pool, err := transport.NewPinnedBufferPool(16*1024, 32)
    if err != nil {
        // Fallback to default pool
        pool = transport.NewDefaultBufferPool(16 * 1024)
    }
    defer pool.Close()

    // Get buffer for transfer
    buf := pool.Get(10240)
    if buf == nil {
        panic("allocation failed")
    }
    defer pool.Put(buf)

    // Use buffer for network transfer
    // ...
}
```

### Worker Integration (Phase 2)

```go
type Worker struct {
    pinnedPool *bindings.PinnedBufferPool
    protocol   *p2p.Protocol
    // ...
}

func (w *Worker) Start() error {
    // Initialize pinned pool
    pool, err := bindings.NewPinnedBufferPoolWithDefaults()
    if err != nil {
        log.Printf("Warning: pinned pool failed: %v", err)
    } else {
        w.pinnedPool = pool
    }

    // Create protocol
    w.protocol = p2p.NewProtocol(w.host)

    // Configure protocol with pinned pool adapter
    if w.pinnedPool != nil {
        adapter := &pinnedPoolAdapter{pool: w.pinnedPool}
        w.protocol.SetBufferPool(adapter)
    }

    return nil
}

func (w *Worker) Shutdown() error {
    if w.pinnedPool != nil {
        w.pinnedPool.Close()
    }
    return nil
}
```

### Transport with Buffer Pool (Phase 2)

```go
// Using functional option
pool, _ := transport.NewPinnedBufferPool(16*1024, 32)
t := transport.NewP2PTransport(host, peerID, transport.WithBufferPool(pool))

// Or set later
t := transport.NewP2PTransport(host, peerID)
t.SetBufferPool(pool)

// Check if pool is configured
if t.GetBufferPool() != nil {
    log.Println("Transport using buffer pool")
}
```

---

## Error Handling

### Allocation Failures

```go
pool, err := transport.NewPinnedBufferPool(16*1024, 32)
if err != nil {
    // Common causes:
    // - CUDA not available
    // - Insufficient pinned memory
    // - GPU initialization failed
    log.Warnf("Pinned pool failed: %v, using fallback", err)
    pool = transport.NewDefaultBufferPool(16 * 1024)
}
```

### Pool Exhaustion

```go
buf := pool.Get(size)
if buf == nil {
    // Pool closed or critical allocation failure
    return errors.New("buffer pool unavailable")
}
```

### Graceful Shutdown

```go
func (w *Worker) Shutdown() error {
    // Close pool to free pinned memory
    if err := w.pinnedPool.Close(); err != nil {
        log.Warnf("Pool close error: %v", err)
    }

    // Pool will:
    // 1. Mark as closed (subsequent Get() returns nil)
    // 2. Free all buffers in channel
    // 3. Buffers still in use will be freed on Put()

    return nil
}
```

---

## Thread Safety Guarantees

| Component | Mechanism | Guarantee |
|-----------|-----------|-----------|
| PinnedBufferPool | Channel | Lock-free Get/Put |
| PinnedTransportBufferPool | sync.RWMutex | Thread-safe bufMap access |
| DefaultBufferPool | sync.Pool | Lock-free |
| Protocol.bufferPool | sync.RWMutex | Thread-safe access |
| P2PTransport.bufferPool | sync.Mutex | Thread-safe SetBufferPool |
| CUDALayerExecutor | sync.RWMutex | Thread-safe Forward |
| AllocPinnedMemory | CUDA Runtime | Thread-safe |
| FreePinnedMemory | CUDA Runtime | Thread-safe |

---

## Performance Characteristics

### Time Complexity

| Operation | Time Complexity | Notes |
|-----------|-----------------|-------|
| Get() | O(1) | Channel receive or fallback alloc |
| Put() | O(1) | Channel send or free |
| Close() | O(n) | Drains all buffers |
| Stats() | O(1) | Reads pool state |

### Memory Usage

| Pool Configuration | Memory Reserved |
|-------------------|-----------------|
| 32 buffers x 16KB | 512 KB |
| 64 buffers x 16KB | 1 MB |
| 32 buffers x 64KB | 2 MB |

Note: Pinned memory is non-pageable and counts against system limits.

---

## Build Tags

```go
//go:build cuda

// PinnedTransportBufferPool requires CUDA build tag
```

Build with CUDA support:
```bash
CGO_ENABLED=1 go build -tags cuda ./...
```

Without CUDA (DefaultBufferPool only):
```bash
go build ./...
```
