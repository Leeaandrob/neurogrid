//go:build cuda

// Package transport provides abstractions for activation transfer between peers.
// Tests for Transport Buffer Pool (PRP Tasks T-002, T-004, T-005, T-007)
//
// These tests verify the buffer pool implementation for transport layer
// optimization using CUDA pinned memory. Tests are designed to FAIL initially
// (TDD RED phase) until implementation is complete.
package transport

import (
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// =============================================================================
// Acceptance Criteria: T-002 - Buffer Pool Implementation
// =============================================================================

// TestBufferPool_PreallocatesBuffersOnCreation verifies that NewPinnedBufferPool
// preallocates the specified number of buffers on creation.
//
// Given: Pool configuration with N buffers
// When: NewPinnedBufferPool is called
// Then: Pool should have N buffers ready for use
func TestBufferPool_PreallocatesBuffersOnCreation(t *testing.T) {
	poolSize := 16
	bufferSize := 16 * 1024 // 16KB typical activation size

	pool, err := NewPinnedBufferPool(bufferSize, poolSize)
	require.NoError(t, err, "NewPinnedBufferPool should succeed")
	require.NotNil(t, pool, "Pool should not be nil")
	defer pool.Close()

	// Verify we can get all preallocated buffers without blocking
	buffers := make([][]byte, poolSize)
	for i := 0; i < poolSize; i++ {
		buf := pool.Get(bufferSize)
		require.NotNil(t, buf, "Buffer %d should be available", i)
		require.Equal(t, bufferSize, len(buf), "Buffer should have correct size")
		buffers[i] = buf
	}

	// Return all buffers
	for _, buf := range buffers {
		pool.Put(buf)
	}

	t.Logf("PASS: Pool preallocated %d buffers of %d bytes each", poolSize, bufferSize)
}

// TestBufferPool_GetReturnsInO1 verifies that Get() returns a buffer in O(1) time.
//
// Given: A pool with available buffers
// When: Get() is called
// Then: It should return in constant time (< 1ms)
func TestBufferPool_GetReturnsInO1(t *testing.T) {
	poolSize := 32
	bufferSize := 16 * 1024

	pool, err := NewPinnedBufferPool(bufferSize, poolSize)
	require.NoError(t, err)
	defer pool.Close()

	// Warm up
	for i := 0; i < 10; i++ {
		buf := pool.Get(bufferSize)
		pool.Put(buf)
	}

	// Measure Get() latency
	iterations := 1000
	var totalDuration time.Duration

	for i := 0; i < iterations; i++ {
		start := time.Now()
		buf := pool.Get(bufferSize)
		elapsed := time.Since(start)
		totalDuration += elapsed
		pool.Put(buf)
	}

	avgLatency := totalDuration / time.Duration(iterations)
	maxAcceptableLatency := 1 * time.Millisecond

	assert.Less(t, avgLatency, maxAcceptableLatency,
		"Average Get() latency %v should be < %v for O(1) behavior", avgLatency, maxAcceptableLatency)

	t.Logf("PASS: Average Get() latency: %v (target < %v)", avgLatency, maxAcceptableLatency)
}

// TestBufferPool_PutReturnsBuffer verifies that Put() returns buffer to pool.
//
// Given: A buffer obtained from the pool
// When: Put() is called
// Then: Buffer should be returned to pool for reuse
func TestBufferPool_PutReturnsBuffer(t *testing.T) {
	poolSize := 4
	bufferSize := 1024

	pool, err := NewPinnedBufferPool(bufferSize, poolSize)
	require.NoError(t, err)
	defer pool.Close()

	// Get all buffers
	buffers := make([][]byte, poolSize)
	for i := 0; i < poolSize; i++ {
		buffers[i] = pool.Get(bufferSize)
		require.NotNil(t, buffers[i])
	}

	// Return first buffer
	pool.Put(buffers[0])

	// Should be able to get one more buffer now
	buf := pool.Get(bufferSize)
	require.NotNil(t, buf, "Should get buffer after Put()")

	// Clean up
	pool.Put(buf)
	for i := 1; i < poolSize; i++ {
		pool.Put(buffers[i])
	}
}

// TestBufferPool_OverflowFreesExcess verifies that returning more buffers
// than pool capacity frees the excess.
//
// Given: A full pool
// When: More buffers are returned than capacity
// Then: Excess buffers should be freed (not accumulated)
func TestBufferPool_OverflowFreesExcess(t *testing.T) {
	poolSize := 4
	bufferSize := 1024

	pool, err := NewPinnedBufferPool(bufferSize, poolSize)
	require.NoError(t, err)
	defer pool.Close()

	// Get all preallocated buffers
	original := make([][]byte, poolSize)
	for i := 0; i < poolSize; i++ {
		original[i] = pool.Get(bufferSize)
	}

	// Return them all
	for _, buf := range original {
		pool.Put(buf)
	}

	// Get new buffers (fallback allocation since pool was exhausted)
	extra := make([][]byte, 4)
	for i := 0; i < 4; i++ {
		extra[i] = pool.Get(bufferSize)
	}

	// Return original buffers again - should fill pool
	for _, buf := range original {
		pool.Put(buf)
	}

	// Return extra buffers - should be freed (overflow)
	for _, buf := range extra {
		pool.Put(buf)
	}

	// Pool stats should show capacity, not capacity + extras
	stats := pool.Stats()
	assert.LessOrEqual(t, stats.Available, poolSize,
		"Pool should not exceed capacity after overflow")
}

// TestBufferPool_ConcurrentAccessThreadSafe verifies thread safety of the pool.
//
// Given: Multiple goroutines accessing the pool
// When: Concurrent Get() and Put() operations
// Then: No data races, panics, or corruption
func TestBufferPool_ConcurrentAccessThreadSafe(t *testing.T) {
	poolSize := 32
	bufferSize := 4096
	numGoroutines := 100
	iterationsPerGoroutine := 1000

	pool, err := NewPinnedBufferPool(bufferSize, poolSize)
	require.NoError(t, err)
	defer pool.Close()

	var wg sync.WaitGroup
	var successCount int64
	var errorCount int64

	for g := 0; g < numGoroutines; g++ {
		wg.Add(1)
		go func(goroutineID int) {
			defer wg.Done()

			for i := 0; i < iterationsPerGoroutine; i++ {
				// Get buffer
				buf := pool.Get(bufferSize)
				if buf == nil {
					atomic.AddInt64(&errorCount, 1)
					continue
				}

				// Write pattern to verify no data races
				pattern := byte((goroutineID + i) % 256)
				for j := 0; j < len(buf); j++ {
					buf[j] = pattern
				}

				// Small delay to increase chance of race conditions
				if i%100 == 0 {
					runtime.Gosched()
				}

				// Verify pattern before returning
				for j := 0; j < len(buf); j++ {
					if buf[j] != pattern {
						atomic.AddInt64(&errorCount, 1)
						break
					}
				}

				pool.Put(buf)
				atomic.AddInt64(&successCount, 1)
			}
		}(g)
	}

	wg.Wait()

	totalOps := int64(numGoroutines * iterationsPerGoroutine)
	assert.Equal(t, totalOps, successCount+errorCount, "All operations should complete")
	assert.Zero(t, errorCount, "No errors should occur during concurrent access")

	t.Logf("PASS: %d concurrent operations completed without errors", successCount)
}

// =============================================================================
// Acceptance Criteria: T-004, T-005 - Transport Integration
// =============================================================================

// TestP2PTransport_UsesPooledBuffersWhenConfigured verifies that P2PTransport
// uses the buffer pool when one is configured.
//
// Given: P2PTransport configured with a buffer pool
// When: SendActivation is called
// Then: Buffer should be obtained from pool, not allocated fresh
func TestP2PTransport_UsesPooledBuffersWhenConfigured(t *testing.T) {
	t.Skip("Requires P2P network environment - test validates integration point")

	poolSize := 16
	bufferSize := 16 * 1024

	pool, err := NewPinnedBufferPool(bufferSize, poolSize)
	require.NoError(t, err)
	defer pool.Close()

	// Create transport with pool
	// Note: This tests the integration point - actual P2P requires network
	transport := NewP2PTransportWithPool(nil, "", pool)
	require.NotNil(t, transport)
	defer transport.Close()

	// Verify pool is configured
	assert.True(t, transport.HasBufferPool(), "Transport should have buffer pool configured")
}

// TestP2PTransport_BackwardCompatibleWithNilPool verifies that P2PTransport
// works correctly when no buffer pool is configured (backward compatibility).
//
// Given: P2PTransport with no buffer pool (nil)
// When: Operations are performed
// Then: Should work with standard allocation
func TestP2PTransport_BackwardCompatibleWithNilPool(t *testing.T) {
	// Create mock transport without pool - should not panic
	pool, _ := NewPinnedBufferPool(1024, 4)
	if pool != nil {
		defer pool.Close()
	}

	// Nil pool should be handled gracefully
	transport := NewP2PTransportWithPool(nil, "", nil)
	if transport != nil {
		assert.False(t, transport.HasBufferPool(), "Transport with nil pool should report no pool")
		transport.Close()
	}
}

// TestBufferPool_ReturnedAfterMessageProcessing verifies that buffers are
// properly returned to the pool after message processing completes.
//
// Given: A buffer pool with tracking
// When: Messages are processed through the transport
// Then: Buffers should be returned to pool
func TestBufferPool_ReturnedAfterMessageProcessing(t *testing.T) {
	poolSize := 8
	bufferSize := 4096

	pool, err := NewPinnedBufferPool(bufferSize, poolSize)
	require.NoError(t, err)
	defer pool.Close()

	initialStats := pool.Stats()

	// Simulate message processing: get buffer, process, return
	iterations := 100
	for i := 0; i < iterations; i++ {
		buf := pool.Get(bufferSize)
		require.NotNil(t, buf)

		// Simulate processing
		for j := 0; j < len(buf); j++ {
			buf[j] = byte(i % 256)
		}

		pool.Put(buf)
	}

	finalStats := pool.Stats()

	// Available count should be same or higher (not leaked)
	assert.GreaterOrEqual(t, finalStats.Available, initialStats.Available,
		"Buffers should be returned after processing")

	t.Logf("PASS: After %d iterations - Initial available: %d, Final available: %d",
		iterations, initialStats.Available, finalStats.Available)
}

// =============================================================================
// Acceptance Criteria: T-007 - Worker Integration
// =============================================================================

// TestWorkerInitializesPinnedPoolOnStart verifies that workers initialize
// the pinned buffer pool during startup.
//
// Given: Worker configuration with pinned buffers enabled
// When: Worker starts
// Then: Pinned buffer pool should be initialized
func TestWorkerInitializesPinnedPoolOnStart(t *testing.T) {
	config := WorkerPoolConfig{
		PoolSize:        16,
		BufferSize:      16 * 1024,
		UsePinnedMemory: true,
	}

	// Simulate worker initialization
	pool, err := InitializeWorkerPool(config)
	require.NoError(t, err, "Worker pool initialization should succeed")
	require.NotNil(t, pool, "Pool should be created")
	defer pool.Close()

	stats := pool.Stats()
	assert.Equal(t, config.PoolSize, stats.Capacity, "Pool should have configured capacity")
}

// TestWorkerGracefulFallbackIfPinnedAllocFails verifies that workers fall
// back gracefully if pinned memory allocation fails.
//
// Given: System where pinned allocation might fail
// When: Worker attempts to initialize pool
// Then: Should fall back to regular buffers without crashing
func TestWorkerGracefulFallbackIfPinnedAllocFails(t *testing.T) {
	config := WorkerPoolConfig{
		PoolSize:        16,
		BufferSize:      16 * 1024,
		UsePinnedMemory: true,
		FallbackEnabled: true,
	}

	// Even if pinned allocation fails, should get a working pool
	pool, err := InitializeWorkerPool(config)

	// Either success or graceful error
	if err != nil {
		assert.Contains(t, err.Error(), "fallback", "Error should indicate fallback behavior")
	} else {
		require.NotNil(t, pool)
		pool.Close()
	}
}

// TestWorkerPoolClosedOnShutdown verifies that the buffer pool is properly
// closed when the worker shuts down.
//
// Given: A running worker with active pool
// When: Worker shuts down
// Then: Pool should be closed and resources freed
func TestWorkerPoolClosedOnShutdown(t *testing.T) {
	config := WorkerPoolConfig{
		PoolSize:        8,
		BufferSize:      4096,
		UsePinnedMemory: true,
	}

	pool, err := InitializeWorkerPool(config)
	require.NoError(t, err)

	// Get some buffers
	buffers := make([][]byte, 4)
	for i := 0; i < 4; i++ {
		buffers[i] = pool.Get(config.BufferSize)
	}

	// Return them
	for _, buf := range buffers {
		pool.Put(buf)
	}

	// Close pool
	err = pool.Close()
	assert.NoError(t, err, "Pool close should succeed")

	// Verify pool is closed - Get should fail or return nil
	buf := pool.Get(config.BufferSize)
	assert.Nil(t, buf, "Get after Close should return nil")
}

// =============================================================================
// DefaultBufferPool Tests (sync.Pool based fallback)
// =============================================================================

// TestDefaultBufferPool_FallbackWorks verifies the default sync.Pool fallback.
//
// Given: DefaultBufferPool (non-pinned fallback)
// When: Get and Put are called
// Then: Should work without CUDA
func TestDefaultBufferPool_FallbackWorks(t *testing.T) {
	pool := NewDefaultBufferPool(4096)
	require.NotNil(t, pool)

	// Get buffer
	buf := pool.Get(4096)
	require.NotNil(t, buf)
	require.Equal(t, 4096, len(buf))

	// Use buffer
	for i := range buf {
		buf[i] = byte(i % 256)
	}

	// Return buffer
	pool.Put(buf)

	// Get again - may be same buffer (reused)
	buf2 := pool.Get(4096)
	require.NotNil(t, buf2)
	require.Equal(t, 4096, len(buf2))

	pool.Put(buf2)
}

// =============================================================================
// BufferPool Interface Tests
// =============================================================================

// TestBufferPoolInterface_PinnedImplements verifies PinnedBufferPool implements
// the BufferPool interface.
func TestBufferPoolInterface_PinnedImplements(t *testing.T) {
	pool, err := NewPinnedBufferPool(1024, 4)
	if err != nil {
		t.Skip("Pinned buffer allocation not available")
	}
	defer pool.Close()

	// Verify interface compliance
	var _ BufferPool = pool
	t.Log("PASS: PinnedBufferPool implements BufferPool interface")
}

// TestBufferPoolInterface_DefaultImplements verifies DefaultBufferPool implements
// the BufferPool interface.
func TestBufferPoolInterface_DefaultImplements(t *testing.T) {
	pool := NewDefaultBufferPool(1024)
	var _ BufferPool = pool
	t.Log("PASS: DefaultBufferPool implements BufferPool interface")
}

// =============================================================================
// Benchmark Tests
// =============================================================================

// BenchmarkBufferPool_GetPut measures pool Get/Put throughput.
func BenchmarkBufferPool_GetPut(b *testing.B) {
	pool, err := NewPinnedBufferPool(16*1024, 32)
	if err != nil {
		b.Skip("Pinned buffer pool not available")
	}
	defer pool.Close()

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		buf := pool.Get(16 * 1024)
		pool.Put(buf)
	}
}

// BenchmarkBufferPool_ConcurrentGetPut measures concurrent pool performance.
func BenchmarkBufferPool_ConcurrentGetPut(b *testing.B) {
	pool, err := NewPinnedBufferPool(16*1024, 64)
	if err != nil {
		b.Skip("Pinned buffer pool not available")
	}
	defer pool.Close()

	b.ResetTimer()
	b.ReportAllocs()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			buf := pool.Get(16 * 1024)
			pool.Put(buf)
		}
	})
}

// BenchmarkDefaultPool_GetPut measures default pool throughput.
func BenchmarkDefaultPool_GetPut(b *testing.B) {
	pool := NewDefaultBufferPool(16 * 1024)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		buf := pool.Get(16 * 1024)
		pool.Put(buf)
	}
}

// =============================================================================
// Helper Types for Tests
// =============================================================================
// All types (BufferPool, BufferPoolStats, WorkerPoolConfig, PinnedTransportBufferPool,
// DefaultBufferPool) are now defined in buffer_pool.go - tests use those implementations.
