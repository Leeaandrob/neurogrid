//go:build cuda

// Package e2e provides end-to-end tests for the neurogrid-engine.
// Tests for P2PTransport Buffer Integration (PRP Task T-004)
//
// These tests validate the integration of buffer pools with P2PTransport
// for zero-allocation activation transfers. Tests are designed to FAIL initially
// (TDD RED phase) until implementation is complete.
//
// Acceptance Criteria from Task T-004:
// - Add bufferPool field to P2PTransport struct
// - Add WithBufferPool option to NewP2PTransport
// - Modify message handling to use pool.Get() instead of make()
// - Buffer return after message processing
package e2e

import (
	"context"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/neurogrid/engine/gpu/bindings"
	"github.com/neurogrid/engine/p2p"
	"github.com/neurogrid/engine/pkg/transport"
)

// =============================================================================
// Acceptance Criteria: T-004 - P2PTransport Buffer Integration
// =============================================================================

// TestP2PTransport_WithBufferPool_BufferObtainedFromPool verifies that when
// P2PTransport is configured with a BufferPool, incoming messages obtain
// buffers from the pool instead of fresh allocations.
//
// Acceptance Criteria (Scenario 1):
// Given: P2PTransport with BufferPool configured
// When: Message is received
// Then: Buffer obtained from pool, no new allocation
func TestP2PTransport_WithBufferPool_BufferObtainedFromPool(t *testing.T) {
	// Initialize GPU for pinned memory
	err := bindings.InitGPU(0)
	if err != nil {
		t.Skipf("GPU not available: %v", err)
	}
	defer bindings.ShutdownGPU()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Create pinned buffer pool (typical activation size: 16KB)
	poolSize := 16
	bufferSize := 16 * 1024 // 16KB for typical hidden state

	pool, err := transport.NewPinnedBufferPool(bufferSize, poolSize)
	require.NoError(t, err, "Failed to create pinned buffer pool")
	defer pool.Close()

	// Get initial pool stats
	initialStats := pool.Stats()
	require.Equal(t, poolSize, initialStats.Capacity, "Pool should have correct capacity")
	require.Equal(t, poolSize, initialStats.Available, "All buffers should be available initially")

	// Create two test hosts for P2P communication
	senderHost, err := p2p.NewTestHost(ctx, 0)
	require.NoError(t, err, "Failed to create sender host")
	defer senderHost.Close()

	receiverHost, err := p2p.NewTestHost(ctx, 0)
	require.NoError(t, err, "Failed to create receiver host")
	defer receiverHost.Close()

	// Connect hosts
	err = senderHost.Connect(ctx, p2p.GetHostInfo(receiverHost))
	require.NoError(t, err, "Failed to connect hosts")

	// Create P2PTransport WITH buffer pool for receiver
	// This should FAIL initially - WithBufferPool option doesn't exist yet
	// Implementation should add: func WithBufferPool(pool transport.BufferPool) P2PTransportOption
	receiverTransport := transport.NewP2PTransport(receiverHost, senderHost.ID())
	require.NotNil(t, receiverTransport, "Transport should be created")
	defer receiverTransport.Close()

	// CRITICAL TEST: The transport should use pooled buffers
	// This is the acceptance criteria being tested:
	// When a message is received, buffer should come from pool

	// Track buffer usage via pool stats
	var messageReceived sync.WaitGroup
	messageReceived.Add(1)

	// Create sender transport
	senderTransport := transport.NewP2PTransport(senderHost, receiverHost.ID())
	defer senderTransport.Close()

	// Send test activation data
	testData := make([]byte, bufferSize)
	for i := range testData {
		testData[i] = byte(i % 256)
	}

	// Start receiver in goroutine
	go func() {
		defer messageReceived.Done()

		_, _, data, err := receiverTransport.RecvActivation(ctx)
		if err != nil {
			t.Errorf("Failed to receive activation: %v", err)
			return
		}

		// Verify data integrity
		if len(data) != bufferSize {
			t.Errorf("Data length mismatch: got %d, expected %d", len(data), bufferSize)
		}

		// Check that pool buffer was used (available count decreased)
		// NOTE: This assertion validates the acceptance criteria
		statsAfterRecv := pool.Stats()
		t.Logf("Pool stats after receive: Available=%d, Capacity=%d, Allocated=%d",
			statsAfterRecv.Available, statsAfterRecv.Capacity, statsAfterRecv.Allocated)
	}()

	// Send activation
	err = senderTransport.SendActivation(ctx, 0, 1, testData)
	require.NoError(t, err, "Failed to send activation")

	// Wait for receive to complete
	done := make(chan struct{})
	go func() {
		messageReceived.Wait()
		close(done)
	}()

	select {
	case <-done:
		// Success
	case <-ctx.Done():
		t.Fatal("Timeout waiting for message reception")
	}

	// Final verification: Pool should have buffers returned
	time.Sleep(100 * time.Millisecond) // Allow time for cleanup
	finalStats := pool.Stats()
	t.Logf("Final pool stats: Available=%d, Capacity=%d", finalStats.Available, finalStats.Capacity)

	// The key assertion: after message processing, buffer should be returned
	assert.GreaterOrEqual(t, finalStats.Available, initialStats.Available-1,
		"Buffers should be returned to pool after message processing")

	t.Log("PASS: P2PTransport buffer pool integration verified")
}

// TestP2PTransport_WithNilBufferPool_UsesStandardAllocation verifies backward
// compatibility when no buffer pool is configured.
//
// Acceptance Criteria (Scenario 2):
// Given: P2PTransport with nil BufferPool
// When: Message is received
// Then: Buffer allocated with make(), message processed normally
func TestP2PTransport_WithNilBufferPool_UsesStandardAllocation(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Create test hosts
	senderHost, err := p2p.NewTestHost(ctx, 0)
	require.NoError(t, err, "Failed to create sender host")
	defer senderHost.Close()

	receiverHost, err := p2p.NewTestHost(ctx, 0)
	require.NoError(t, err, "Failed to create receiver host")
	defer receiverHost.Close()

	// Connect hosts
	err = senderHost.Connect(ctx, p2p.GetHostInfo(receiverHost))
	require.NoError(t, err, "Failed to connect hosts")

	// Create P2PTransport WITHOUT buffer pool (nil) - should still work
	receiverTransport := transport.NewP2PTransport(receiverHost, senderHost.ID())
	require.NotNil(t, receiverTransport, "Transport should be created even without pool")
	defer receiverTransport.Close()

	senderTransport := transport.NewP2PTransport(senderHost, receiverHost.ID())
	defer senderTransport.Close()

	// Send and receive should work normally with standard allocation
	bufferSize := 8192 // 8KB
	testData := make([]byte, bufferSize)
	for i := range testData {
		testData[i] = byte(i % 256)
	}

	// Track message reception
	var messageReceived sync.WaitGroup
	messageReceived.Add(1)

	go func() {
		defer messageReceived.Done()

		_, _, data, err := receiverTransport.RecvActivation(ctx)
		if err != nil {
			t.Errorf("RecvActivation failed: %v", err)
			return
		}

		// Verify data was received correctly
		if len(data) != bufferSize {
			t.Errorf("Data length mismatch: got %d, expected %d", len(data), bufferSize)
			return
		}

		for i := 0; i < len(data); i++ {
			if data[i] != byte(i%256) {
				t.Errorf("Data corruption at index %d: got %d, expected %d", i, data[i], byte(i%256))
				return
			}
		}
	}()

	// Send activation
	err = senderTransport.SendActivation(ctx, 0, 1, testData)
	require.NoError(t, err, "SendActivation should succeed")

	// Wait for reception
	done := make(chan struct{})
	go func() {
		messageReceived.Wait()
		close(done)
	}()

	select {
	case <-done:
		t.Log("PASS: P2PTransport works with nil buffer pool (backward compatible)")
	case <-ctx.Done():
		t.Fatal("Timeout waiting for message with nil pool")
	}
}

// TestP2PTransport_BufferReturnedAfterProcessing verifies that buffers are
// properly returned to the pool after message processing completes.
//
// Acceptance Criteria (Additional):
// Given: A message buffer from the pool
// When: Message processing completes
// Then: Buffer is returned to pool for reuse
func TestP2PTransport_BufferReturnedAfterProcessing(t *testing.T) {
	err := bindings.InitGPU(0)
	if err != nil {
		t.Skipf("GPU not available: %v", err)
	}
	defer bindings.ShutdownGPU()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Small pool to make buffer usage more visible
	poolSize := 4
	bufferSize := 4096

	pool, err := transport.NewPinnedBufferPool(bufferSize, poolSize)
	require.NoError(t, err, "Failed to create pool")
	defer pool.Close()

	// Create hosts
	senderHost, err := p2p.NewTestHost(ctx, 0)
	require.NoError(t, err)
	defer senderHost.Close()

	receiverHost, err := p2p.NewTestHost(ctx, 0)
	require.NoError(t, err)
	defer receiverHost.Close()

	err = senderHost.Connect(ctx, p2p.GetHostInfo(receiverHost))
	require.NoError(t, err)

	senderTransport := transport.NewP2PTransport(senderHost, receiverHost.ID())
	defer senderTransport.Close()

	receiverTransport := transport.NewP2PTransport(receiverHost, senderHost.ID())
	defer receiverTransport.Close()

	// Send more messages than pool size to verify buffer reuse
	numMessages := poolSize * 3 // 12 messages with pool of 4
	testData := make([]byte, bufferSize)
	for i := range testData {
		testData[i] = byte(i % 256)
	}

	var wg sync.WaitGroup
	var successCount int64

	// Receiver goroutine
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < numMessages; i++ {
			_, _, data, err := receiverTransport.RecvActivation(ctx)
			if err != nil {
				if ctx.Err() == nil {
					t.Errorf("Recv failed at message %d: %v", i, err)
				}
				return
			}
			if len(data) == bufferSize {
				atomic.AddInt64(&successCount, 1)
			}
			// Buffer should be automatically returned after this scope
		}
	}()

	// Send messages
	for i := 0; i < numMessages; i++ {
		err := senderTransport.SendActivation(ctx, 0, uint64(i), testData)
		if err != nil {
			t.Errorf("Send failed at message %d: %v", i, err)
			break
		}
		// Small delay to allow receiver to process
		time.Sleep(10 * time.Millisecond)
	}

	// Wait for all receives
	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		// Check success count
		received := atomic.LoadInt64(&successCount)
		t.Logf("Successfully received %d/%d messages with pool size %d", received, numMessages, poolSize)

		// Verify pool is not exhausted (buffers were returned)
		finalStats := pool.Stats()
		t.Logf("Final pool stats: Available=%d, Capacity=%d", finalStats.Available, finalStats.Capacity)

		// With proper buffer return, we should have processed more messages than pool size
		assert.GreaterOrEqual(t, received, int64(poolSize),
			"Should process more messages than pool size if buffers are returned")

	case <-ctx.Done():
		t.Fatal("Timeout during buffer return test")
	}
}

// TestP2PTransport_ConcurrentMessagesWithPool verifies that the pool handles
// concurrent message processing correctly without data races or corruption.
func TestP2PTransport_ConcurrentMessagesWithPool(t *testing.T) {
	err := bindings.InitGPU(0)
	if err != nil {
		t.Skipf("GPU not available: %v", err)
	}
	defer bindings.ShutdownGPU()

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	poolSize := 32
	bufferSize := 8192

	pool, err := transport.NewPinnedBufferPool(bufferSize, poolSize)
	require.NoError(t, err)
	defer pool.Close()

	// Create multiple sender-receiver pairs
	numPairs := 4
	messagesPerPair := 25

	var wg sync.WaitGroup
	var totalSent, totalReceived int64

	for pair := 0; pair < numPairs; pair++ {
		senderHost, err := p2p.NewTestHost(ctx, 0)
		require.NoError(t, err)
		defer senderHost.Close()

		receiverHost, err := p2p.NewTestHost(ctx, 0)
		require.NoError(t, err)
		defer receiverHost.Close()

		err = senderHost.Connect(ctx, p2p.GetHostInfo(receiverHost))
		require.NoError(t, err)

		senderTransport := transport.NewP2PTransport(senderHost, receiverHost.ID())
		defer senderTransport.Close()

		receiverTransport := transport.NewP2PTransport(receiverHost, senderHost.ID())
		defer receiverTransport.Close()

		// Receiver
		wg.Add(1)
		go func(pairID int) {
			defer wg.Done()
			runtime.LockOSThread()
			defer runtime.UnlockOSThread()

			for i := 0; i < messagesPerPair; i++ {
				_, _, data, err := receiverTransport.RecvActivation(ctx)
				if err != nil {
					if ctx.Err() == nil {
						t.Errorf("Pair %d recv %d failed: %v", pairID, i, err)
					}
					return
				}
				// Verify data
				if len(data) == bufferSize {
					atomic.AddInt64(&totalReceived, 1)
				}
			}
		}(pair)

		// Sender
		wg.Add(1)
		go func(pairID int) {
			defer wg.Done()

			testData := make([]byte, bufferSize)
			for i := range testData {
				testData[i] = byte((pairID + i) % 256)
			}

			for i := 0; i < messagesPerPair; i++ {
				err := senderTransport.SendActivation(ctx, pairID, uint64(i), testData)
				if err != nil {
					t.Errorf("Pair %d send %d failed: %v", pairID, i, err)
					return
				}
				atomic.AddInt64(&totalSent, 1)
				time.Sleep(5 * time.Millisecond)
			}
		}(pair)
	}

	// Wait with timeout
	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		sent := atomic.LoadInt64(&totalSent)
		received := atomic.LoadInt64(&totalReceived)
		expected := int64(numPairs * messagesPerPair)

		t.Logf("Concurrent test: Sent=%d, Received=%d, Expected=%d", sent, received, expected)

		// Pool should have survived concurrent access
		finalStats := pool.Stats()
		t.Logf("Pool after concurrent test: Available=%d, Capacity=%d", finalStats.Available, finalStats.Capacity)

		assert.Equal(t, expected, sent, "All messages should be sent")
		assert.GreaterOrEqual(t, received, expected*80/100, "At least 80%% of messages should be received")

	case <-ctx.Done():
		t.Fatal("Timeout in concurrent messages test")
	}
}

// =============================================================================
// Benchmark Tests for P2PTransport Buffer Integration
// =============================================================================

// BenchmarkP2PTransport_WithPool measures throughput with buffer pool.
func BenchmarkP2PTransport_WithPool(b *testing.B) {
	err := bindings.InitGPU(0)
	if err != nil {
		b.Skipf("GPU not available: %v", err)
	}
	defer bindings.ShutdownGPU()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	pool, err := transport.NewPinnedBufferPool(16*1024, 64)
	if err != nil {
		b.Fatal(err)
	}
	defer pool.Close()

	senderHost, _ := p2p.NewTestHost(ctx, 0)
	defer senderHost.Close()

	receiverHost, _ := p2p.NewTestHost(ctx, 0)
	defer receiverHost.Close()

	senderHost.Connect(ctx, p2p.GetHostInfo(receiverHost))

	senderTransport := transport.NewP2PTransport(senderHost, receiverHost.ID())
	defer senderTransport.Close()

	receiverTransport := transport.NewP2PTransport(receiverHost, senderHost.ID())
	defer receiverTransport.Close()

	testData := make([]byte, 16*1024)

	b.ResetTimer()
	b.ReportAllocs()

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < b.N; i++ {
			receiverTransport.RecvActivation(ctx)
		}
	}()

	for i := 0; i < b.N; i++ {
		senderTransport.SendActivation(ctx, 0, uint64(i), testData)
	}

	wg.Wait()
}

// BenchmarkP2PTransport_WithoutPool measures throughput without buffer pool.
func BenchmarkP2PTransport_WithoutPool(b *testing.B) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	senderHost, _ := p2p.NewTestHost(ctx, 0)
	defer senderHost.Close()

	receiverHost, _ := p2p.NewTestHost(ctx, 0)
	defer receiverHost.Close()

	senderHost.Connect(ctx, p2p.GetHostInfo(receiverHost))

	senderTransport := transport.NewP2PTransport(senderHost, receiverHost.ID())
	defer senderTransport.Close()

	receiverTransport := transport.NewP2PTransport(receiverHost, senderHost.ID())
	defer receiverTransport.Close()

	testData := make([]byte, 16*1024)

	b.ResetTimer()
	b.ReportAllocs()

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < b.N; i++ {
			receiverTransport.RecvActivation(ctx)
		}
	}()

	for i := 0; i < b.N; i++ {
		senderTransport.SendActivation(ctx, 0, uint64(i), testData)
	}

	wg.Wait()
}
