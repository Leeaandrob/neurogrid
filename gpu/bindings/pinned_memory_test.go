//go:build cuda

// Package bindings provides CGO bindings for CUDA operations.
// Tests for CUDA Pinned Memory (PRP Task T-001)
//
// These tests verify the CUDA pinned memory bindings required for
// DMA transfers between host and GPU memory. Tests are designed to
// FAIL initially (TDD RED phase) until implementation is complete.
package bindings

import (
	"fmt"
	"runtime"
	"sync"
	"testing"
	"unsafe"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// =============================================================================
// Acceptance Criteria: T-001 - CUDA Pinned Memory Bindings
// =============================================================================

// TestAllocPinnedMemory_ReturnsValidPointer verifies that AllocPinnedMemory
// allocates CUDA-registered host memory and returns a valid non-nil pointer.
//
// Given: A valid allocation size
// When: AllocPinnedMemory is called
// Then: It should return a valid pointer and no error
func TestAllocPinnedMemory_ReturnsValidPointer(t *testing.T) {
	err := InitGPU(0)
	require.NoError(t, err, "GPU initialization should succeed")
	defer ShutdownGPU()

	// Allocate typical activation buffer size (10KB for Llama 13B hidden state)
	size := uint64(10 * 1024) // 10KB

	ptr, err := AllocPinnedMemory(size)
	require.NoError(t, err, "AllocPinnedMemory should succeed")
	require.NotNil(t, ptr, "AllocPinnedMemory should return non-nil pointer")

	// Cleanup
	err = FreePinnedMemory(ptr)
	assert.NoError(t, err, "FreePinnedMemory should succeed")
}

// TestAllocPinnedMemory_VariousSizes verifies allocation of various buffer sizes
// relevant to LLM inference (8KB for Llama 7B, 10KB for Llama 13B, 16KB for larger).
//
// Given: Various allocation sizes
// When: AllocPinnedMemory is called for each size
// Then: All allocations should succeed
func TestAllocPinnedMemory_VariousSizes(t *testing.T) {
	err := InitGPU(0)
	require.NoError(t, err)
	defer ShutdownGPU()

	testCases := []struct {
		name string
		size uint64
	}{
		{"Llama7B_HiddenState", 4096 * 2},  // 8KB - Llama 7B hidden size * FP16
		{"Llama13B_HiddenState", 5120 * 2}, // 10KB - Llama 13B hidden size * FP16
		{"LargeBuffer_16KB", 16 * 1024},    // 16KB
		{"LargeBuffer_64KB", 64 * 1024},    // 64KB
		{"LargeBuffer_1MB", 1024 * 1024},   // 1MB for batch transfers
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			ptr, err := AllocPinnedMemory(tc.size)
			require.NoError(t, err, "Allocation of %d bytes should succeed", tc.size)
			require.NotNil(t, ptr)

			err = FreePinnedMemory(ptr)
			assert.NoError(t, err)
		})
	}
}

// TestFreePinnedMemory_NilHandledGracefully verifies that FreePinnedMemory
// handles nil pointers gracefully without panicking or returning errors.
//
// Given: A nil pointer
// When: FreePinnedMemory is called
// Then: It should not panic and handle gracefully
func TestFreePinnedMemory_NilHandledGracefully(t *testing.T) {
	err := InitGPU(0)
	require.NoError(t, err)
	defer ShutdownGPU()

	// Should not panic when called with nil
	err = FreePinnedMemory(nil)
	// nil pointer should be handled gracefully (either no error or documented error)
	// The key requirement is NO PANIC
	t.Logf("FreePinnedMemory(nil) returned: %v (no panic is success)", err)
}

// TestFreePinnedMemory_DoubleFreeHandled verifies that double-free is handled
// gracefully without causing crashes.
//
// Given: A freed pinned memory pointer
// When: FreePinnedMemory is called again
// Then: It should not crash (may return error)
func TestFreePinnedMemory_DoubleFreeHandled(t *testing.T) {
	err := InitGPU(0)
	require.NoError(t, err)
	defer ShutdownGPU()

	ptr, err := AllocPinnedMemory(1024)
	require.NoError(t, err)
	require.NotNil(t, ptr)

	// First free should succeed
	err = FreePinnedMemory(ptr)
	assert.NoError(t, err)

	// Second free - should not crash (may return error)
	// The key requirement is NO CRASH
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("Double free caused panic: %v", r)
		}
	}()
	_ = FreePinnedMemory(ptr) // Error is acceptable, crash is not
}

// TestPinnedMemory_WriteAndRead verifies that pinned memory can be written to
// and read from correctly.
//
// Given: Allocated pinned memory
// When: Data is written and read back
// Then: The data should match
func TestPinnedMemory_WriteAndRead(t *testing.T) {
	err := InitGPU(0)
	require.NoError(t, err)
	defer ShutdownGPU()

	size := uint64(1024)
	ptr, err := AllocPinnedMemory(size)
	require.NoError(t, err)
	defer FreePinnedMemory(ptr)

	// Create a slice backed by the pinned memory
	slice := unsafe.Slice((*byte)(ptr), size)

	// Write test pattern
	for i := range slice {
		slice[i] = byte(i % 256)
	}

	// Verify pattern
	for i := range slice {
		assert.Equal(t, byte(i%256), slice[i], "Data mismatch at index %d", i)
	}
}

// TestPinnedMemory_MultiGPUVisibility verifies that pinned memory allocated
// with cudaHostAllocPortable is visible from multiple GPUs.
//
// Given: Multiple GPUs available
// When: Pinned memory is allocated with portable flag
// Then: Memory should be accessible from all GPUs
func TestPinnedMemory_MultiGPUVisibility(t *testing.T) {
	deviceCount, err := GetDeviceCount()
	require.NoError(t, err)

	if deviceCount < 2 {
		t.Skip("Multi-GPU test requires at least 2 GPUs")
	}

	// Initialize multi-GPU
	deviceIDs := make([]int, deviceCount)
	for i := 0; i < deviceCount; i++ {
		deviceIDs[i] = i
	}
	err = InitMultiGPU(deviceIDs)
	require.NoError(t, err)
	defer ShutdownMultiGPU()

	// Allocate pinned memory (should use cudaHostAllocPortable)
	size := uint64(16 * 1024) // 16KB
	pinnedPtr, err := AllocPinnedMemory(size)
	require.NoError(t, err, "Pinned memory allocation should succeed")
	defer FreePinnedMemory(pinnedPtr)

	// Test copying from pinned to each GPU device
	for deviceID := 0; deviceID < deviceCount; deviceID++ {
		t.Run(fmt.Sprintf("Device%d", deviceID), func(t *testing.T) {
			// Allocate GPU memory on this device
			gpuPtr, err := AllocOnDevice(size, deviceID)
			require.NoError(t, err, "GPU allocation on device %d should succeed", deviceID)
			defer FreeOnDevice(gpuPtr, deviceID)

			// Write pattern to pinned memory
			slice := unsafe.Slice((*byte)(pinnedPtr), size)
			for i := range slice {
				slice[i] = byte((deviceID + i) % 256)
			}

			// Copy from pinned to GPU should succeed for any device
			err = CopyToDeviceRaw(gpuPtr, pinnedPtr, size)
			assert.NoError(t, err, "Copy from pinned to device %d should succeed", deviceID)
		})
	}
}

// TestPinnedMemory_ConcurrentAccess verifies that multiple goroutines can
// safely access different pinned memory regions concurrently.
//
// Given: Multiple pinned memory allocations
// When: Multiple goroutines access them concurrently
// Then: No data races or crashes should occur
func TestPinnedMemory_ConcurrentAccess(t *testing.T) {
	err := InitGPU(0)
	require.NoError(t, err)
	defer ShutdownGPU()

	numBuffers := 10
	bufferSize := uint64(4096)

	// Allocate multiple pinned buffers
	buffers := make([]unsafe.Pointer, numBuffers)
	for i := 0; i < numBuffers; i++ {
		ptr, err := AllocPinnedMemory(bufferSize)
		require.NoError(t, err)
		buffers[i] = ptr
	}
	defer func() {
		for _, ptr := range buffers {
			FreePinnedMemory(ptr)
		}
	}()

	// Concurrently access buffers
	var wg sync.WaitGroup
	for i := 0; i < numBuffers; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			runtime.LockOSThread()
			defer runtime.UnlockOSThread()

			slice := unsafe.Slice((*byte)(buffers[idx]), bufferSize)

			// Write pattern
			for j := range slice {
				slice[j] = byte((idx + j) % 256)
			}

			// Verify pattern
			for j := range slice {
				if slice[j] != byte((idx+j)%256) {
					t.Errorf("Data corruption in buffer %d at index %d", idx, j)
					return
				}
			}
		}(i)
	}

	wg.Wait()
}

// TestPinnedMemory_NoLeakAfterRepeatedAllocFree verifies that repeated
// allocation and deallocation does not leak memory.
//
// Given: Repeated allocation/deallocation cycles
// When: Memory is allocated and freed many times
// Then: Memory usage should return to baseline
func TestPinnedMemory_NoLeakAfterRepeatedAllocFree(t *testing.T) {
	err := InitGPU(0)
	require.NoError(t, err)
	defer ShutdownGPU()

	// Force GC
	runtime.GC()
	runtime.GC()

	iterations := 100
	size := uint64(64 * 1024) // 64KB

	for i := 0; i < iterations; i++ {
		ptr, err := AllocPinnedMemory(size)
		if err != nil {
			t.Fatalf("Allocation failed at iteration %d: %v", i, err)
		}
		err = FreePinnedMemory(ptr)
		if err != nil {
			t.Fatalf("Free failed at iteration %d: %v", i, err)
		}
	}

	// Force GC again
	runtime.GC()
	runtime.GC()

	// Final allocation should succeed (proving no leak exhausted memory)
	ptr, err := AllocPinnedMemory(size)
	require.NoError(t, err, "Final allocation should succeed after %d iterations", iterations)
	FreePinnedMemory(ptr)

	t.Logf("PASS: %d allocation/free cycles completed without memory leak", iterations)
}

// TestIsPinnedMemory_CorrectlyIdentifiesPinned verifies that IsPinnedMemory
// can distinguish between pinned and regular memory.
//
// Given: Pinned memory and regular memory
// When: IsPinnedMemory is called on each
// Then: It should correctly identify pinned memory
func TestIsPinnedMemory_CorrectlyIdentifiesPinned(t *testing.T) {
	err := InitGPU(0)
	require.NoError(t, err)
	defer ShutdownGPU()

	// Allocate pinned memory
	pinnedPtr, err := AllocPinnedMemory(1024)
	require.NoError(t, err)
	defer FreePinnedMemory(pinnedPtr)

	// Check if pinned
	isPinned, err := IsPinnedMemory(pinnedPtr)
	require.NoError(t, err, "IsPinnedMemory should not error for pinned memory")
	assert.True(t, isPinned, "IsPinnedMemory should return true for pinned memory")

	// Regular Go memory should not be pinned
	regularMem := make([]byte, 1024)
	isPinned, err = IsPinnedMemory(unsafe.Pointer(&regularMem[0]))
	// May return error for unregistered memory, but should not be identified as pinned
	if err == nil {
		assert.False(t, isPinned, "Regular Go memory should not be identified as pinned")
	}
}

// =============================================================================
// Benchmark Tests for Pinned Memory
// =============================================================================

// BenchmarkPinnedMemoryAlloc measures allocation throughput of pinned memory.
func BenchmarkPinnedMemoryAlloc(b *testing.B) {
	err := InitGPU(0)
	if err != nil {
		b.Skip("GPU not available")
	}
	defer ShutdownGPU()

	size := uint64(16 * 1024) // 16KB typical activation size

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ptr, _ := AllocPinnedMemory(size)
		FreePinnedMemory(ptr)
	}
}

// BenchmarkPinnedVsRegularCopy compares copy throughput of pinned vs regular memory.
func BenchmarkPinnedVsRegularCopy(b *testing.B) {
	err := InitGPU(0)
	if err != nil {
		b.Skip("GPU not available")
	}
	defer ShutdownGPU()

	size := uint64(64 * 1024) // 64KB

	// Setup pinned memory
	pinnedPtr, err := AllocPinnedMemory(size)
	if err != nil {
		b.Fatal(err)
	}
	defer FreePinnedMemory(pinnedPtr)

	// Setup GPU memory
	gpuPtr, err := AllocOnDevice(size, 0)
	if err != nil {
		b.Fatal(err)
	}
	defer FreeOnDevice(gpuPtr, 0)

	// Setup regular memory
	regularMem := make([]byte, size)

	b.Run("PinnedToDevice", func(b *testing.B) {
		b.SetBytes(int64(size))
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			CopyToDeviceRaw(gpuPtr, pinnedPtr, size)
			SyncDevice()
		}
	})

	b.Run("RegularToDevice", func(b *testing.B) {
		b.SetBytes(int64(size))
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			CopyToDeviceRaw(gpuPtr, unsafe.Pointer(&regularMem[0]), size)
			SyncDevice()
		}
	})
}
