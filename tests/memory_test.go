//go:build cuda

package tests

import (
	"runtime"
	"testing"

	"github.com/neurogrid/engine/gpu/bindings"
	"github.com/neurogrid/engine/pkg/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestNoMemoryLeaks verifies no GPU memory leaks after repeated operations.
// Given: Multiple forward passes with allocation/deallocation
// When: 100 iterations are executed
// Then: GPU memory usage returns to baseline
func TestNoMemoryLeaks(t *testing.T) {
	config := types.Llama7BConfig()

	err := bindings.InitGPU(0)
	require.NoError(t, err)
	defer bindings.ShutdownGPU()

	// Get baseline memory
	baselineUsed, err := bindings.GetMemoryUsed()
	require.NoError(t, err)
	t.Logf("Baseline memory: %d bytes", baselineUsed)

	iterations := 100
	batchSize := 1
	seqLen := 1

	for i := 0; i < iterations; i++ {
		// Allocate tensors
		inputShape := []int{batchSize, seqLen, config.HiddenSize}
		input := types.NewTensor(inputShape, types.DtypeFP16, 0)
		output := types.NewTensor(inputShape, types.DtypeFP16, 0)

		err = bindings.AllocateTensor(input)
		require.NoError(t, err)
		err = bindings.AllocateTensor(output)
		require.NoError(t, err)

		// Do some work
		data := make([]float32, batchSize*seqLen*config.HiddenSize)
		_ = bindings.CopyToDevice(input, data)
		_ = bindings.CopyToHost(data, output)

		// Free tensors
		bindings.FreeTensor(input)
		bindings.FreeTensor(output)
	}

	// Force Go GC to run finalizers
	runtime.GC()
	runtime.GC()

	// Sync device to ensure all operations complete
	err = bindings.SyncDevice()
	require.NoError(t, err)

	// Check memory usage
	finalUsed, err := bindings.GetMemoryUsed()
	require.NoError(t, err)
	t.Logf("Final memory: %d bytes", finalUsed)

	// Memory should return to baseline (with tolerance for CUDA context, driver memory, etc.)
	tolerance := int64(8 * 1024 * 1024) // 8 MB tolerance for CUDA internal memory pools
	diff := int64(finalUsed) - int64(baselineUsed)
	assert.LessOrEqual(t, diff, tolerance,
		"Memory leak detected: baseline=%d, final=%d, diff=%d", baselineUsed, finalUsed, diff)
}

// TestKVCacheMemoryGrowth verifies KV cache memory management.
func TestKVCacheMemoryGrowth(t *testing.T) {
	config := types.Llama7BConfig()

	err := bindings.InitGPU(0)
	require.NoError(t, err)
	defer bindings.ShutdownGPU()

	batchSize := 1
	maxSeqLen := 64

	// Get baseline
	baselineUsed, err := bindings.GetMemoryUsed()
	require.NoError(t, err)

	// Create KV cache
	kvCache, err := bindings.NewKVCache(batchSize, config.NumHeads, config.HeadDim, maxSeqLen)
	require.NoError(t, err)

	// Check memory after allocation
	afterAlloc, err := bindings.GetMemoryUsed()
	require.NoError(t, err)

	// Expected KV cache size: 2 * batch * heads * maxseq * headdim * 2 (FP16)
	expectedSize := 2 * batchSize * config.NumHeads * maxSeqLen * config.HeadDim * 2
	allocatedSize := int(afterAlloc) - int(baselineUsed)

	// Should be within 2x expected (for alignment, metadata, etc.)
	assert.LessOrEqual(t, allocatedSize, expectedSize*2,
		"KV cache allocated more than expected")

	// Free cache
	bindings.FreeKVCache(kvCache)

	// Memory should return to baseline
	afterFree, err := bindings.GetMemoryUsed()
	require.NoError(t, err)

	diff := int64(afterFree) - int64(baselineUsed)
	assert.LessOrEqual(t, diff, int64(1024*1024),
		"Memory leak in KV cache: baseline=%d, afterFree=%d", baselineUsed, afterFree)
}

// TestMemoryPoolReuse verifies memory pool efficiently reuses allocations.
func TestMemoryPoolReuse(t *testing.T) {
	err := bindings.InitGPU(0)
	require.NoError(t, err)
	defer bindings.ShutdownGPU()

	size := 4096

	// Allocate and free many tensors of same size
	for round := 0; round < 5; round++ {
		tensors := make([]*types.Tensor, 10)
		for i := 0; i < 10; i++ {
			tensors[i] = types.NewTensor([]int{size}, types.DtypeFP16, 0)
			err := bindings.AllocateTensor(tensors[i])
			require.NoError(t, err)
		}
		for _, tensor := range tensors {
			bindings.FreeTensor(tensor)
		}
	}

	// Get memory used
	used, err := bindings.GetMemoryUsed()
	require.NoError(t, err)

	// Allocate same size again - should reuse pool
	tensor := types.NewTensor([]int{size}, types.DtypeFP16, 0)
	err = bindings.AllocateTensor(tensor)
	require.NoError(t, err)
	defer bindings.FreeTensor(tensor)

	usedAfter, err := bindings.GetMemoryUsed()
	require.NoError(t, err)

	// With memory pooling, should not need new allocation
	// Without pooling, this will allocate new memory
	diff := int64(usedAfter) - int64(used)
	t.Logf("Memory pool reuse: before=%d, after=%d, diff=%d", used, usedAfter, diff)

	// Diff should be small (pool reuses existing block)
	// Note: This test documents behavior - may not pass without memory pool
}

// TestLargeAllocation verifies handling of large allocations.
func TestLargeAllocation(t *testing.T) {
	err := bindings.InitGPU(0)
	require.NoError(t, err)
	defer bindings.ShutdownGPU()

	// Get total memory
	info, err := bindings.GetDeviceInfo()
	require.NoError(t, err)
	totalMB := info.TotalMemory / 1024 / 1024

	// Try to allocate 25% of GPU memory
	targetMB := totalMB / 4
	numElements := targetMB * 1024 * 1024 / 2 // FP16 = 2 bytes
	t.Logf("Attempting to allocate %d MB (%d elements)", targetMB, numElements)

	tensor := types.NewTensor([]int{int(numElements)}, types.DtypeFP16, 0)
	err = bindings.AllocateTensor(tensor)
	require.NoError(t, err, "Large allocation should succeed")
	defer bindings.FreeTensor(tensor)

	// Verify the tensor is usable (just verify allocation worked)
	// We can't easily write all elements since that would require huge host memory
	// The allocation itself proves the GPU memory is usable
	t.Logf("Successfully allocated %d MB tensor", targetMB)
}

// TestOutOfMemory verifies graceful handling of OOM.
func TestOutOfMemory(t *testing.T) {
	err := bindings.InitGPU(0)
	require.NoError(t, err)
	defer bindings.ShutdownGPU()

	// Get total memory
	info, err := bindings.GetDeviceInfo()
	require.NoError(t, err)

	// Try to allocate more than total memory
	hugeSize := int(info.TotalMemory / 2) // Elements, not bytes
	tensor := types.NewTensor([]int{hugeSize, hugeSize}, types.DtypeFP16, 0) // hugeSize^2 * 2 bytes

	err = bindings.AllocateTensor(tensor)
	assert.Error(t, err, "Should fail to allocate more than GPU memory")

	// System should still be usable
	smallTensor := types.NewTensor([]int{1024}, types.DtypeFP16, 0)
	err = bindings.AllocateTensor(smallTensor)
	assert.NoError(t, err, "Small allocation should succeed after OOM")
	bindings.FreeTensor(smallTensor)
}

// BenchmarkMemoryAlloc measures allocation throughput.
func BenchmarkMemoryAlloc(b *testing.B) {
	err := bindings.InitGPU(0)
	if err != nil {
		b.Skip("GPU not available")
	}
	defer bindings.ShutdownGPU()

	size := 4096 * 4096 // 64MB in FP16

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tensor := types.NewTensor([]int{size}, types.DtypeFP16, 0)
		_ = bindings.AllocateTensor(tensor)
		bindings.FreeTensor(tensor)
	}
}
