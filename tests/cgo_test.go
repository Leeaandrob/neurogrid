//go:build cuda

package tests

import (
	"testing"
	"time"

	"github.com/neurogrid/engine/gpu/bindings"
	"github.com/neurogrid/engine/pkg/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestCGOInitShutdown verifies GPU initialization and cleanup.
func TestCGOInitShutdown(t *testing.T) {
	// First init
	err := bindings.InitGPU(0)
	require.NoError(t, err, "First GPU init failed")

	// Get device info
	info, err := bindings.GetDeviceInfo()
	require.NoError(t, err, "Failed to get device info")
	t.Logf("GPU: %s, Compute: %d.%d, Memory: %d MB",
		info.Name, info.Major, info.Minor, info.TotalMemory/1024/1024)

	// Shutdown
	bindings.ShutdownGPU()

	// Re-init should work
	err = bindings.InitGPU(0)
	require.NoError(t, err, "Re-init after shutdown failed")
	bindings.ShutdownGPU()
}

// BenchmarkCGOOverhead measures the overhead of CGO calls.
// Target: < 10μs per call
func BenchmarkCGOOverhead(b *testing.B) {
	err := bindings.InitGPU(0)
	if err != nil {
		b.Skip("GPU not available")
	}
	defer bindings.ShutdownGPU()

	// Allocate a small tensor
	tensor := types.NewTensor([]int{1}, types.DtypeFP16, 0)
	err = bindings.AllocateTensor(tensor)
	if err != nil {
		b.Fatal(err)
	}
	defer bindings.FreeTensor(tensor)

	// Measure overhead of a minimal CGO call
	b.ResetTimer()
	start := time.Now()
	for i := 0; i < b.N; i++ {
		_ = bindings.SyncDevice()
	}
	elapsed := time.Since(start)

	avgNs := float64(elapsed.Nanoseconds()) / float64(b.N)
	avgUs := avgNs / 1000

	b.ReportMetric(avgUs, "μs/call")

	// Assert target: < 10μs
	if avgUs > 10 {
		b.Logf("WARNING: CGO overhead %f μs exceeds 10μs target", avgUs)
	}
}

// BenchmarkCGOAllocFree measures allocation/deallocation overhead.
func BenchmarkCGOAllocFree(b *testing.B) {
	err := bindings.InitGPU(0)
	if err != nil {
		b.Skip("GPU not available")
	}
	defer bindings.ShutdownGPU()

	size := 4096 // Typical hidden dimension

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tensor := types.NewTensor([]int{size}, types.DtypeFP16, 0)
		_ = bindings.AllocateTensor(tensor)
		bindings.FreeTensor(tensor)
	}
}

// BenchmarkCGOCopySmall measures small data transfer overhead.
func BenchmarkCGOCopySmall(b *testing.B) {
	err := bindings.InitGPU(0)
	if err != nil {
		b.Skip("GPU not available")
	}
	defer bindings.ShutdownGPU()

	size := 128 // Small tensor
	tensor := types.NewTensor([]int{size}, types.DtypeFP16, 0)
	_ = bindings.AllocateTensor(tensor)
	defer bindings.FreeTensor(tensor)

	data := make([]float32, size)
	result := make([]float32, size)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = bindings.CopyToDevice(tensor, data)
		_ = bindings.CopyToHost(result, tensor)
	}
}

// BenchmarkCGOCopyLarge measures large data transfer overhead.
func BenchmarkCGOCopyLarge(b *testing.B) {
	err := bindings.InitGPU(0)
	if err != nil {
		b.Skip("GPU not available")
	}
	defer bindings.ShutdownGPU()

	size := 4096 * 4096 // Large tensor (64MB in FP16)
	tensor := types.NewTensor([]int{size}, types.DtypeFP16, 0)
	_ = bindings.AllocateTensor(tensor)
	defer bindings.FreeTensor(tensor)

	data := make([]float32, size)
	result := make([]float32, size)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = bindings.CopyToDevice(tensor, data)
		_ = bindings.CopyToHost(result, tensor)
	}

	// Report bandwidth
	bytes := int64(size) * 2 * 2 // FP16 both ways
	b.SetBytes(bytes)
}

// TestCGOErrorHandling verifies error propagation from CUDA to Go.
func TestCGOErrorHandling(t *testing.T) {
	err := bindings.InitGPU(0)
	require.NoError(t, err)
	defer bindings.ShutdownGPU()

	// Try to allocate with invalid device
	tensor := types.NewTensor([]int{1}, types.DtypeFP16, 999) // Invalid device
	err = bindings.AllocateTensor(tensor)
	assert.Error(t, err, "Should fail with invalid device")

	// Try to free nil tensor
	nilTensor := &types.Tensor{Data: nil}
	bindings.FreeTensor(nilTensor) // Should not panic

	// Try to copy to nil tensor
	err = bindings.CopyToDevice(nilTensor, []float32{1.0})
	assert.Error(t, err, "Should fail copying to nil tensor")
}

// TestCGOMultipleDevices tests device selection (if multiple GPUs available).
func TestCGOMultipleDevices(t *testing.T) {
	numDevices, err := bindings.GetDeviceCount()
	require.NoError(t, err)
	t.Logf("Found %d CUDA devices", numDevices)

	if numDevices < 2 {
		t.Skip("Need at least 2 GPUs for multi-device test")
	}

	// Init device 0
	err = bindings.InitGPU(0)
	require.NoError(t, err)

	tensor0 := types.NewTensor([]int{1024}, types.DtypeFP16, 0)
	err = bindings.AllocateTensor(tensor0)
	require.NoError(t, err)
	defer bindings.FreeTensor(tensor0)

	// Init device 1
	err = bindings.SetDevice(1)
	require.NoError(t, err)

	tensor1 := types.NewTensor([]int{1024}, types.DtypeFP16, 1)
	err = bindings.AllocateTensor(tensor1)
	require.NoError(t, err)
	defer bindings.FreeTensor(tensor1)

	bindings.ShutdownGPU()
}

// BenchmarkCGOKernelLaunch measures actual kernel launch overhead.
func BenchmarkCGOKernelLaunch(b *testing.B) {
	err := bindings.InitGPU(0)
	if err != nil {
		b.Skip("GPU not available")
	}
	defer bindings.ShutdownGPU()

	// Small tensor for minimal computation
	size := 256
	tensor := types.NewTensor([]int{size}, types.DtypeFP16, 0)
	_ = bindings.AllocateTensor(tensor)
	defer bindings.FreeTensor(tensor)

	out := types.NewTensor([]int{size}, types.DtypeFP16, 0)
	_ = bindings.AllocateTensor(out)
	defer bindings.FreeTensor(out)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = bindings.SiLU(out, tensor) // Minimal kernel
	}
	_ = bindings.SyncDevice()
}

// BenchmarkCGORMSNorm measures RMSNorm kernel overhead for typical LLM dimensions.
func BenchmarkCGORMSNorm(b *testing.B) {
	err := bindings.InitGPU(0)
	if err != nil {
		b.Skip("GPU not available")
	}
	defer bindings.ShutdownGPU()

	hiddenSize := 4096 // Llama 7B
	input := types.NewTensor([]int{1, 1, hiddenSize}, types.DtypeFP16, 0)
	weight := types.NewTensor([]int{hiddenSize}, types.DtypeFP16, 0)
	output := types.NewTensor([]int{1, 1, hiddenSize}, types.DtypeFP16, 0)

	_ = bindings.AllocateTensor(input)
	defer bindings.FreeTensor(input)
	_ = bindings.AllocateTensor(weight)
	defer bindings.FreeTensor(weight)
	_ = bindings.AllocateTensor(output)
	defer bindings.FreeTensor(output)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = bindings.RMSNorm(output, input, weight, 1e-6)
	}
	_ = bindings.SyncDevice()
}

// TestCGOOverheadValidation explicitly validates CGO overhead meets Phase 1 target.
func TestCGOOverheadValidation(t *testing.T) {
	err := bindings.InitGPU(0)
	require.NoError(t, err)
	defer bindings.ShutdownGPU()

	// Warm up
	for i := 0; i < 100; i++ {
		_ = bindings.SyncDevice()
	}

	// Measure CGO call overhead
	iterations := 10000
	start := time.Now()
	for i := 0; i < iterations; i++ {
		_ = bindings.SyncDevice()
	}
	elapsed := time.Since(start)

	avgNs := float64(elapsed.Nanoseconds()) / float64(iterations)
	avgUs := avgNs / 1000.0

	t.Logf("CGO Overhead: %.2f ns/call (%.3f μs/call)", avgNs, avgUs)
	t.Logf("Target: < 10 μs/call")
	t.Logf("Result: %.1fx better than target", 10.0/avgUs)

	// Phase 1 requirement: < 10μs per CGO call
	assert.Less(t, avgUs, 10.0, "CGO overhead exceeds 10μs target")
}

// TestCGOConcurrentCalls verifies thread safety of CGO bindings.
func TestCGOConcurrentCalls(t *testing.T) {
	err := bindings.InitGPU(0)
	require.NoError(t, err)
	defer bindings.ShutdownGPU()

	numGoroutines := 10
	iterations := 100

	done := make(chan bool, numGoroutines)

	for g := 0; g < numGoroutines; g++ {
		go func() {
			for i := 0; i < iterations; i++ {
				tensor := types.NewTensor([]int{256}, types.DtypeFP16, 0)
				err := bindings.AllocateTensor(tensor)
				if err != nil {
					t.Error(err)
					done <- false
					return
				}

				data := make([]float32, 256)
				_ = bindings.CopyToDevice(tensor, data)
				_ = bindings.CopyToHost(data, tensor)

				bindings.FreeTensor(tensor)
			}
			done <- true
		}()
	}

	// Wait for all goroutines
	allOk := true
	for i := 0; i < numGoroutines; i++ {
		if !<-done {
			allOk = false
		}
	}
	assert.True(t, allOk, "Concurrent CGO calls should succeed")
}
