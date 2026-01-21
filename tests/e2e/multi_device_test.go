// Package e2e contains end-to-end tests for the distributed inference engine.
// These tests validate the multi-GPU infrastructure according to TASK-001 acceptance criteria.
package e2e

import (
	"testing"
	"unsafe"

	"github.com/neurogrid/engine/gpu/bindings"
)

// =============================================================================
// TASK-001: Multi-Device Context Manager Tests
// Acceptance Criteria from PRP
// =============================================================================

// TestMultiDeviceInit_TwoGPUs validates Scenario 1:
// Given a system with 2+ CUDA-capable GPUs
// When cuda_multi_init is called with device IDs [0, 1]
// Then both device contexts are created successfully
// And each context has dedicated compute and transfer streams
// And P2P access matrix is populated for all device pairs
func TestMultiDeviceInit_TwoGPUs(t *testing.T) {
	// Get device count
	count, err := bindings.GetDeviceCount()
	if err != nil {
		t.Fatalf("Failed to get device count: %v", err)
	}

	if count < 2 {
		t.Skip("Test requires at least 2 GPUs, found:", count)
	}

	// Initialize multi-device manager with GPUs 0 and 1
	deviceIDs := []int{0, 1}
	err = bindings.InitMultiGPU(deviceIDs)
	if err != nil {
		t.Fatalf("cuda_multi_init failed: %v", err)
	}
	defer bindings.ShutdownMultiGPU()

	// Verify both device contexts exist
	for _, deviceID := range deviceIDs {
		ctx, err := bindings.GetDeviceContext(deviceID)
		if err != nil {
			t.Errorf("Failed to get context for device %d: %v", deviceID, err)
		}

		// Verify context has valid streams
		if ctx.ComputeStream == nil {
			t.Errorf("Device %d: compute stream is nil", deviceID)
		}
		if ctx.TransferStream == nil {
			t.Errorf("Device %d: transfer stream is nil", deviceID)
		}

		// Verify memory info is populated
		if ctx.TotalMemory == 0 {
			t.Errorf("Device %d: total memory not populated", deviceID)
		}
	}

	// Verify P2P access matrix is populated
	p2pMatrix, err := bindings.GetP2PAccessMatrix()
	if err != nil {
		t.Fatalf("Failed to get P2P access matrix: %v", err)
	}

	// Matrix should be 2x2 for 2 devices
	if len(p2pMatrix) != 2 {
		t.Errorf("P2P matrix has wrong dimensions: expected 2, got %d", len(p2pMatrix))
	}

	// Diagonal should always be false (device can't P2P with itself)
	for i := 0; i < len(p2pMatrix); i++ {
		if p2pMatrix[i][i] {
			t.Errorf("P2P matrix diagonal [%d][%d] should be false", i, i)
		}
	}

	t.Log("PASS: Multi-device initialization with 2 GPUs successful")
}

// TestMultiDeviceInit_SingleGPU validates Scenario 2:
// Given a system with 1 CUDA-capable GPU
// When cuda_multi_init is called with device ID [0]
// Then single device context is created
// And P2P matrix shows no peer access available
func TestMultiDeviceInit_SingleGPU(t *testing.T) {
	// Initialize with single GPU
	deviceIDs := []int{0}
	err := bindings.InitMultiGPU(deviceIDs)
	if err != nil {
		t.Fatalf("cuda_multi_init failed with single GPU: %v", err)
	}
	defer bindings.ShutdownMultiGPU()

	// Verify single device context exists
	ctx, err := bindings.GetDeviceContext(0)
	if err != nil {
		t.Fatalf("Failed to get context for device 0: %v", err)
	}

	// Verify context has valid streams
	if ctx.ComputeStream == nil {
		t.Error("Device 0: compute stream is nil")
	}
	if ctx.TransferStream == nil {
		t.Error("Device 0: transfer stream is nil")
	}

	// Verify P2P matrix shows no peer access (1x1 matrix with false)
	p2pMatrix, err := bindings.GetP2PAccessMatrix()
	if err != nil {
		t.Fatalf("Failed to get P2P access matrix: %v", err)
	}

	if len(p2pMatrix) != 1 {
		t.Errorf("P2P matrix has wrong dimensions for single GPU: expected 1, got %d", len(p2pMatrix))
	}

	// Single device has no peer access
	if p2pMatrix[0][0] {
		t.Error("P2P matrix should show no peer access for single device")
	}

	t.Log("PASS: Multi-device initialization with single GPU successful")
}

// TestDeviceContextFields validates that DeviceContext has all required fields
func TestDeviceContextFields(t *testing.T) {
	// Initialize with device 0
	deviceIDs := []int{0}
	err := bindings.InitMultiGPU(deviceIDs)
	if err != nil {
		t.Fatalf("cuda_multi_init failed: %v", err)
	}
	defer bindings.ShutdownMultiGPU()

	ctx, err := bindings.GetDeviceContext(0)
	if err != nil {
		t.Fatalf("Failed to get device context: %v", err)
	}

	// Validate all required fields from PRP specification
	t.Run("device_id", func(t *testing.T) {
		if ctx.DeviceID != 0 {
			t.Errorf("Expected device_id 0, got %d", ctx.DeviceID)
		}
	})

	t.Run("total_memory", func(t *testing.T) {
		if ctx.TotalMemory == 0 {
			t.Error("total_memory should be > 0")
		}
	})

	t.Run("used_memory", func(t *testing.T) {
		// used_memory can be 0 initially, but field should exist
		// Just verify it's accessible (compile-time check)
		_ = ctx.UsedMemory
	})

	t.Run("compute_stream", func(t *testing.T) {
		if ctx.ComputeStream == nil {
			t.Error("compute_stream should not be nil")
		}
	})

	t.Run("transfer_stream", func(t *testing.T) {
		if ctx.TransferStream == nil {
			t.Error("transfer_stream should not be nil")
		}
	})

	t.Run("peer_access", func(t *testing.T) {
		if ctx.PeerAccess == nil {
			t.Error("peer_access array should not be nil")
		}
	})
}

// TestMultiDeviceManagerFields validates MultiDeviceManager structure
func TestMultiDeviceManagerFields(t *testing.T) {
	deviceIDs := []int{0}
	err := bindings.InitMultiGPU(deviceIDs)
	if err != nil {
		t.Fatalf("cuda_multi_init failed: %v", err)
	}
	defer bindings.ShutdownMultiGPU()

	// Get manager info
	info, err := bindings.GetMultiDeviceManagerInfo()
	if err != nil {
		t.Fatalf("Failed to get manager info: %v", err)
	}

	t.Run("num_devices", func(t *testing.T) {
		if info.NumDevices != 1 {
			t.Errorf("Expected 1 device, got %d", info.NumDevices)
		}
	})

	t.Run("staging_buffer", func(t *testing.T) {
		// Staging buffer should be allocated for cross-device copies
		if info.StagingBufferSize == 0 {
			t.Error("staging_buffer_size should be > 0")
		}
	})
}

// TestMultiDeviceShutdown validates cuda_multi_shutdown releases all resources
func TestMultiDeviceShutdown(t *testing.T) {
	deviceIDs := []int{0}
	err := bindings.InitMultiGPU(deviceIDs)
	if err != nil {
		t.Fatalf("cuda_multi_init failed: %v", err)
	}

	// Shutdown
	err = bindings.ShutdownMultiGPU()
	if err != nil {
		t.Errorf("cuda_multi_shutdown failed: %v", err)
	}

	// After shutdown, getting context should fail
	_, err = bindings.GetDeviceContext(0)
	if err == nil {
		t.Error("GetDeviceContext should fail after shutdown")
	}
}

// =============================================================================
// TASK-002: P2P Access Detection Tests
// =============================================================================

// TestP2PAccessDetection validates P2P detection between GPU pairs
func TestP2PAccessDetection(t *testing.T) {
	count, err := bindings.GetDeviceCount()
	if err != nil {
		t.Fatalf("Failed to get device count: %v", err)
	}

	if count < 2 {
		t.Skip("P2P test requires at least 2 GPUs")
	}

	deviceIDs := []int{0, 1}
	err = bindings.InitMultiGPU(deviceIDs)
	if err != nil {
		t.Fatalf("cuda_multi_init failed: %v", err)
	}
	defer bindings.ShutdownMultiGPU()

	// Check P2P status between devices
	canAccess, err := bindings.CanAccessPeer(0, 1)
	if err != nil {
		t.Fatalf("Failed to check P2P access: %v", err)
	}

	// Log the P2P capability (result depends on hardware)
	if canAccess {
		t.Log("P2P access is AVAILABLE between GPU 0 and GPU 1")
	} else {
		t.Log("P2P access is NOT available between GPU 0 and GPU 1 (will use staged copy)")
	}

	// Verify the matrix reflects the detection
	p2pMatrix, err := bindings.GetP2PAccessMatrix()
	if err != nil {
		t.Fatalf("Failed to get P2P matrix: %v", err)
	}

	if p2pMatrix[0][1] != canAccess {
		t.Errorf("P2P matrix[0][1] doesn't match detection result")
	}
}

// =============================================================================
// TASK-003: Cross-Device Memory Copy Tests
// =============================================================================

// TestCrossDeviceCopy_SameDevice validates same-device copy
func TestCrossDeviceCopy_SameDevice(t *testing.T) {
	deviceIDs := []int{0}
	err := bindings.InitMultiGPU(deviceIDs)
	if err != nil {
		t.Fatalf("cuda_multi_init failed: %v", err)
	}
	defer bindings.ShutdownMultiGPU()

	// Allocate source and destination on same device
	size := uint64(1024 * 1024) // 1MB
	src, err := bindings.AllocOnDevice(size, 0)
	if err != nil {
		t.Fatalf("Failed to allocate source: %v", err)
	}
	defer bindings.FreeOnDevice(src, 0)

	dst, err := bindings.AllocOnDevice(size, 0)
	if err != nil {
		t.Fatalf("Failed to allocate destination: %v", err)
	}
	defer bindings.FreeOnDevice(dst, 0)

	// Copy should use regular device-to-device copy
	err = bindings.CrossDeviceCopy(dst, 0, src, 0, size)
	if err != nil {
		t.Errorf("CrossDeviceCopy on same device failed: %v", err)
	}
}

// TestCrossDeviceCopy_BetweenGPUs validates cross-device transfer
func TestCrossDeviceCopy_BetweenGPUs(t *testing.T) {
	count, err := bindings.GetDeviceCount()
	if err != nil {
		t.Fatalf("Failed to get device count: %v", err)
	}

	if count < 2 {
		t.Skip("Cross-device copy test requires at least 2 GPUs")
	}

	deviceIDs := []int{0, 1}
	err = bindings.InitMultiGPU(deviceIDs)
	if err != nil {
		t.Fatalf("cuda_multi_init failed: %v", err)
	}
	defer bindings.ShutdownMultiGPU()

	// Allocate on GPU 0
	size := uint64(16 * 1024 * 1024) // 16MB
	src, err := bindings.AllocOnDevice(size, 0)
	if err != nil {
		t.Fatalf("Failed to allocate on GPU 0: %v", err)
	}
	defer bindings.FreeOnDevice(src, 0)

	// Allocate on GPU 1
	dst, err := bindings.AllocOnDevice(size, 1)
	if err != nil {
		t.Fatalf("Failed to allocate on GPU 1: %v", err)
	}
	defer bindings.FreeOnDevice(dst, 1)

	// Initialize source data
	testData := make([]byte, size)
	for i := range testData {
		testData[i] = byte(i % 256)
	}
	err = bindings.CopyToDeviceRaw(src, unsafe.Pointer(&testData[0]), size)
	if err != nil {
		t.Fatalf("Failed to copy test data to GPU 0: %v", err)
	}

	// Cross-device copy
	err = bindings.CrossDeviceCopy(dst, 1, src, 0, size)
	if err != nil {
		t.Fatalf("CrossDeviceCopy failed: %v", err)
	}

	// Verify data on GPU 1
	resultData := make([]byte, size)
	err = bindings.CopyFromDeviceRaw(unsafe.Pointer(&resultData[0]), dst, size)
	if err != nil {
		t.Fatalf("Failed to copy result from GPU 1: %v", err)
	}

	// Compare
	for i := 0; i < int(size); i++ {
		if testData[i] != resultData[i] {
			t.Errorf("Data mismatch at index %d: expected %d, got %d", i, testData[i], resultData[i])
			break
		}
	}

	t.Log("PASS: Cross-device copy data integrity verified")
}

// =============================================================================
// TASK-004: Multi-Device Allocation Tests
// =============================================================================

// TestAllocOnDevice validates cuda_alloc_on_device
func TestAllocOnDevice(t *testing.T) {
	deviceIDs := []int{0}
	err := bindings.InitMultiGPU(deviceIDs)
	if err != nil {
		t.Fatalf("cuda_multi_init failed: %v", err)
	}
	defer bindings.ShutdownMultiGPU()

	// Get initial memory usage
	ctx, _ := bindings.GetDeviceContext(0)
	initialUsed := ctx.UsedMemory

	// Allocate 1MB
	size := uint64(1024 * 1024)
	ptr, err := bindings.AllocOnDevice(size, 0)
	if err != nil {
		t.Fatalf("AllocOnDevice failed: %v", err)
	}
	defer bindings.FreeOnDevice(ptr, 0)

	// Verify memory tracking updated
	ctx, _ = bindings.GetDeviceContext(0)
	if ctx.UsedMemory < initialUsed+size {
		t.Errorf("Memory tracking not updated: expected increase of %d, got %d",
			size, ctx.UsedMemory-initialUsed)
	}
}

// TestAllocOnDevice_MultipleDevices validates allocation on different devices
func TestAllocOnDevice_MultipleDevices(t *testing.T) {
	count, err := bindings.GetDeviceCount()
	if err != nil {
		t.Fatalf("Failed to get device count: %v", err)
	}

	if count < 2 {
		t.Skip("Test requires at least 2 GPUs")
	}

	deviceIDs := []int{0, 1}
	err = bindings.InitMultiGPU(deviceIDs)
	if err != nil {
		t.Fatalf("cuda_multi_init failed: %v", err)
	}
	defer bindings.ShutdownMultiGPU()

	// Allocate on GPU 0
	ptr0, err := bindings.AllocOnDevice(1024*1024, 0)
	if err != nil {
		t.Errorf("Failed to allocate on GPU 0: %v", err)
	}
	defer bindings.FreeOnDevice(ptr0, 0)

	// Allocate on GPU 1
	ptr1, err := bindings.AllocOnDevice(1024*1024, 1)
	if err != nil {
		t.Errorf("Failed to allocate on GPU 1: %v", err)
	}
	defer bindings.FreeOnDevice(ptr1, 1)

	// Verify device context restoration
	// After allocation on GPU 1, we should still be able to work normally
	err = bindings.SyncDevice()
	if err != nil {
		t.Errorf("Device sync failed after multi-device allocation: %v", err)
	}
}

// TestFreeOnDevice validates cuda_free_on_device
func TestFreeOnDevice(t *testing.T) {
	deviceIDs := []int{0}
	err := bindings.InitMultiGPU(deviceIDs)
	if err != nil {
		t.Fatalf("cuda_multi_init failed: %v", err)
	}
	defer bindings.ShutdownMultiGPU()

	// Allocate
	size := uint64(1024 * 1024)
	ptr, err := bindings.AllocOnDevice(size, 0)
	if err != nil {
		t.Fatalf("AllocOnDevice failed: %v", err)
	}

	// Free should not error
	err = bindings.FreeOnDevice(ptr, 0)
	if err != nil {
		t.Errorf("FreeOnDevice failed: %v", err)
	}
}

// =============================================================================
// TASK-005: Go Bindings Integration Tests
// =============================================================================

// TestGoBindings_InitMultiGPU validates Go can call cuda_multi_init
func TestGoBindings_InitMultiGPU(t *testing.T) {
	deviceIDs := []int{0}
	err := bindings.InitMultiGPU(deviceIDs)
	if err != nil {
		t.Fatalf("InitMultiGPU binding failed: %v", err)
	}

	err = bindings.ShutdownMultiGPU()
	if err != nil {
		t.Errorf("ShutdownMultiGPU binding failed: %v", err)
	}
}

// TestGoBindings_CrossDeviceCopy validates Go can perform cross-device copies
func TestGoBindings_CrossDeviceCopy(t *testing.T) {
	count, err := bindings.GetDeviceCount()
	if err != nil {
		t.Fatalf("GetDeviceCount binding failed: %v", err)
	}

	if count < 2 {
		t.Skip("Test requires 2+ GPUs")
	}

	deviceIDs := []int{0, 1}
	err = bindings.InitMultiGPU(deviceIDs)
	if err != nil {
		t.Fatalf("InitMultiGPU failed: %v", err)
	}
	defer bindings.ShutdownMultiGPU()

	// Allocate
	size := uint64(1024)
	src, _ := bindings.AllocOnDevice(size, 0)
	dst, _ := bindings.AllocOnDevice(size, 1)
	defer bindings.FreeOnDevice(src, 0)
	defer bindings.FreeOnDevice(dst, 1)

	// Cross-device copy
	err = bindings.CrossDeviceCopy(dst, 1, src, 0, size)
	if err != nil {
		t.Errorf("CrossDeviceCopy binding failed: %v", err)
	}
}
