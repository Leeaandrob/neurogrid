// Package e2e provides end-to-end tests for the NeuroGrid distributed inference engine.
// Tests for TASK-015: VRAM Memory Tracker
// Tests for TASK-016: Layer Memory Estimation
// Tests for TASK-017: Layer Assignment Algorithm
// Tests for TASK-018: Prefetch Coordinator
package e2e

import (
	"sync"
	"testing"
	"time"

	"github.com/neurogrid/engine/pkg/scheduler"
)

// =============================================================================
// TASK-015: VRAM Memory Tracker Tests
// =============================================================================

// TestVRAMTracker_Creation validates tracker can be created
func TestVRAMTracker_Creation(t *testing.T) {
	tracker := scheduler.NewVRAMTracker()
	if tracker == nil {
		t.Fatal("NewVRAMTracker returned nil")
	}

	t.Log("PASS: VRAMTracker created successfully")
}

// TestVRAMTracker_RegisterPeer validates peer registration
func TestVRAMTracker_RegisterPeer(t *testing.T) {
	tracker := scheduler.NewVRAMTracker()

	// 8GB total, 2GB used
	err := tracker.RegisterPeer("gpu-0", 8*1024*1024*1024, 2*1024*1024*1024)
	if err != nil {
		t.Fatalf("RegisterPeer failed: %v", err)
	}

	info, err := tracker.GetPeerInfo("gpu-0")
	if err != nil {
		t.Fatalf("GetPeerInfo failed: %v", err)
	}

	if info.Total != 8*1024*1024*1024 {
		t.Errorf("Total VRAM mismatch: got %d", info.Total)
	}

	if info.Used != 2*1024*1024*1024 {
		t.Errorf("Used VRAM mismatch: got %d", info.Used)
	}

	t.Log("PASS: VRAMTracker RegisterPeer works")
}

// TestVRAMTracker_Reserve validates VRAM reservation
func TestVRAMTracker_Reserve(t *testing.T) {
	tracker := scheduler.NewVRAMTracker()

	// 8GB total, 2GB used = 6GB available
	tracker.RegisterPeer("gpu-0", 8*1024*1024*1024, 2*1024*1024*1024)

	// Reserve 4GB - should succeed
	err := tracker.Reserve("gpu-0", 4*1024*1024*1024)
	if err != nil {
		t.Fatalf("Reserve failed: %v", err)
	}

	// Check remaining
	info, _ := tracker.GetPeerInfo("gpu-0")
	available := info.Available()

	// Should have 2GB available (6GB - 4GB reserved)
	expected := uint64(2 * 1024 * 1024 * 1024)
	if available != expected {
		t.Errorf("Available VRAM mismatch: got %d, expected %d", available, expected)
	}

	t.Log("PASS: VRAMTracker Reserve works")
}

// TestVRAMTracker_ReserveOverAllocation validates over-allocation is rejected
func TestVRAMTracker_ReserveOverAllocation(t *testing.T) {
	tracker := scheduler.NewVRAMTracker()

	// 8GB total, 6GB used = 2GB available
	tracker.RegisterPeer("gpu-0", 8*1024*1024*1024, 6*1024*1024*1024)

	// Try to reserve 4GB - should fail
	err := tracker.Reserve("gpu-0", 4*1024*1024*1024)
	if err == nil {
		t.Error("Reserve should have failed for over-allocation")
	}

	t.Log("PASS: VRAMTracker rejects over-allocation")
}

// TestVRAMTracker_Commit validates commit operation
func TestVRAMTracker_Commit(t *testing.T) {
	tracker := scheduler.NewVRAMTracker()
	tracker.RegisterPeer("gpu-0", 8*1024*1024*1024, 0)

	// Reserve 2GB
	tracker.Reserve("gpu-0", 2*1024*1024*1024)

	// Commit 2GB
	err := tracker.Commit("gpu-0", 2*1024*1024*1024)
	if err != nil {
		t.Fatalf("Commit failed: %v", err)
	}

	info, _ := tracker.GetPeerInfo("gpu-0")
	if info.Used != 2*1024*1024*1024 {
		t.Errorf("Used VRAM after commit mismatch: got %d", info.Used)
	}
	if info.Reserved != 0 {
		t.Errorf("Reserved VRAM after commit should be 0: got %d", info.Reserved)
	}

	t.Log("PASS: VRAMTracker Commit works")
}

// TestVRAMTracker_Release validates release operation
func TestVRAMTracker_Release(t *testing.T) {
	tracker := scheduler.NewVRAMTracker()
	tracker.RegisterPeer("gpu-0", 8*1024*1024*1024, 4*1024*1024*1024)

	// Release 2GB
	err := tracker.Release("gpu-0", 2*1024*1024*1024)
	if err != nil {
		t.Fatalf("Release failed: %v", err)
	}

	info, _ := tracker.GetPeerInfo("gpu-0")
	if info.Used != 2*1024*1024*1024 {
		t.Errorf("Used VRAM after release mismatch: got %d", info.Used)
	}

	t.Log("PASS: VRAMTracker Release works")
}

// TestVRAMTracker_Concurrent validates thread safety
func TestVRAMTracker_Concurrent(t *testing.T) {
	tracker := scheduler.NewVRAMTracker()
	tracker.RegisterPeer("gpu-0", 100*1024*1024*1024, 0) // 100GB to avoid contention

	var wg sync.WaitGroup
	numGoroutines := 10

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < 100; j++ {
				tracker.Reserve("gpu-0", 1*1024*1024) // 1MB
				tracker.Commit("gpu-0", 1*1024*1024)  // 1MB
				tracker.Release("gpu-0", 1*1024*1024) // 1MB
				tracker.GetPeerInfo("gpu-0")
			}
		}()
	}

	wg.Wait()

	// Should be back to 0 used
	info, _ := tracker.GetPeerInfo("gpu-0")
	if info.Used != 0 {
		t.Errorf("Used VRAM should be 0 after concurrent ops: got %d", info.Used)
	}

	t.Log("PASS: VRAMTracker is thread-safe")
}

// =============================================================================
// TASK-016: Layer Memory Estimation Tests
// =============================================================================

// TestScheduler_EstimateLayerMemory_7B validates 7B layer estimate
func TestScheduler_EstimateLayerMemory_7B(t *testing.T) {
	config := scheduler.DefaultLlama7BConfig()
	sched := scheduler.NewScheduler(config)

	layerMem := sched.EstimateLayerMemory()

	// Expected: ~200-350MB per layer for 7B (including KV cache and activations)
	minExpected := uint64(150 * 1024 * 1024) // 150MB
	maxExpected := uint64(400 * 1024 * 1024) // 400MB

	if layerMem < minExpected || layerMem > maxExpected {
		t.Errorf("Layer memory estimate out of range: got %d MB, expected 150-300MB",
			layerMem/(1024*1024))
	}

	t.Logf("PASS: 7B layer memory estimate: %d MB", layerMem/(1024*1024))
}

// TestScheduler_EstimateLayerMemory_13B validates 13B layer estimate
func TestScheduler_EstimateLayerMemory_13B(t *testing.T) {
	config := scheduler.DefaultLlama13BConfig()
	sched := scheduler.NewScheduler(config)

	layerMem := sched.EstimateLayerMemory()

	// Expected: ~300-400MB per layer for 13B
	minExpected := uint64(250 * 1024 * 1024) // 250MB
	maxExpected := uint64(500 * 1024 * 1024) // 500MB

	if layerMem < minExpected || layerMem > maxExpected {
		t.Errorf("Layer memory estimate out of range: got %d MB, expected 250-500MB",
			layerMem/(1024*1024))
	}

	t.Logf("PASS: 13B layer memory estimate: %d MB", layerMem/(1024*1024))
}

// TestScheduler_TotalModelMemory_7B validates total model memory
func TestScheduler_TotalModelMemory_7B(t *testing.T) {
	config := scheduler.DefaultLlama7BConfig()
	sched := scheduler.NewScheduler(config)

	totalMem := sched.TotalModelMemory()

	// 7B INT8: ~7GB weights + overhead
	minExpected := uint64(6 * 1024 * 1024 * 1024)  // 6GB
	maxExpected := uint64(12 * 1024 * 1024 * 1024) // 12GB

	if totalMem < minExpected || totalMem > maxExpected {
		t.Errorf("Total model memory out of range: got %d GB, expected 6-12GB",
			totalMem/(1024*1024*1024))
	}

	t.Logf("PASS: 7B total model memory: %.2f GB", float64(totalMem)/(1024*1024*1024))
}

// =============================================================================
// TASK-017: Layer Assignment Algorithm Tests
// =============================================================================

// TestScheduler_ComputeAssignments_SinglePeer validates single peer assignment
func TestScheduler_ComputeAssignments_SinglePeer(t *testing.T) {
	config := scheduler.DefaultLlama7BConfig()
	config.NumLayers = 4 // Small model for testing
	sched := scheduler.NewScheduler(config)

	// Register one peer with enough VRAM for entire model
	err := sched.RegisterPeer("gpu-0", 16*1024*1024*1024, 0) // 16GB
	if err != nil {
		t.Fatalf("RegisterPeer failed: %v", err)
	}

	assignments, err := sched.ComputeAssignments()
	if err != nil {
		t.Fatalf("ComputeAssignments failed: %v", err)
	}

	// Should have embedding + 4 layers + output = 6 assignments
	if len(assignments) != 6 {
		t.Errorf("Expected 6 assignments, got %d", len(assignments))
	}

	// All should be assigned to gpu-0
	for _, a := range assignments {
		if a.PeerID != "gpu-0" {
			t.Errorf("Layer %d assigned to wrong peer: %s", a.LayerID, a.PeerID)
		}
	}

	t.Log("PASS: Single peer assignment works")
}

// TestScheduler_ComputeAssignments_MultiplePeers validates multi-peer assignment
func TestScheduler_ComputeAssignments_MultiplePeers(t *testing.T) {
	config := scheduler.DefaultLlama7BConfig()
	config.NumLayers = 4
	sched := scheduler.NewScheduler(config)

	// Register multiple peers with varying VRAM
	sched.RegisterPeer("gpu-0", 8*1024*1024*1024, 0) // 8GB
	sched.RegisterPeer("gpu-1", 4*1024*1024*1024, 0) // 4GB
	sched.RegisterPeer("gpu-2", 4*1024*1024*1024, 0) // 4GB

	assignments, err := sched.ComputeAssignments()
	if err != nil {
		t.Fatalf("ComputeAssignments failed: %v", err)
	}

	// Should have 6 assignments distributed across peers
	if len(assignments) != 6 {
		t.Errorf("Expected 6 assignments, got %d", len(assignments))
	}

	// Count layers per peer
	peerCounts := make(map[string]int)
	for _, a := range assignments {
		peerCounts[a.PeerID]++
	}

	// gpu-0 should have more layers (more VRAM)
	if peerCounts["gpu-0"] < peerCounts["gpu-1"] {
		t.Error("gpu-0 should have at least as many layers as gpu-1")
	}

	t.Logf("PASS: Multi-peer assignment: gpu-0=%d, gpu-1=%d, gpu-2=%d",
		peerCounts["gpu-0"], peerCounts["gpu-1"], peerCounts["gpu-2"])
}

// TestScheduler_ComputeAssignments_InsufficientVRAM validates error on insufficient VRAM
func TestScheduler_ComputeAssignments_InsufficientVRAM(t *testing.T) {
	config := scheduler.DefaultLlama7BConfig()
	config.NumLayers = 32
	sched := scheduler.NewScheduler(config)

	// Register peer with insufficient VRAM
	sched.RegisterPeer("gpu-0", 1*1024*1024*1024, 0) // 1GB

	_, err := sched.ComputeAssignments()
	if err == nil {
		t.Error("Expected error for insufficient VRAM")
	}

	t.Log("PASS: Insufficient VRAM returns error")
}

// TestScheduler_ComputeAssignments_HeterogeneousVRAM validates Scenario 1 from TASK-017
// Given 5 GPUs with VRAM: 24GB, 8GB, 8GB, 8GB, 8GB
// And 32 layers each requiring ~250MB
// When ComputeAssignments is called
// Then all 32 layers are assigned
// And larger GPU gets more layers
// And no GPU exceeds its VRAM limit
func TestScheduler_ComputeAssignments_HeterogeneousVRAM(t *testing.T) {
	config := scheduler.DefaultLlama7BConfig()
	config.NumLayers = 32
	sched := scheduler.NewScheduler(config)

	// Register 5 GPUs with heterogeneous VRAM: 24GB, 8GB, 8GB, 8GB, 8GB
	sched.RegisterPeer("gpu-0", 24*1024*1024*1024, 0) // 24GB
	sched.RegisterPeer("gpu-1", 8*1024*1024*1024, 0)  // 8GB
	sched.RegisterPeer("gpu-2", 8*1024*1024*1024, 0)  // 8GB
	sched.RegisterPeer("gpu-3", 8*1024*1024*1024, 0)  // 8GB
	sched.RegisterPeer("gpu-4", 8*1024*1024*1024, 0)  // 8GB

	assignments, err := sched.ComputeAssignments()
	if err != nil {
		t.Fatalf("ComputeAssignments failed: %v", err)
	}

	// Should have embedding + 32 layers + output = 34 assignments
	expectedAssignments := 34
	if len(assignments) != expectedAssignments {
		t.Errorf("Expected %d assignments, got %d", expectedAssignments, len(assignments))
	}

	// Count layers per peer and total memory per peer
	peerCounts := make(map[string]int)
	peerMemory := make(map[string]uint64)
	for _, a := range assignments {
		peerCounts[a.PeerID]++
		peerMemory[a.PeerID] += a.MemoryNeeded
	}

	// Validate: gpu-0 (24GB) should have more layers than any 8GB GPU
	for peerID, count := range peerCounts {
		if peerID == "gpu-0" {
			continue
		}
		if peerCounts["gpu-0"] < count {
			t.Errorf("gpu-0 (24GB) should have more layers than %s (8GB): gpu-0=%d, %s=%d",
				peerID, peerCounts["gpu-0"], peerID, count)
		}
	}

	// Validate: no GPU exceeds its VRAM limit
	vramLimits := map[string]uint64{
		"gpu-0": 24 * 1024 * 1024 * 1024,
		"gpu-1": 8 * 1024 * 1024 * 1024,
		"gpu-2": 8 * 1024 * 1024 * 1024,
		"gpu-3": 8 * 1024 * 1024 * 1024,
		"gpu-4": 8 * 1024 * 1024 * 1024,
	}

	for peerID, usedMemory := range peerMemory {
		limit := vramLimits[peerID]
		if usedMemory > limit {
			t.Errorf("Peer %s exceeds VRAM limit: used %d MB, limit %d MB",
				peerID, usedMemory/(1024*1024), limit/(1024*1024))
		}
	}

	t.Logf("PASS: Scenario 1 - Heterogeneous VRAM distribution:")
	for peerID, count := range peerCounts {
		t.Logf("  %s: %d layers, %d MB used", peerID, count, peerMemory[peerID]/(1024*1024))
	}
}

// TestScheduler_ComputeAssignments_InsufficientVRAM_Scenario2 validates Scenario 2 from TASK-017
// Given 2 GPUs with 4GB each
// And 80 layers each requiring 250MB (total 20GB needed)
// When ComputeAssignments is called
// Then error is returned indicating insufficient VRAM
func TestScheduler_ComputeAssignments_InsufficientVRAM_Scenario2(t *testing.T) {
	config := scheduler.DefaultLlama7BConfig()
	config.NumLayers = 80 // 80 layers requiring ~250MB each = ~20GB
	sched := scheduler.NewScheduler(config)

	// Register 2 GPUs with 4GB each = 8GB total
	sched.RegisterPeer("gpu-0", 4*1024*1024*1024, 0) // 4GB
	sched.RegisterPeer("gpu-1", 4*1024*1024*1024, 0) // 4GB

	_, err := sched.ComputeAssignments()
	if err == nil {
		t.Error("Expected error for insufficient VRAM (80 layers ~20GB, only 8GB available)")
	} else {
		t.Logf("PASS: Scenario 2 - Insufficient VRAM correctly returns error: %v", err)
	}
}

// TestScheduler_ValidateAssignments validates assignment validation
func TestScheduler_ValidateAssignments(t *testing.T) {
	config := scheduler.DefaultLlama7BConfig()
	config.NumLayers = 2
	sched := scheduler.NewScheduler(config)

	sched.RegisterPeer("gpu-0", 8*1024*1024*1024, 0)

	assignments, _ := sched.ComputeAssignments()

	err := sched.ValidateAssignments(assignments)
	if err != nil {
		t.Errorf("ValidateAssignments failed: %v", err)
	}

	t.Log("PASS: ValidateAssignments works")
}

// =============================================================================
// TASK-018: Prefetch Coordinator Tests
// =============================================================================

// TestPrefetchCoordinator_Creation validates coordinator creation
func TestPrefetchCoordinator_Creation(t *testing.T) {
	assignments := []scheduler.LayerAssignment{
		{LayerID: 0, PeerID: "gpu-0"},
		{LayerID: 1, PeerID: "gpu-1"},
	}

	pc := scheduler.NewPrefetchCoordinator(assignments, 100)
	if pc == nil {
		t.Fatal("NewPrefetchCoordinator returned nil")
	}

	t.Log("PASS: PrefetchCoordinator created successfully")
}

// TestPrefetchCoordinator_GetLayerPeer validates layer-peer lookup
func TestPrefetchCoordinator_GetLayerPeer(t *testing.T) {
	assignments := []scheduler.LayerAssignment{
		{LayerID: -1, PeerID: "gpu-0"}, // Embedding
		{LayerID: 0, PeerID: "gpu-0"},
		{LayerID: 1, PeerID: "gpu-1"},
		{LayerID: 2, PeerID: "gpu-1"},
	}

	pc := scheduler.NewPrefetchCoordinator(assignments, 100)

	peer, found := pc.GetLayerPeer(0)
	if !found || peer != "gpu-0" {
		t.Errorf("GetLayerPeer(0) failed: peer=%s, found=%v", peer, found)
	}

	peer, found = pc.GetLayerPeer(1)
	if !found || peer != "gpu-1" {
		t.Errorf("GetLayerPeer(1) failed: peer=%s, found=%v", peer, found)
	}

	_, found = pc.GetLayerPeer(99)
	if found {
		t.Error("GetLayerPeer(99) should not find anything")
	}

	t.Log("PASS: GetLayerPeer works")
}

// TestPrefetchCoordinator_NeedsPrefetch validates prefetch detection
func TestPrefetchCoordinator_NeedsPrefetch(t *testing.T) {
	assignments := []scheduler.LayerAssignment{
		{LayerID: -1, PeerID: "gpu-0"}, // Embedding
		{LayerID: 0, PeerID: "gpu-0"},
		{LayerID: 1, PeerID: "gpu-1"}, // Peer change!
		{LayerID: 2, PeerID: "gpu-1"},
	}

	pc := scheduler.NewPrefetchCoordinator(assignments, 100)

	// Layer 0 to 1 crosses peer boundary
	if !pc.NeedsPrefetch(0) {
		t.Error("NeedsPrefetch(0) should return true (0->1 crosses peer)")
	}

	// Layer 1 to 2 stays on same peer
	if pc.NeedsPrefetch(1) {
		t.Error("NeedsPrefetch(1) should return false (1->2 same peer)")
	}

	t.Log("PASS: NeedsPrefetch works")
}

// TestPrefetchCoordinator_GetPrefetchPlan validates plan generation
func TestPrefetchCoordinator_GetPrefetchPlan(t *testing.T) {
	assignments := []scheduler.LayerAssignment{
		{LayerID: -1, PeerID: "gpu-0"},
		{LayerID: 0, PeerID: "gpu-0"},
		{LayerID: 1, PeerID: "gpu-1"},
		{LayerID: 2, PeerID: "gpu-1"},
		{LayerID: 3, PeerID: "gpu-0"},
	}

	pc := scheduler.NewPrefetchCoordinator(assignments, 100)

	plan := pc.GetPrefetchPlan(4)

	// Should include layers that transition to different peer
	// -1->0 (same), 0->1 (diff), 1->2 (same), 2->3 (diff)
	expectedPrefetches := 2

	if len(plan) != expectedPrefetches {
		t.Errorf("Expected %d prefetches, got %d: %v", expectedPrefetches, len(plan), plan)
	}

	t.Logf("PASS: GetPrefetchPlan works: %v", plan)
}

// TestPrefetchCoordinator_RequestPrefetch validates prefetch request handling
func TestPrefetchCoordinator_RequestPrefetch(t *testing.T) {
	assignments := []scheduler.LayerAssignment{
		{LayerID: 0, PeerID: "gpu-0"},
		{LayerID: 1, PeerID: "gpu-1"},
	}

	pc := scheduler.NewPrefetchCoordinator(assignments, 100)
	pc.Start(2) // 2 workers
	defer pc.Stop()

	callbackCalled := make(chan bool, 1)

	req := scheduler.PrefetchRequest{
		LayerID:  1,
		PeerID:   "gpu-1",
		SeqID:    12345,
		Priority: 1,
	}

	err := pc.RequestPrefetch(req, func(status scheduler.PrefetchStatus) {
		if status.Request.SeqID != 12345 {
			t.Errorf("SeqID mismatch in callback")
		}
		callbackCalled <- true
	})

	if err != nil {
		t.Fatalf("RequestPrefetch failed: %v", err)
	}

	select {
	case <-callbackCalled:
		t.Log("PASS: Prefetch callback received")
	case <-time.After(time.Second):
		t.Error("Prefetch callback not received")
	}
}

// TestPrefetchCoordinator_IsPrefetching validates in-flight tracking
func TestPrefetchCoordinator_IsPrefetching(t *testing.T) {
	assignments := []scheduler.LayerAssignment{
		{LayerID: 0, PeerID: "gpu-0"},
	}

	pc := scheduler.NewPrefetchCoordinator(assignments, 100)
	// Don't start workers so request stays pending

	req := scheduler.PrefetchRequest{
		LayerID:  0,
		PeerID:   "gpu-0",
		SeqID:    999,
		Priority: 1,
	}

	pc.RequestPrefetch(req, nil)

	if !pc.IsPrefetching(999) {
		t.Error("IsPrefetching should return true for pending request")
	}

	pc.CancelPrefetch(999)

	if pc.IsPrefetching(999) {
		t.Error("IsPrefetching should return false after cancel")
	}

	t.Log("PASS: IsPrefetching works")
}
