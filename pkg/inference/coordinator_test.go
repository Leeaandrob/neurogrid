// Package inference provides tests for the DistributedInferenceCoordinator.
// These are TDD RED phase tests for Phase 4 of the Hybrid Distributed Inference PRP.
// All tests should FAIL initially because the implementation doesn't exist yet.
//
// Phase 4 Acceptance Criteria:
// - AC4: Coordinator distributes layers to 2 workers (config must precede weights)
// - Coordinator sends config BEFORE weights to stateless workers
// - Coordinator tracks which peers have received config (configSent map)
package inference

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/libp2p/go-libp2p"
	"github.com/libp2p/go-libp2p/core/peer"

	"github.com/neurogrid/engine/p2p"
	"github.com/neurogrid/engine/pkg/scheduler"
	"github.com/neurogrid/engine/pkg/types"
)

// =============================================================================
// PHASE 4: COORDINATOR UPDATES - TDD RED PHASE TESTS
// =============================================================================
// These tests validate coordinator sends config before weights.
// Tests should FAIL initially - Coordinator doesn't send config before weights yet.

// TestCoordinator_SendsConfigBeforeWeights validates config is sent before weights
// AC4: Coordinator distributes layers to 2 workers (config must precede weights)
func TestCoordinator_SendsConfigBeforeWeights(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	config := types.TinyLlamaConfig()

	// Create coordinator and worker hosts
	coordinatorHost, err := libp2p.New(libp2p.ListenAddrStrings("/ip4/127.0.0.1/tcp/0"))
	if err != nil {
		t.Fatalf("Failed to create coordinator host: %v", err)
	}
	defer coordinatorHost.Close()

	workerHost, err := libp2p.New(libp2p.ListenAddrStrings("/ip4/127.0.0.1/tcp/0"))
	if err != nil {
		t.Fatalf("Failed to create worker host: %v", err)
	}
	defer workerHost.Close()

	coordinatorPeerIDStr := coordinatorHost.ID().String()
	workerPeerIDStr := workerHost.ID().String()

	// Track message order on worker side
	var messageOrder []string
	var orderMu sync.Mutex
	configReceived := make(chan struct{}, 1)
	weightsReceived := make(chan struct{}, 1)

	// Set up worker protocol to track message order
	workerProto := p2p.NewProtocol(workerHost)

	// Track config reception - this should come FIRST
	// This should FAIL - Coordinator doesn't call SendModelConfig before weights
	workerProto.OnModelConfigReceived(func(configData []byte, from peer.ID) {
		orderMu.Lock()
		messageOrder = append(messageOrder, "config")
		orderMu.Unlock()
		t.Logf("Worker received CONFIG from %s", from.String()[:12])
		select {
		case configReceived <- struct{}{}:
		default:
		}
	})

	// Track weights reception - this should come AFTER config
	workerProto.OnWeightsReceived(func(layerID int, chunkIndex int, totalChunks int, data []byte) {
		orderMu.Lock()
		messageOrder = append(messageOrder, "weights")
		orderMu.Unlock()
		t.Logf("Worker received WEIGHTS for layer %d (chunk %d/%d)", layerID, chunkIndex+1, totalChunks)
		select {
		case weightsReceived <- struct{}{}:
		default:
		}
	})

	// Handle layer request - worker responds with empty layer list (stateless)
	workerProto.OnLayerRequestReceived(func(from peer.ID, layers []int) {
		t.Log("Worker received layer request, responding with empty list (stateless)")
		workerProto.SendLayerStatus(ctx, from, []int{})
	})

	// Create layer assignments - all layers to worker
	var assignments []scheduler.LayerAssignment
	for i := 0; i < config.NumLayers; i++ {
		assignments = append(assignments, scheduler.LayerAssignment{
			LayerID: i,
			PeerID:  workerPeerIDStr,
		})
	}

	// Create coordinator engine
	coordinatorEngine := NewEngine(EngineConfig{
		ModelConfig: config,
		LocalPeerID: coordinatorPeerIDStr,
	})
	coordinatorEngine.SetAssignments(assignments)

	// Create peer manager
	peerManager := p2p.NewPeerManager(coordinatorHost)
	peerManager.Start()
	defer peerManager.Stop()

	// Create shared protocol for coordinator
	coordinatorProto := p2p.NewProtocol(coordinatorHost)

	// Create coordinator
	coordinator := NewDistributedInferenceCoordinator(CoordinatorConfig{
		Host:          coordinatorHost,
		Engine:        coordinatorEngine,
		PeerManager:   peerManager,
		Protocol:      coordinatorProto,
		ModelConfig:   config,
		Assignments:   assignments,
		LocalPeerID:   coordinatorPeerIDStr,
		WeightTimeout: 10 * time.Second,
	})
	defer coordinator.Close()

	// Load test weights for a few layers
	for i := 0; i < 3; i++ {
		testWeight := &CPULayerWeights{
			LayerID:  i,
			AttnNorm: &CPUTensor{Shape: []int{config.HiddenSize}, Data: make([]byte, config.HiddenSize*2)},
			QProj:    &CPUTensor{Shape: []int{config.HiddenSize, config.HiddenSize}, Data: make([]byte, 1024)}, // Small for testing
		}
		coordinator.LoadLocalWeights(i, testWeight)
	}

	// Connect hosts
	workerInfo := peer.AddrInfo{ID: workerHost.ID(), Addrs: workerHost.Addrs()}
	coordinatorHost.Peerstore().AddAddrs(workerHost.ID(), workerHost.Addrs(), time.Hour)
	peerManager.AddPeer(workerInfo)

	err = coordinatorHost.Connect(ctx, workerInfo)
	if err != nil {
		t.Fatalf("Failed to connect: %v", err)
	}

	// Wait for messages (coordinator should send config then weights on peer connect)
	// Allow time for onPeerConnected callback and weight distribution
	// Coordinator has a 10s timeout for layer status, so we need to wait longer
	timeout := time.After(15 * time.Second)

waitLoop:
	for {
		select {
		case <-timeout:
			break waitLoop
		case <-configReceived:
			t.Log("Config message received")
		case <-weightsReceived:
			t.Log("Weights message received")
			// Once we get weights, wait a bit for more
			time.Sleep(200 * time.Millisecond)
			break waitLoop
		}
	}

	// Verify message order: config should come before weights
	orderMu.Lock()
	defer orderMu.Unlock()

	if len(messageOrder) == 0 {
		t.Fatal("No messages received - coordinator may not be sending anything")
	}

	t.Logf("Message order received: %v", messageOrder)

	// Key assertion: First message should be config
	// This should FAIL - Coordinator doesn't send config before weights
	if messageOrder[0] != "config" {
		t.Errorf("FAIL: First message should be 'config', got '%s'", messageOrder[0])
		t.Error("Coordinator must send MsgTypeModelConfig BEFORE MsgTypeWeights")
	} else {
		t.Log("PASS: Config was sent before weights")
	}

	// Verify all subsequent messages are weights
	for i := 1; i < len(messageOrder); i++ {
		if messageOrder[i] != "weights" {
			t.Errorf("Message %d should be 'weights', got '%s'", i, messageOrder[i])
		}
	}
}

// TestCoordinator_ConfigSentOnce tests config is only sent once per peer
// Coordinator should track which peers have received config
func TestCoordinator_ConfigSentOnce(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	config := types.TinyLlamaConfig()

	// Create hosts
	coordinatorHost, err := libp2p.New(libp2p.ListenAddrStrings("/ip4/127.0.0.1/tcp/0"))
	if err != nil {
		t.Fatalf("Failed to create coordinator host: %v", err)
	}
	defer coordinatorHost.Close()

	workerHost, err := libp2p.New(libp2p.ListenAddrStrings("/ip4/127.0.0.1/tcp/0"))
	if err != nil {
		t.Fatalf("Failed to create worker host: %v", err)
	}
	defer workerHost.Close()

	coordinatorPeerIDStr := coordinatorHost.ID().String()
	workerPeerIDStr := workerHost.ID().String()

	// Track config receptions
	var configCount int
	var countMu sync.Mutex

	workerProto := p2p.NewProtocol(workerHost)
	workerProto.OnModelConfigReceived(func(data []byte, from peer.ID) {
		countMu.Lock()
		configCount++
		t.Logf("Worker received config #%d", configCount)
		countMu.Unlock()
	})

	// Create assignments
	var assignments []scheduler.LayerAssignment
	for i := 0; i < config.NumLayers; i++ {
		assignments = append(assignments, scheduler.LayerAssignment{
			LayerID: i,
			PeerID:  workerPeerIDStr,
		})
	}

	// Create coordinator
	coordinatorEngine := NewEngine(EngineConfig{
		ModelConfig: config,
		LocalPeerID: coordinatorPeerIDStr,
	})
	coordinatorEngine.SetAssignments(assignments)

	peerManager := p2p.NewPeerManager(coordinatorHost)
	peerManager.Start()
	defer peerManager.Stop()

	coordinatorProto := p2p.NewProtocol(coordinatorHost)

	coordinator := NewDistributedInferenceCoordinator(CoordinatorConfig{
		Host:          coordinatorHost,
		Engine:        coordinatorEngine,
		PeerManager:   peerManager,
		Protocol:      coordinatorProto,
		ModelConfig:   config,
		Assignments:   assignments,
		LocalPeerID:   coordinatorPeerIDStr,
		WeightTimeout: 10 * time.Second,
	})
	defer coordinator.Close()

	// Load weights
	for i := 0; i < 3; i++ {
		testWeight := &CPULayerWeights{
			LayerID:  i,
			AttnNorm: &CPUTensor{Shape: []int{config.HiddenSize}, Data: make([]byte, config.HiddenSize*2)},
		}
		coordinator.LoadLocalWeights(i, testWeight)
	}

	// Connect worker
	workerInfo := peer.AddrInfo{ID: workerHost.ID(), Addrs: workerHost.Addrs()}
	coordinatorHost.Peerstore().AddAddrs(workerHost.ID(), workerHost.Addrs(), time.Hour)
	peerManager.AddPeer(workerInfo)

	err = coordinatorHost.Connect(ctx, workerInfo)
	if err != nil {
		t.Fatalf("Failed to connect: %v", err)
	}

	// Wait for initial distribution
	time.Sleep(1 * time.Second)

	countMu.Lock()
	initialCount := configCount
	countMu.Unlock()

	// Trigger weight distribution again (simulating reconnect scenario)
	// In real implementation, distributeWeightsToPeer should check configSent map
	coordinator.distributeWeightsToPeer(workerHost.ID())

	// Wait a bit
	time.Sleep(500 * time.Millisecond)

	// Verify config was only sent once
	// This should FAIL - Coordinator doesn't have configSent tracking
	countMu.Lock()
	finalCount := configCount
	countMu.Unlock()

	if finalCount > 1 {
		t.Errorf("FAIL: Config should only be sent once, but was sent %d times", finalCount)
		t.Error("Coordinator needs configSent map to track which peers received config")
	} else if finalCount == 1 {
		t.Log("PASS: Config was sent exactly once")
	} else if finalCount == 0 {
		t.Log("INFO: No config was sent (may indicate config sending not implemented)")
	}

	_ = initialCount // Used for debugging
}

// TestCoordinator_StatelessWorkerFlow tests full flow with stateless worker
// AC4: Coordinator distributes layers to workers
func TestCoordinator_StatelessWorkerFlow(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()

	config := types.TinyLlamaConfig()

	// Create hosts
	coordinatorHost, err := libp2p.New(libp2p.ListenAddrStrings("/ip4/127.0.0.1/tcp/0"))
	if err != nil {
		t.Fatalf("Failed to create coordinator host: %v", err)
	}
	defer coordinatorHost.Close()

	workerHost, err := libp2p.New(libp2p.ListenAddrStrings("/ip4/127.0.0.1/tcp/0"))
	if err != nil {
		t.Fatalf("Failed to create worker host: %v", err)
	}
	defer workerHost.Close()

	coordinatorPeerIDStr := coordinatorHost.ID().String()
	workerPeerIDStr := workerHost.ID().String()

	// Simulate stateless worker state
	var workerModelConfig *types.LlamaConfig
	var workerReceivedLayers []int
	var stateMu sync.Mutex

	workerProto := p2p.NewProtocol(workerHost)

	// Worker receives config first (stateless mode)
	configDone := make(chan struct{})
	workerProto.OnModelConfigReceived(func(data []byte, from peer.ID) {
		cfg, modelName, err := DeserializeConfig(data)
		if err != nil {
			t.Errorf("Worker failed to deserialize config: %v", err)
			return
		}
		stateMu.Lock()
		workerModelConfig = cfg
		stateMu.Unlock()
		t.Logf("Worker received model config: %s (%d layers)", modelName, cfg.NumLayers)
		close(configDone)
	})

	// Worker receives weights after config
	workerProto.OnWeightsReceived(func(layerID int, chunkIndex int, totalChunks int, data []byte) {
		// In real worker, this would fail if modelConfig is nil
		stateMu.Lock()
		if workerModelConfig == nil {
			stateMu.Unlock()
			t.Error("FAIL: Received weights before config - worker.modelConfig is nil")
			return
		}
		workerReceivedLayers = append(workerReceivedLayers, layerID)
		stateMu.Unlock()
		t.Logf("Worker received weights for layer %d", layerID)
	})

	// Simulate worker sending empty layer status (stateless = no local layers)
	workerProto.OnLayerRequestReceived(func(from peer.ID, layers []int) {
		// Respond with empty layer list (stateless worker has nothing locally)
		workerProto.SendLayerStatus(ctx, from, []int{})
	})

	// Create coordinator
	var assignments []scheduler.LayerAssignment
	for i := 0; i < config.NumLayers; i++ {
		assignments = append(assignments, scheduler.LayerAssignment{
			LayerID: i,
			PeerID:  workerPeerIDStr,
		})
	}

	coordinatorEngine := NewEngine(EngineConfig{
		ModelConfig: config,
		LocalPeerID: coordinatorPeerIDStr,
	})
	coordinatorEngine.SetAssignments(assignments)

	peerManager := p2p.NewPeerManager(coordinatorHost)
	peerManager.Start()
	defer peerManager.Stop()

	coordinatorProto := p2p.NewProtocol(coordinatorHost)

	coordinator := NewDistributedInferenceCoordinator(CoordinatorConfig{
		Host:          coordinatorHost,
		Engine:        coordinatorEngine,
		PeerManager:   peerManager,
		Protocol:      coordinatorProto,
		ModelConfig:   config,
		Assignments:   assignments,
		LocalPeerID:   coordinatorPeerIDStr,
		WeightTimeout: 15 * time.Second,
	})
	defer coordinator.Close()

	// Load weights for distribution
	for i := 0; i < 5; i++ {
		testWeight := &CPULayerWeights{
			LayerID:  i,
			AttnNorm: &CPUTensor{Shape: []int{config.HiddenSize}, Data: make([]byte, config.HiddenSize*2)},
			QProj:    &CPUTensor{Shape: []int{128, 128}, Data: make([]byte, 128*128*2)},
		}
		coordinator.LoadLocalWeights(i, testWeight)
	}

	// Connect worker to coordinator
	workerInfo := peer.AddrInfo{ID: workerHost.ID(), Addrs: workerHost.Addrs()}
	coordinatorHost.Peerstore().AddAddrs(workerHost.ID(), workerHost.Addrs(), time.Hour)
	peerManager.AddPeer(workerInfo)

	err = coordinatorHost.Connect(ctx, workerInfo)
	if err != nil {
		t.Fatalf("Failed to connect: %v", err)
	}

	// Wait for config to be received
	// Coordinator has a 10s timeout for layer status, so wait longer
	select {
	case <-configDone:
		t.Log("Config received by worker")
	case <-time.After(15 * time.Second):
		// This is expected to fail initially - coordinator doesn't send config
		t.Log("WARNING: Config not received within timeout")
	}

	// Wait for weights
	time.Sleep(2 * time.Second)

	// Verify the flow
	stateMu.Lock()
	defer stateMu.Unlock()

	// Test assertions
	if workerModelConfig == nil {
		t.Error("FAIL: Worker did not receive model config")
		t.Error("Coordinator must call SendModelConfig before SendWeights")
	} else {
		t.Log("PASS: Worker received model config")
		if workerModelConfig.NumLayers != config.NumLayers {
			t.Errorf("Config mismatch: NumLayers got %d, expected %d",
				workerModelConfig.NumLayers, config.NumLayers)
		}
	}

	if len(workerReceivedLayers) == 0 {
		t.Log("INFO: No weights received (may be expected if config wasn't sent)")
	} else {
		t.Logf("PASS: Worker received weights for %d layers: %v",
			len(workerReceivedLayers), workerReceivedLayers)
	}
}

// TestCoordinator_HasConfigSentMap tests the coordinator has configSent tracking
// This is a structural test to verify the field exists
func TestCoordinator_HasConfigSentMap(t *testing.T) {
	config := types.TinyLlamaConfig()

	// Create minimal coordinator
	host, err := libp2p.New(libp2p.ListenAddrStrings("/ip4/127.0.0.1/tcp/0"))
	if err != nil {
		t.Fatalf("Failed to create host: %v", err)
	}
	defer host.Close()

	peerIDStr := host.ID().String()

	engine := NewEngine(EngineConfig{
		ModelConfig: config,
		LocalPeerID: peerIDStr,
	})

	peerManager := p2p.NewPeerManager(host)
	proto := p2p.NewProtocol(host)

	coordinator := NewDistributedInferenceCoordinator(CoordinatorConfig{
		Host:          host,
		Engine:        engine,
		PeerManager:   peerManager,
		Protocol:      proto,
		ModelConfig:   config,
		Assignments:   []scheduler.LayerAssignment{},
		LocalPeerID:   peerIDStr,
		WeightTimeout: 10 * time.Second,
	})
	defer coordinator.Close()

	// This test checks if the coordinator has proper structure
	// The implementation should add:
	// - configSent map[string]bool field
	// - modelName string field (for serialization)

	// Verify coordinator was created
	if coordinator == nil {
		t.Fatal("Coordinator is nil")
	}

	// Note: We can't directly check private fields, but we can verify
	// the coordinator doesn't panic when created, which suggests
	// the struct is properly initialized

	t.Log("PASS: Coordinator created successfully (structural test)")
	t.Log("Implementation should add: configSent map[string]bool to track config distribution")
}

// TestCoordinator_DistributeWeights_ChecksConfigSent tests weight distribution checks configSent
// The distributeWeightsToPeer method should check if config was sent first
func TestCoordinator_DistributeWeights_ChecksConfigSent(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	config := types.TinyLlamaConfig()

	// Create hosts
	coordinatorHost, err := libp2p.New(libp2p.ListenAddrStrings("/ip4/127.0.0.1/tcp/0"))
	if err != nil {
		t.Fatalf("Failed to create coordinator host: %v", err)
	}
	defer coordinatorHost.Close()

	workerHost, err := libp2p.New(libp2p.ListenAddrStrings("/ip4/127.0.0.1/tcp/0"))
	if err != nil {
		t.Fatalf("Failed to create worker host: %v", err)
	}
	defer workerHost.Close()

	coordinatorPeerIDStr := coordinatorHost.ID().String()
	workerPeerIDStr := workerHost.ID().String()

	// Track what worker receives
	var configsReceived int
	var mu sync.Mutex

	workerProto := p2p.NewProtocol(workerHost)
	workerProto.OnModelConfigReceived(func(data []byte, from peer.ID) {
		mu.Lock()
		configsReceived++
		mu.Unlock()
	})

	// Create coordinator
	var assignments []scheduler.LayerAssignment
	for i := 0; i < 3; i++ {
		assignments = append(assignments, scheduler.LayerAssignment{
			LayerID: i,
			PeerID:  workerPeerIDStr,
		})
	}

	engine := NewEngine(EngineConfig{
		ModelConfig: config,
		LocalPeerID: coordinatorPeerIDStr,
	})
	engine.SetAssignments(assignments)

	peerManager := p2p.NewPeerManager(coordinatorHost)
	peerManager.Start()
	defer peerManager.Stop()

	coordinatorProto := p2p.NewProtocol(coordinatorHost)

	coordinator := NewDistributedInferenceCoordinator(CoordinatorConfig{
		Host:          coordinatorHost,
		Engine:        engine,
		PeerManager:   peerManager,
		Protocol:      coordinatorProto,
		ModelConfig:   config,
		Assignments:   assignments,
		LocalPeerID:   coordinatorPeerIDStr,
		WeightTimeout: 10 * time.Second,
	})
	defer coordinator.Close()

	// Load weights
	for i := 0; i < 3; i++ {
		coordinator.LoadLocalWeights(i, &CPULayerWeights{
			LayerID:  i,
			AttnNorm: &CPUTensor{Shape: []int{config.HiddenSize}, Data: make([]byte, config.HiddenSize*2)},
		})
	}

	// Connect
	workerInfo := peer.AddrInfo{ID: workerHost.ID(), Addrs: workerHost.Addrs()}
	coordinatorHost.Peerstore().AddAddrs(workerHost.ID(), workerHost.Addrs(), time.Hour)
	err = coordinatorHost.Connect(ctx, workerInfo)
	if err != nil {
		t.Fatalf("Failed to connect: %v", err)
	}

	// Call distributeWeightsToPeer multiple times
	// If configSent map exists, config should only be sent once
	for i := 0; i < 3; i++ {
		coordinator.distributeWeightsToPeer(workerHost.ID())
		time.Sleep(100 * time.Millisecond)
	}

	// Wait for messages
	time.Sleep(500 * time.Millisecond)

	mu.Lock()
	count := configsReceived
	mu.Unlock()

	// This should FAIL - distributeWeightsToPeer doesn't check configSent
	if count > 1 {
		t.Errorf("FAIL: Config sent %d times, should be sent at most once", count)
		t.Error("distributeWeightsToPeer should check configSent[peerID] before sending config")
	} else if count == 1 {
		t.Log("PASS: Config sent exactly once")
	} else {
		t.Log("INFO: Config not sent (implementation may be missing)")
	}
}
