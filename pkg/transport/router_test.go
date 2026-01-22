// Package transport provides unit tests for the TransportRouter.
package transport

import (
	"context"
	"sync"
	"testing"
	"time"
)

// mockTransport is a mock Transport for testing.
type mockTransport struct {
	peerInfo    PeerDescriptor
	recvChan    chan *ActivationMessage
	sentMsgs    []*ActivationMessage
	mu          sync.Mutex
	closed      bool
	sendErr     error
}

func newMockTransport(peerID string, isLocal bool, deviceID int) *mockTransport {
	return &mockTransport{
		peerInfo: PeerDescriptor{
			ID:       peerID,
			IsLocal:  isLocal,
			DeviceID: deviceID,
		},
		recvChan: make(chan *ActivationMessage, 10),
		sentMsgs: make([]*ActivationMessage, 0),
	}
}

func (m *mockTransport) SendActivation(ctx context.Context, layerID int, seqID uint64, data []byte) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.sendErr != nil {
		return m.sendErr
	}

	msg := &ActivationMessage{
		LayerID:   layerID,
		SeqID:     seqID,
		Data:      data,
		Timestamp: time.Now(),
	}
	m.sentMsgs = append(m.sentMsgs, msg)
	return nil
}

func (m *mockTransport) RecvActivation(ctx context.Context) (int, uint64, []byte, error) {
	select {
	case msg := <-m.recvChan:
		return msg.LayerID, msg.SeqID, msg.Data, nil
	case <-ctx.Done():
		return 0, 0, nil, ctx.Err()
	}
}

func (m *mockTransport) PeerInfo() PeerDescriptor {
	return m.peerInfo
}

func (m *mockTransport) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.closed = true
	close(m.recvChan)
	return nil
}

func (m *mockTransport) getSentMessages() []*ActivationMessage {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.sentMsgs
}

// TestNewTransportRouter tests router creation.
func TestNewTransportRouter(t *testing.T) {
	router := NewTransportRouter()
	if router == nil {
		t.Fatal("NewTransportRouter returned nil")
	}
	if router.localTransports == nil {
		t.Error("localTransports map not initialized")
	}
	if router.remoteTransports == nil {
		t.Error("remoteTransports map not initialized")
	}
	if router.layerToPeer == nil {
		t.Error("layerToPeer map not initialized")
	}
}

// TestRegisterLocalTransport tests local transport registration.
func TestRegisterLocalTransport(t *testing.T) {
	router := NewTransportRouter()
	mock := newMockTransport("gpu-0", true, 0)

	err := router.RegisterLocalTransport(0, mock)
	if err != nil {
		t.Fatalf("RegisterLocalTransport failed: %v", err)
	}

	// Verify transport is registered
	transport, err := router.GetLocalTransport(0)
	if err != nil {
		t.Fatalf("GetLocalTransport failed: %v", err)
	}
	if transport != mock {
		t.Error("Retrieved transport doesn't match registered transport")
	}
}

// TestRegisterRemoteTransport tests remote transport registration.
func TestRegisterRemoteTransport(t *testing.T) {
	router := NewTransportRouter()
	mock := newMockTransport("peer-A", false, -1)

	err := router.RegisterRemoteTransport("peer-A", mock)
	if err != nil {
		t.Fatalf("RegisterRemoteTransport failed: %v", err)
	}

	// Verify transport is registered
	transport, err := router.GetRemoteTransport("peer-A")
	if err != nil {
		t.Fatalf("GetRemoteTransport failed: %v", err)
	}
	if transport != mock {
		t.Error("Retrieved transport doesn't match registered transport")
	}
}

// TestAssignLayerToPeer tests layer assignment.
func TestAssignLayerToPeer(t *testing.T) {
	router := NewTransportRouter()

	err := router.AssignLayerToPeer(5, "peer-A")
	if err != nil {
		t.Fatalf("AssignLayerToPeer failed: %v", err)
	}

	peerID, err := router.GetPeerForLayer(5)
	if err != nil {
		t.Fatalf("GetPeerForLayer failed: %v", err)
	}
	if peerID != "peer-A" {
		t.Errorf("Expected peer-A, got %s", peerID)
	}
}

// TestRouteActivation_LocalTransport tests routing to local transport.
// Scenario 1: Route to local transport
// Given layer 5 is assigned to local GPU 1
// When RouteActivation is called for layer 5
// Then the local CUDA transport for GPU 1 is used
// And activation is sent successfully
func TestRouteActivation_LocalTransport(t *testing.T) {
	router := NewTransportRouter()

	// Create mock local transport for GPU 1
	mock := newMockTransport("gpu-1", true, 1)
	router.RegisterLocalTransport(1, mock)

	// Assign layer 5 to GPU 1's peer ID
	router.AssignLayerToPeer(5, "gpu-1")

	// Send activation
	ctx := context.Background()
	data := []byte("test activation data")
	err := router.RouteActivation(ctx, 5, 100, data)
	if err != nil {
		t.Fatalf("RouteActivation failed: %v", err)
	}

	// Verify message was sent to correct transport
	msgs := mock.getSentMessages()
	if len(msgs) != 1 {
		t.Fatalf("Expected 1 message, got %d", len(msgs))
	}
	if msgs[0].LayerID != 5 {
		t.Errorf("Expected layerID 5, got %d", msgs[0].LayerID)
	}
	if msgs[0].SeqID != 100 {
		t.Errorf("Expected seqID 100, got %d", msgs[0].SeqID)
	}
	if string(msgs[0].Data) != "test activation data" {
		t.Errorf("Data mismatch")
	}

	t.Log("PASS: Scenario 1 - Route to local transport works correctly")
}

// TestRouteActivation_RemoteTransport tests routing to remote transport.
// Scenario 2: Route to remote transport
// Given layer 10 is assigned to remote peer "peer-A"
// When RouteActivation is called for layer 10
// Then the P2P transport for "peer-A" is used
// And activation is sent over the network
func TestRouteActivation_RemoteTransport(t *testing.T) {
	router := NewTransportRouter()

	// Create mock remote transport
	mock := newMockTransport("peer-A", false, -1)
	router.RegisterRemoteTransport("peer-A", mock)

	// Assign layer 10 to peer-A
	router.AssignLayerToPeer(10, "peer-A")

	// Send activation
	ctx := context.Background()
	data := []byte("remote activation data")
	err := router.RouteActivation(ctx, 10, 200, data)
	if err != nil {
		t.Fatalf("RouteActivation failed: %v", err)
	}

	// Verify message was sent to correct transport
	msgs := mock.getSentMessages()
	if len(msgs) != 1 {
		t.Fatalf("Expected 1 message, got %d", len(msgs))
	}
	if msgs[0].LayerID != 10 {
		t.Errorf("Expected layerID 10, got %d", msgs[0].LayerID)
	}
	if msgs[0].SeqID != 200 {
		t.Errorf("Expected seqID 200, got %d", msgs[0].SeqID)
	}

	t.Log("PASS: Scenario 2 - Route to remote transport works correctly")
}

// TestRouteActivation_UnassignedLayer tests error handling for unassigned layers.
func TestRouteActivation_UnassignedLayer(t *testing.T) {
	router := NewTransportRouter()

	ctx := context.Background()
	err := router.RouteActivation(ctx, 99, 1, []byte("data"))
	if err == nil {
		t.Error("Expected error for unassigned layer, got nil")
	}

	t.Log("PASS: Unassigned layer returns error")
}

// TestRouteActivation_NoTransport tests error handling for missing transport.
func TestRouteActivation_NoTransport(t *testing.T) {
	router := NewTransportRouter()

	// Assign layer to peer but don't register transport
	router.AssignLayerToPeer(5, "unknown-peer")

	ctx := context.Background()
	err := router.RouteActivation(ctx, 5, 1, []byte("data"))
	if err == nil {
		t.Error("Expected error for missing transport, got nil")
	}

	t.Log("PASS: Missing transport returns error")
}

// TestUpdateLayerAssignments tests bulk assignment update.
func TestUpdateLayerAssignments(t *testing.T) {
	router := NewTransportRouter()

	assignments := map[int]string{
		0:  "peer-A",
		1:  "peer-A",
		2:  "peer-B",
		3:  "peer-B",
	}

	router.UpdateLayerAssignments(assignments)

	// Verify all assignments
	for layerID, expectedPeer := range assignments {
		peerID, err := router.GetPeerForLayer(layerID)
		if err != nil {
			t.Errorf("GetPeerForLayer(%d) failed: %v", layerID, err)
		}
		if peerID != expectedPeer {
			t.Errorf("Layer %d: expected %s, got %s", layerID, expectedPeer, peerID)
		}
	}

	t.Log("PASS: Bulk layer assignments work correctly")
}

// TestGetAllLayerAssignments tests retrieving all assignments.
func TestGetAllLayerAssignments(t *testing.T) {
	router := NewTransportRouter()

	router.AssignLayerToPeer(0, "peer-A")
	router.AssignLayerToPeer(1, "peer-B")

	assignments := router.GetAllLayerAssignments()
	if len(assignments) != 2 {
		t.Errorf("Expected 2 assignments, got %d", len(assignments))
	}
	if assignments[0] != "peer-A" {
		t.Errorf("Expected peer-A for layer 0, got %s", assignments[0])
	}
	if assignments[1] != "peer-B" {
		t.Errorf("Expected peer-B for layer 1, got %s", assignments[1])
	}
}

// TestGetRegisteredPeers tests retrieving registered peer IDs.
func TestGetRegisteredPeers(t *testing.T) {
	router := NewTransportRouter()

	router.RegisterRemoteTransport("peer-A", newMockTransport("peer-A", false, -1))
	router.RegisterRemoteTransport("peer-B", newMockTransport("peer-B", false, -1))

	peers := router.GetRegisteredPeers()
	if len(peers) != 2 {
		t.Errorf("Expected 2 peers, got %d", len(peers))
	}

	peerMap := make(map[string]bool)
	for _, p := range peers {
		peerMap[p] = true
	}
	if !peerMap["peer-A"] || !peerMap["peer-B"] {
		t.Error("Missing expected peers")
	}
}

// TestGetRegisteredDevices tests retrieving registered device IDs.
func TestGetRegisteredDevices(t *testing.T) {
	router := NewTransportRouter()

	router.RegisterLocalTransport(0, newMockTransport("gpu-0", true, 0))
	router.RegisterLocalTransport(1, newMockTransport("gpu-1", true, 1))

	devices := router.GetRegisteredDevices()
	if len(devices) != 2 {
		t.Errorf("Expected 2 devices, got %d", len(devices))
	}
}

// TestRouterClose tests proper cleanup of all transports.
func TestRouterClose(t *testing.T) {
	router := NewTransportRouter()

	mock1 := newMockTransport("gpu-0", true, 0)
	mock2 := newMockTransport("peer-A", false, -1)

	router.RegisterLocalTransport(0, mock1)
	router.RegisterRemoteTransport("peer-A", mock2)

	err := router.Close()
	if err != nil {
		t.Errorf("Close failed: %v", err)
	}

	// Verify transports were closed
	if !mock1.closed {
		t.Error("Local transport not closed")
	}
	if !mock2.closed {
		t.Error("Remote transport not closed")
	}

	t.Log("PASS: Router close cleans up all transports")
}

// TestConcurrentRouting tests thread safety of concurrent routing.
func TestConcurrentRouting(t *testing.T) {
	router := NewTransportRouter()

	// Register multiple transports
	mock1 := newMockTransport("peer-A", false, -1)
	mock2 := newMockTransport("peer-B", false, -1)
	router.RegisterRemoteTransport("peer-A", mock1)
	router.RegisterRemoteTransport("peer-B", mock2)

	// Assign layers
	router.AssignLayerToPeer(0, "peer-A")
	router.AssignLayerToPeer(1, "peer-B")

	// Concurrent routing
	var wg sync.WaitGroup
	ctx := context.Background()
	numGoroutines := 100

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			layerID := idx % 2
			router.RouteActivation(ctx, layerID, uint64(idx), []byte("data"))
		}(i)
	}

	wg.Wait()

	// Verify all messages were sent
	totalMsgs := len(mock1.getSentMessages()) + len(mock2.getSentMessages())
	if totalMsgs != numGoroutines {
		t.Errorf("Expected %d messages, got %d", numGoroutines, totalMsgs)
	}

	t.Log("PASS: Concurrent routing is thread-safe")
}
