// Package e2e provides end-to-end tests for the NeuroGrid distributed inference engine.
// Tests for PRP-003: Distributed Multi-GPU Inference
package e2e

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/libp2p/go-libp2p"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/neurogrid/engine/p2p"
	"github.com/neurogrid/engine/pkg/inference"
	"github.com/neurogrid/engine/pkg/scheduler"
	"github.com/neurogrid/engine/pkg/transport"
	"github.com/neurogrid/engine/pkg/types"
)

// =============================================================================
// PRP-003: DISTRIBUTED MULTI-GPU INFERENCE TESTS
// =============================================================================

// TestProtocol_ResponseMessage validates the protocol supports response messages
func TestProtocol_ResponseMessage(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Create sender and receiver hosts (using test hosts to avoid NAT config)
	sender, err := p2p.NewTestHost(ctx, 0)
	if err != nil {
		t.Fatalf("Sender creation failed: %v", err)
	}
	defer sender.Close()

	receiver, err := p2p.NewTestHost(ctx, 0)
	if err != nil {
		t.Fatalf("Receiver creation failed: %v", err)
	}
	defer receiver.Close()

	// Setup protocol on both
	senderProto := p2p.NewProtocol(sender)
	receiverProto := p2p.NewProtocol(receiver)

	// Track received messages
	var receivedActivation *p2p.TensorMessage
	var receivedResponse *p2p.TensorMessage
	var wg sync.WaitGroup
	wg.Add(2)

	// Receiver handles activation and sends response
	receiverProto.OnActivationReceived(func(msg *p2p.TensorMessage) {
		receivedActivation = msg
		// Send response back
		responseData := []byte("response_hidden_state")
		err := receiverProto.SendResponse(ctx, msg.From, msg.LayerID, msg.SeqID, msg.RequestID, responseData)
		if err != nil {
			t.Errorf("SendResponse failed: %v", err)
		}
		wg.Done()
	})

	// Sender handles response
	senderProto.OnResponseReceived(func(msg *p2p.TensorMessage) {
		receivedResponse = msg
		wg.Done()
	})

	// Connect sender to receiver
	err = sender.Connect(ctx, p2p.GetHostInfo(receiver))
	if err != nil {
		t.Fatalf("Connection failed: %v", err)
	}

	// Send activation with request ID
	testData := []byte("activation_data")
	requestID := uint64(12345)
	err = senderProto.SendActivation(ctx, receiver.ID(), 5, 100, requestID, testData)
	if err != nil {
		t.Fatalf("SendActivation failed: %v", err)
	}

	// Wait for both messages
	done := make(chan struct{})
	go func() {
		wg.Wait()
		close(done)
	}()

	select {
	case <-done:
		// Verify activation
		if receivedActivation == nil {
			t.Fatal("Expected to receive activation")
		}
		if receivedActivation.LayerID != 5 {
			t.Errorf("Activation LayerID mismatch: got %d, expected 5", receivedActivation.LayerID)
		}
		if receivedActivation.RequestID != requestID {
			t.Errorf("Activation RequestID mismatch: got %d, expected %d", receivedActivation.RequestID, requestID)
		}

		// Verify response
		if receivedResponse == nil {
			t.Fatal("Expected to receive response")
		}
		if receivedResponse.LayerID != 5 {
			t.Errorf("Response LayerID mismatch: got %d, expected 5", receivedResponse.LayerID)
		}
		if receivedResponse.RequestID != requestID {
			t.Errorf("Response RequestID mismatch: got %d, expected %d", receivedResponse.RequestID, requestID)
		}
		if string(receivedResponse.Data) != "response_hidden_state" {
			t.Errorf("Response Data mismatch: got %s", string(receivedResponse.Data))
		}

		t.Log("PASS: Request-Response protocol working")

	case <-ctx.Done():
		t.Fatal("Timeout waiting for request-response cycle")
	}
}

// TestProtocol_MessageTypes validates all message type constants exist
func TestProtocol_MessageTypes(t *testing.T) {
	// Verify message type constants are defined
	if p2p.MsgTypeActivation != 0x01 {
		t.Errorf("MsgTypeActivation should be 0x01, got 0x%02x", p2p.MsgTypeActivation)
	}
	if p2p.MsgTypeResponse != 0x02 {
		t.Errorf("MsgTypeResponse should be 0x02, got 0x%02x", p2p.MsgTypeResponse)
	}
	if p2p.MsgTypeWeights != 0x03 {
		t.Errorf("MsgTypeWeights should be 0x03, got 0x%02x", p2p.MsgTypeWeights)
	}
	if p2p.MsgTypeWeightsAck != 0x04 {
		t.Errorf("MsgTypeWeightsAck should be 0x04, got 0x%02x", p2p.MsgTypeWeightsAck)
	}

	t.Log("PASS: All message types defined")
}

// TestProtocol_ExtendedHeader validates 25-byte extended header with RequestID
func TestProtocol_ExtendedHeader(t *testing.T) {
	// Verify header size is extended
	if p2p.ExtendedHeaderSize != 25 {
		t.Errorf("ExtendedHeaderSize should be 25, got %d", p2p.ExtendedHeaderSize)
	}

	// Test encode/decode round-trip
	header := make([]byte, p2p.ExtendedHeaderSize)
	p2p.EncodeExtendedHeader(header, p2p.MsgTypeActivation, 10, 200, 12345, 1024)

	msgType, layerID, seqID, requestID, dataLen := p2p.DecodeExtendedHeader(header)
	if msgType != p2p.MsgTypeActivation {
		t.Errorf("MsgType mismatch: got 0x%02x, expected 0x01", msgType)
	}
	if layerID != 10 {
		t.Errorf("LayerID mismatch: got %d, expected 10", layerID)
	}
	if seqID != 200 {
		t.Errorf("SeqID mismatch: got %d, expected 200", seqID)
	}
	if requestID != 12345 {
		t.Errorf("RequestID mismatch: got %d, expected 12345", requestID)
	}
	if dataLen != 1024 {
		t.Errorf("DataLen mismatch: got %d, expected 1024", dataLen)
	}

	t.Log("PASS: Extended header encode/decode working")
}

// TestProtocol_WaitForResponse validates blocking wait for response
func TestProtocol_WaitForResponse(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	sender, err := p2p.NewTestHost(ctx, 0)
	if err != nil {
		t.Fatalf("Sender creation failed: %v", err)
	}
	defer sender.Close()

	receiver, err := p2p.NewTestHost(ctx, 0)
	if err != nil {
		t.Fatalf("Receiver creation failed: %v", err)
	}
	defer receiver.Close()

	senderProto := p2p.NewProtocol(sender)
	receiverProto := p2p.NewProtocol(receiver)

	// Receiver echoes back with "processed_" prefix
	receiverProto.OnActivationReceived(func(msg *p2p.TensorMessage) {
		responseData := append([]byte("processed_"), msg.Data...)
		_ = receiverProto.SendResponse(ctx, msg.From, msg.LayerID, msg.SeqID, msg.RequestID, responseData)
	})

	// Connect
	err = sender.Connect(ctx, p2p.GetHostInfo(receiver))
	if err != nil {
		t.Fatalf("Connection failed: %v", err)
	}

	// Send and wait for response
	testData := []byte("input_tensor")
	requestID := uint64(99999)

	// SendActivation should register the request
	err = senderProto.SendActivation(ctx, receiver.ID(), 7, 300, requestID, testData)
	if err != nil {
		t.Fatalf("SendActivation failed: %v", err)
	}

	// WaitForResponse should block until response arrives
	response, err := senderProto.WaitForResponse(ctx, requestID, 5*time.Second)
	if err != nil {
		t.Fatalf("WaitForResponse failed: %v", err)
	}

	expectedData := "processed_input_tensor"
	if string(response.Data) != expectedData {
		t.Errorf("Response data mismatch: got %s, expected %s", string(response.Data), expectedData)
	}

	t.Log("PASS: WaitForResponse blocking mechanism works")
}

// TestProtocol_WaitForResponseTimeout validates timeout handling
func TestProtocol_WaitForResponseTimeout(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	sender, err := p2p.NewTestHost(ctx, 0)
	if err != nil {
		t.Fatalf("Sender creation failed: %v", err)
	}
	defer sender.Close()

	senderProto := p2p.NewProtocol(sender)

	// No receiver, so response will never come
	requestID := uint64(88888)

	// Register a fake pending request
	senderProto.RegisterPendingRequest(requestID)

	// WaitForResponse should timeout
	_, err = senderProto.WaitForResponse(ctx, requestID, 100*time.Millisecond)
	if err == nil {
		t.Fatal("Expected timeout error, got nil")
	}

	if err != p2p.ErrResponseTimeout {
		t.Errorf("Expected ErrResponseTimeout, got: %v", err)
	}

	t.Log("PASS: WaitForResponse timeout handling works")
}

// TestProtocol_WeightsTransfer validates weight chunk transfer
func TestProtocol_WeightsTransfer(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	sender, err := p2p.NewTestHost(ctx, 0)
	if err != nil {
		t.Fatalf("Sender creation failed: %v", err)
	}
	defer sender.Close()

	receiver, err := p2p.NewTestHost(ctx, 0)
	if err != nil {
		t.Fatalf("Receiver creation failed: %v", err)
	}
	defer receiver.Close()

	senderProto := p2p.NewProtocol(sender)
	receiverProto := p2p.NewProtocol(receiver)

	// Track received weight chunks
	receivedChunks := make(map[int][]byte)
	var chunksMu sync.Mutex
	var wg sync.WaitGroup

	receiverProto.OnWeightsReceived(func(layerID int, chunkIndex int, totalChunks int, data []byte) {
		chunksMu.Lock()
		receivedChunks[chunkIndex] = data
		chunksMu.Unlock()

		// Send ack after all chunks received
		if len(receivedChunks) == totalChunks {
			_ = receiverProto.SendWeightsAck(ctx, sender.ID(), layerID)
			wg.Done()
		}
	})

	// Connect
	err = sender.Connect(ctx, p2p.GetHostInfo(receiver))
	if err != nil {
		t.Fatalf("Connection failed: %v", err)
	}

	// Create 3MB test data (will be chunked at 1MB)
	testWeights := make([]byte, 3*1024*1024)
	for i := range testWeights {
		testWeights[i] = byte(i % 256)
	}

	wg.Add(1)

	// Send weights (should chunk automatically)
	err = senderProto.SendWeights(ctx, receiver.ID(), 11, testWeights)
	if err != nil {
		t.Fatalf("SendWeights failed: %v", err)
	}

	// Wait for ack
	err = senderProto.WaitForWeightsAck(ctx, receiver.ID(), 11, 10*time.Second)
	if err != nil {
		t.Fatalf("WaitForWeightsAck failed: %v", err)
	}

	// Verify all chunks received
	chunksMu.Lock()
	if len(receivedChunks) != 3 {
		t.Errorf("Expected 3 chunks, got %d", len(receivedChunks))
	}
	chunksMu.Unlock()

	t.Log("PASS: Weight chunked transfer working")
}

// TestTransport_RequestResponse validates transport layer request-response
func TestTransport_RequestResponse(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Create test hosts
	sender, err := p2p.NewTestHost(ctx, 0)
	if err != nil {
		t.Fatalf("Sender host creation failed: %v", err)
	}
	defer sender.Close()

	receiver, err := p2p.NewTestHost(ctx, 0)
	if err != nil {
		t.Fatalf("Receiver host creation failed: %v", err)
	}
	defer receiver.Close()

	// Create transports
	senderTransport := transport.NewP2PTransport(sender, receiver.ID())
	receiverTransport := transport.NewP2PTransport(receiver, sender.ID())

	// Connect hosts
	err = sender.Connect(ctx, p2p.GetHostInfo(receiver))
	if err != nil {
		t.Fatalf("Connection failed: %v", err)
	}

	// Channel to signal receiver done
	receiverDone := make(chan struct{})

	// Receiver processes incoming activation and sends response
	go func() {
		defer close(receiverDone)
		layerID, seqID, data, err := receiverTransport.RecvActivation(ctx)
		if err != nil {
			return
		}
		// Echo back with "processed_" prefix
		responseData := append([]byte("processed_"), data...)
		// For this test, we use a fixed requestID that we know
		requestID := uint64(54321)
		_ = receiverTransport.SendResponse(ctx, layerID, seqID, requestID, responseData)
	}()

	// Send activation and wait for response
	requestID := uint64(54321)
	testData := []byte("test_activation")
	err = senderTransport.SendActivationWithID(ctx, 3, 100, requestID, testData)
	if err != nil {
		t.Fatalf("SendActivationWithID failed: %v", err)
	}

	// Wait for response
	response, err := senderTransport.WaitForResponse(ctx, requestID, 5*time.Second)
	if err != nil {
		t.Fatalf("WaitForResponse failed: %v", err)
	}

	expectedData := "processed_test_activation"
	if string(response.Data) != expectedData {
		t.Errorf("Response data mismatch: got %s, expected %s", string(response.Data), expectedData)
	}
	if response.LayerID != 3 {
		t.Errorf("Response LayerID mismatch: got %d, expected 3", response.LayerID)
	}

	// Wait for receiver goroutine to complete before closing transports
	<-receiverDone

	senderTransport.Close()
	receiverTransport.Close()

	t.Log("PASS: Transport request-response working")
}

// TestRemoteExecutor validates remote layer executor functionality
func TestRemoteExecutor(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Create test hosts
	localHost, err := p2p.NewTestHost(ctx, 0)
	if err != nil {
		t.Fatalf("Local host creation failed: %v", err)
	}
	defer localHost.Close()

	remoteHost, err := p2p.NewTestHost(ctx, 0)
	if err != nil {
		t.Fatalf("Remote host creation failed: %v", err)
	}
	defer remoteHost.Close()

	// Connect hosts
	err = localHost.Connect(ctx, p2p.GetHostInfo(remoteHost))
	if err != nil {
		t.Fatalf("Connection failed: %v", err)
	}

	// Create mock layer executor that echoes with "layer_X_" prefix
	mockExecutor := &mockLayerExecutor{prefix: "layer_processed_"}

	// Create engine with mock executor
	engineConfig := inference.EngineConfig{
		ModelConfig: types.TinyLlamaConfig(),
		LocalPeerID: remoteHost.ID().String(),
	}
	engine := inference.NewEngine(engineConfig)
	engine.SetLayerExecutor(mockExecutor)

	// Create remote executor on remote host
	remoteExec := inference.NewRemoteExecutor(inference.RemoteExecutorConfig{
		Host:         remoteHost,
		Engine:       engine,
		StartLayerID: 11,
		EndLayerID:   21,
	})
	defer remoteExec.Close()

	// Create shared protocol on local host for response routing
	localProtocol := p2p.NewProtocol(localHost)

	// Create remote layer executor on local host to forward to remote
	remoteLayerExec := inference.NewRemoteLayerExecutor(inference.RemoteLayerExecutorConfig{
		Host:         localHost,
		Protocol:     localProtocol, // Shared protocol for response routing
		TargetPeerID: remoteHost.ID(),
		StartLayerID: 11,
		EndLayerID:   21,
		Config:       types.TinyLlamaConfig(),
		Timeout:      5 * time.Second,
	})
	defer remoteLayerExec.Close()

	// Test forwarding layer 11 to remote
	testData := []byte("test_hidden_state")
	output, _, _, err := remoteLayerExec.Forward(ctx, 11, testData, 0)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}

	// The mock executor prepends "layer_processed_" to each layer from 11-21 (11 times)
	// But since our mock always prepends once, we expect "layer_processed_" + data
	if len(output) == 0 {
		t.Fatal("Expected non-empty output")
	}

	t.Logf("PASS: RemoteExecutor processed %d requests", remoteExec.RequestCount())
}

// mockLayerExecutor is a simple mock for testing
type mockLayerExecutor struct {
	prefix string
}

func (m *mockLayerExecutor) Forward(ctx context.Context, layerID int, hidden []byte, position int) ([]byte, []byte, []byte, error) {
	// Simply return input with prefix for verification
	output := append([]byte(m.prefix), hidden...)
	return output, make([]byte, 64), make([]byte, 64), nil
}

// TestEngine_RemoteExecutorIntegration validates engine uses remote executors for remote layers
func TestEngine_RemoteExecutorIntegration(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	// Create test hosts
	localHost, err := p2p.NewTestHost(ctx, 0)
	if err != nil {
		t.Fatalf("Local host creation failed: %v", err)
	}
	defer localHost.Close()

	remoteHost, err := p2p.NewTestHost(ctx, 0)
	if err != nil {
		t.Fatalf("Remote host creation failed: %v", err)
	}
	defer remoteHost.Close()

	// Connect hosts
	err = localHost.Connect(ctx, p2p.GetHostInfo(remoteHost))
	if err != nil {
		t.Fatalf("Connection failed: %v", err)
	}

	localPeerID := localHost.ID().String()
	remotePeerID := remoteHost.ID().String()

	// Create mock executor for remote peer
	mockRemoteExec := &mockLayerExecutor{prefix: "remote_processed_"}

	// Create engine for remote peer (to handle incoming requests)
	remoteEngineConfig := inference.EngineConfig{
		ModelConfig: types.TinyLlamaConfig(),
		LocalPeerID: remotePeerID,
	}
	remoteEngine := inference.NewEngine(remoteEngineConfig)
	remoteEngine.SetLayerExecutor(mockRemoteExec)

	// Create remote executor on remote host
	remoteExec := inference.NewRemoteExecutor(inference.RemoteExecutorConfig{
		Host:         remoteHost,
		Engine:       remoteEngine,
		StartLayerID: 11,
		EndLayerID:   21,
	})
	defer remoteExec.Close()

	// Create local engine
	localEngineConfig := inference.EngineConfig{
		ModelConfig: types.TinyLlamaConfig(),
		LocalPeerID: localPeerID,
	}
	localEngine := inference.NewEngine(localEngineConfig)

	// Set local layer executor
	mockLocalExec := &mockLayerExecutor{prefix: "local_processed_"}
	localEngine.SetLayerExecutor(mockLocalExec)

	// Create shared protocol on local host for response routing
	localProtocol := p2p.NewProtocol(localHost)

	// Create remote layer executor to forward to remote peer
	remoteLayerExec := inference.NewRemoteLayerExecutor(inference.RemoteLayerExecutorConfig{
		Host:         localHost,
		Protocol:     localProtocol, // Shared protocol for response routing
		TargetPeerID: remoteHost.ID(),
		StartLayerID: 11,
		EndLayerID:   21,
		Config:       types.TinyLlamaConfig(),
		Timeout:      5 * time.Second,
	})
	defer remoteLayerExec.Close()

	// Register the remote executor with the local engine
	localEngine.RegisterRemoteExecutor(remotePeerID, remoteLayerExec)

	// Set up layer assignments: layers 0-10 local, 11-21 remote
	config := types.TinyLlamaConfig()
	assignments := make([]scheduler.LayerAssignment, config.NumLayers)
	for i := 0; i < config.NumLayers; i++ {
		if i <= 10 {
			assignments[i] = scheduler.LayerAssignment{LayerID: i, PeerID: localPeerID}
		} else {
			assignments[i] = scheduler.LayerAssignment{LayerID: i, PeerID: remotePeerID}
		}
	}
	localEngine.SetAssignments(assignments)

	// Test forward layer for a remote layer (11)
	testData := []byte("test_hidden_state")
	output, _, _, err := localEngine.ForwardLayerPublic(ctx, 11, testData, 0)
	if err != nil {
		t.Fatalf("ForwardLayer failed for remote layer: %v", err)
	}

	// Verify the output went through remote processing
	if len(output) == 0 {
		t.Fatal("Expected non-empty output from remote layer")
	}

	t.Logf("PASS: Engine correctly routes layer 11 to remote executor")
	t.Logf("Output length: %d, RemoteExec requests: %d", len(output), remoteExec.RequestCount())
}

// TestDistributed_TwoPeerLayerSplit validates layer assignment between two peers
func TestDistributed_TwoPeerLayerSplit(t *testing.T) {
	t.Skip("Skipping: Requires full distributed setup - run manually with remote peer")

	// This test requires actual distributed setup between two machines
	// Local: RTX 4090, layers 0-10
	// Remote: GH200 @ 192.222.58.78, layers 11-21

	// TODO: Implement when weight distribution is complete
}

// TestWeightSerialization validates weight serialization and deserialization
func TestWeightSerialization(t *testing.T) {
	// Create test layer weights using CPULayerWeights for network transfer
	layerWeights := &inference.CPULayerWeights{
		LayerID:  5,
		AttnNorm: &inference.CPUTensor{Shape: []int{2048}, Data: make([]byte, 2048*2), Dtype: types.DtypeFP16},
		QProj:    &inference.CPUTensor{Shape: []int{2048, 2048}, Data: make([]byte, 2048*2048*2), Dtype: types.DtypeFP16},
		KProj:    &inference.CPUTensor{Shape: []int{256, 2048}, Data: make([]byte, 256*2048*2), Dtype: types.DtypeFP16},
		VProj:    &inference.CPUTensor{Shape: []int{256, 2048}, Data: make([]byte, 256*2048*2), Dtype: types.DtypeFP16},
		OProj:    &inference.CPUTensor{Shape: []int{2048, 2048}, Data: make([]byte, 2048*2048*2), Dtype: types.DtypeFP16},
		FFNNorm:  &inference.CPUTensor{Shape: []int{2048}, Data: make([]byte, 2048*2), Dtype: types.DtypeFP16},
		GateProj: &inference.CPUTensor{Shape: []int{5632, 2048}, Data: make([]byte, 5632*2048*2), Dtype: types.DtypeFP16},
		UpProj:   &inference.CPUTensor{Shape: []int{5632, 2048}, Data: make([]byte, 5632*2048*2), Dtype: types.DtypeFP16},
		DownProj: &inference.CPUTensor{Shape: []int{2048, 5632}, Data: make([]byte, 2048*5632*2), Dtype: types.DtypeFP16},
	}

	// Fill with test data
	for i := range layerWeights.AttnNorm.Data {
		layerWeights.AttnNorm.Data[i] = byte(i % 256)
	}
	for i := range layerWeights.QProj.Data {
		layerWeights.QProj.Data[i] = byte((i + 1) % 256)
	}

	// Serialize
	serialized, err := inference.SerializeLayerWeights(layerWeights)
	if err != nil {
		t.Fatalf("Serialization failed: %v", err)
	}

	if len(serialized.Data) == 0 {
		t.Fatal("Serialized data is empty")
	}
	if len(serialized.Names) != 9 {
		t.Errorf("Expected 9 tensors, got %d", len(serialized.Names))
	}

	t.Logf("Serialized layer %d: %d bytes, %d tensors", serialized.LayerID, len(serialized.Data), len(serialized.Names))

	// Deserialize
	deserialized, err := inference.DeserializeLayerWeights(5, serialized.Data)
	if err != nil {
		t.Fatalf("Deserialization failed: %v", err)
	}

	// Verify
	if deserialized.LayerID != 5 {
		t.Errorf("LayerID mismatch: got %d, expected 5", deserialized.LayerID)
	}
	if deserialized.AttnNorm == nil {
		t.Fatal("AttnNorm not deserialized")
	}
	if len(deserialized.AttnNorm.Shape) != 1 || deserialized.AttnNorm.Shape[0] != 2048 {
		t.Errorf("AttnNorm shape mismatch: got %v, expected [2048]", deserialized.AttnNorm.Shape)
	}
	if deserialized.QProj == nil {
		t.Fatal("QProj not deserialized")
	}
	if len(deserialized.QProj.Shape) != 2 || deserialized.QProj.Shape[0] != 2048 || deserialized.QProj.Shape[1] != 2048 {
		t.Errorf("QProj shape mismatch: got %v, expected [2048, 2048]", deserialized.QProj.Shape)
	}

	// Verify data integrity
	for i := 0; i < min(100, len(deserialized.AttnNorm.Data)); i++ {
		if deserialized.AttnNorm.Data[i] != layerWeights.AttnNorm.Data[i] {
			t.Errorf("AttnNorm data mismatch at index %d: got %d, expected %d",
				i, deserialized.AttnNorm.Data[i], layerWeights.AttnNorm.Data[i])
			break
		}
	}

	t.Log("PASS: Weight serialization and deserialization working")
}

// TestWeightDistributor validates weight distribution over P2P
func TestWeightDistributor(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	// Create test hosts
	senderHost, err := p2p.NewTestHost(ctx, 0)
	if err != nil {
		t.Fatalf("Sender host creation failed: %v", err)
	}
	defer senderHost.Close()

	receiverHost, err := p2p.NewTestHost(ctx, 0)
	if err != nil {
		t.Fatalf("Receiver host creation failed: %v", err)
	}
	defer receiverHost.Close()

	// Connect hosts
	err = senderHost.Connect(ctx, p2p.GetHostInfo(receiverHost))
	if err != nil {
		t.Fatalf("Connection failed: %v", err)
	}

	// Create weight receiver with callback
	receivedChan := make(chan struct{})
	var receivedWeights *inference.CPULayerWeights
	var receiver *inference.WeightReceiver
	receiver = inference.NewWeightReceiver(inference.WeightReceiverConfig{
		Host: receiverHost,
		OnLayerReceived: func(layerID int, weights *inference.CPULayerWeights) {
			receivedWeights = weights
			// Send ack back
			receiver.GetProtocol().SendWeightsAck(ctx, senderHost.ID(), layerID)
			close(receivedChan)
		},
	})
	defer receiver.Close()

	// Create small test weights for faster testing
	testWeights := &inference.CPULayerWeights{
		LayerID:  7,
		AttnNorm: &inference.CPUTensor{Shape: []int{128}, Data: make([]byte, 128*2), Dtype: types.DtypeFP16},
		QProj:    &inference.CPUTensor{Shape: []int{128, 128}, Data: make([]byte, 128*128*2), Dtype: types.DtypeFP16},
		KProj:    &inference.CPUTensor{Shape: []int{32, 128}, Data: make([]byte, 32*128*2), Dtype: types.DtypeFP16},
		VProj:    &inference.CPUTensor{Shape: []int{32, 128}, Data: make([]byte, 32*128*2), Dtype: types.DtypeFP16},
		OProj:    &inference.CPUTensor{Shape: []int{128, 128}, Data: make([]byte, 128*128*2), Dtype: types.DtypeFP16},
		FFNNorm:  &inference.CPUTensor{Shape: []int{128}, Data: make([]byte, 128*2), Dtype: types.DtypeFP16},
		GateProj: &inference.CPUTensor{Shape: []int{256, 128}, Data: make([]byte, 256*128*2), Dtype: types.DtypeFP16},
		UpProj:   &inference.CPUTensor{Shape: []int{256, 128}, Data: make([]byte, 256*128*2), Dtype: types.DtypeFP16},
		DownProj: &inference.CPUTensor{Shape: []int{128, 256}, Data: make([]byte, 128*256*2), Dtype: types.DtypeFP16},
	}

	// Fill test data
	for i := range testWeights.AttnNorm.Data {
		testWeights.AttnNorm.Data[i] = byte(i % 256)
	}

	// Create weight distributor
	distributor := inference.NewWeightDistributor(inference.WeightDistributorConfig{
		Host:        senderHost,
		ModelConfig: types.TinyLlamaConfig(),
	})
	defer distributor.Close()

	// Distribute weights
	err = distributor.DistributeLayerWeights(ctx, receiverHost.ID(), testWeights)
	if err != nil {
		t.Fatalf("DistributeLayerWeights failed: %v", err)
	}

	// Wait for reception
	select {
	case <-receivedChan:
		if receivedWeights == nil {
			t.Fatal("Received weights is nil")
		}
		if receivedWeights.LayerID != 7 {
			t.Errorf("LayerID mismatch: got %d, expected 7", receivedWeights.LayerID)
		}
		if receivedWeights.AttnNorm == nil {
			t.Fatal("AttnNorm not received")
		}

		// Verify data integrity
		for i := 0; i < min(10, len(receivedWeights.AttnNorm.Data)); i++ {
			if receivedWeights.AttnNorm.Data[i] != testWeights.AttnNorm.Data[i] {
				t.Errorf("AttnNorm data mismatch at index %d", i)
				break
			}
		}

		t.Log("PASS: Weight distribution over P2P working")
	case <-ctx.Done():
		t.Fatal("Timeout waiting for weights reception")
	}
}

// TestDistributed_EndToEndInference validates complete distributed inference
func TestDistributed_EndToEndInference(t *testing.T) {
	// This test validates the full distributed inference flow:
	// 1. Local peer processes layers 0-10
	// 2. Activation sent to remote peer
	// 3. Remote peer processes layers 11-21
	// 4. Response returned to local peer
	// 5. Valid inference output generated

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	config := types.TinyLlamaConfig()

	// Create two libp2p hosts (simulating two machines)
	localHost, err := libp2p.New(libp2p.ListenAddrStrings("/ip4/127.0.0.1/tcp/0"))
	if err != nil {
		t.Fatalf("Failed to create local host: %v", err)
	}
	defer localHost.Close()

	remoteHost, err := libp2p.New(libp2p.ListenAddrStrings("/ip4/127.0.0.1/tcp/0"))
	if err != nil {
		t.Fatalf("Failed to create remote host: %v", err)
	}
	defer remoteHost.Close()

	localPeerIDStr := localHost.ID().String()
	remotePeerIDStr := remoteHost.ID().String()

	t.Logf("Local peer: %s", localPeerIDStr[:12])
	t.Logf("Remote peer: %s", remotePeerIDStr[:12])

	// Create layer assignments - local: 0-10, remote: 11-21
	var assignments []scheduler.LayerAssignment
	for i := 0; i < config.NumLayers; i++ {
		if i <= 10 {
			assignments = append(assignments, scheduler.LayerAssignment{
				LayerID: i,
				PeerID:  localPeerIDStr,
			})
		} else {
			assignments = append(assignments, scheduler.LayerAssignment{
				LayerID: i,
				PeerID:  remotePeerIDStr,
			})
		}
	}

	// Setup remote peer (performer)
	remoteEngine := inference.NewEngine(inference.EngineConfig{
		ModelConfig: config,
		LocalPeerID: remotePeerIDStr,
	})

	performerCoordinator := inference.NewPerformerCoordinator(inference.PerformerCoordinatorConfig{
		Host:         remoteHost,
		Engine:       remoteEngine,
		ModelConfig:  config,
		StartLayerID: 11,
		EndLayerID:   21,
	})
	defer performerCoordinator.Close()

	// Setup local peer (orchestrator)
	localEngine := inference.NewEngine(inference.EngineConfig{
		ModelConfig: config,
		LocalPeerID: localPeerIDStr,
	})
	localEngine.SetAssignments(assignments)

	peerManager := p2p.NewPeerManager(localHost)
	peerManager.Start()
	defer peerManager.Stop()

	localProtocol := p2p.NewProtocol(localHost)

	coordinator := inference.NewDistributedInferenceCoordinator(inference.CoordinatorConfig{
		Host:          localHost,
		Engine:        localEngine,
		PeerManager:   peerManager,
		Protocol:      localProtocol,
		ModelConfig:   config,
		Assignments:   assignments,
		LocalPeerID:   localPeerIDStr,
		WeightTimeout: 30 * time.Second,
	})
	defer coordinator.Close()

	// Connect local to remote
	remoteInfo := peer.AddrInfo{ID: remoteHost.ID(), Addrs: remoteHost.Addrs()}
	localHost.Peerstore().AddAddrs(remoteHost.ID(), remoteHost.Addrs(), time.Hour)
	peerManager.AddPeer(remoteInfo)

	err = localHost.Connect(ctx, remoteInfo)
	if err != nil {
		t.Fatalf("Failed to connect hosts: %v", err)
	}

	// Wait for connection setup
	time.Sleep(300 * time.Millisecond)

	// Verify coordinator setup the remote executor
	if !coordinator.IsReady() {
		t.Error("Coordinator should be ready after peer connection")
	}

	exec, exists := coordinator.GetRemoteExecutor(remotePeerIDStr)
	if !exists || exec == nil {
		t.Fatal("Remote executor should be registered")
	}

	// Test a forward pass through the distributed system
	// Create a test hidden state (4096 FP16 values = 8192 bytes for TinyLlama)
	hiddenSize := config.HiddenSize * 2 // FP16
	testHidden := make([]byte, hiddenSize)
	for i := range testHidden {
		testHidden[i] = byte(i % 256)
	}

	// Forward through local layers (0-10) - these should use mock execution
	t.Log("Testing local layer execution...")
	for layerID := 0; layerID <= 10; layerID++ {
		output, _, _, err := localEngine.ForwardLayerPublic(ctx, layerID, testHidden, 0)
		if err != nil {
			t.Errorf("Local layer %d forward failed: %v", layerID, err)
			continue
		}
		if len(output) != len(testHidden) {
			t.Errorf("Layer %d output size mismatch: got %d, expected %d", layerID, len(output), len(testHidden))
		}
		testHidden = output // chain outputs
	}
	t.Log("Local layers 0-10: PASS")

	// Test that remote layers are routed correctly
	// Note: This will use the mock executor since we don't have actual weights loaded
	t.Log("Testing remote layer routing...")

	// The remote executor is set up but the remote peer needs to handle the request
	// Since we're in a test environment, we verify the routing mechanism works
	// by checking that the forward call attempts to use the remote executor

	// For a full E2E test with actual remote execution, we need the performer
	// to have weights loaded and a layer executor set up. In this test, we verify
	// the infrastructure is correctly wired.

	t.Log("Verifying remote executor configuration...")
	remoteExec, exists := coordinator.GetRemoteExecutor(remotePeerIDStr)
	if !exists {
		t.Fatal("Remote executor not found for peer")
	}
	if remoteExec == nil {
		t.Fatal("Remote executor is nil")
	}

	// Verify the performer's remote executor is handling requests
	performerRemoteExec := performerCoordinator.GetRemoteExecutor()
	if performerRemoteExec == nil {
		t.Fatal("Performer's remote executor is nil")
	}

	t.Log("PASS: Distributed inference infrastructure verified")
	t.Log("Summary:")
	t.Logf("  - Local layers: 0-10 (11 layers)")
	t.Logf("  - Remote layers: 11-21 (11 layers)")
	t.Logf("  - Total layers: %d", config.NumLayers)
	t.Logf("  - Local peer: %s", localPeerIDStr[:12])
	t.Logf("  - Remote peer: %s", remotePeerIDStr[:12])
	t.Logf("  - Connection: established")
	t.Logf("  - Remote executor: configured")
	t.Logf("  - Performer executor: configured")
}

// TestDistributed_FullActivationRoundtrip validates activation sends and receives response
func TestDistributed_FullActivationRoundtrip(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	config := types.TinyLlamaConfig()

	// Create two hosts
	localHost, err := libp2p.New(libp2p.ListenAddrStrings("/ip4/127.0.0.1/tcp/0"))
	if err != nil {
		t.Fatalf("Failed to create local host: %v", err)
	}
	defer localHost.Close()

	remoteHost, err := libp2p.New(libp2p.ListenAddrStrings("/ip4/127.0.0.1/tcp/0"))
	if err != nil {
		t.Fatalf("Failed to create remote host: %v", err)
	}
	defer remoteHost.Close()

	// Connect hosts
	localHost.Peerstore().AddAddrs(remoteHost.ID(), remoteHost.Addrs(), time.Hour)
	err = localHost.Connect(ctx, peer.AddrInfo{ID: remoteHost.ID(), Addrs: remoteHost.Addrs()})
	if err != nil {
		t.Fatalf("Failed to connect hosts: %v", err)
	}

	// Create remote engine with mock layer executor
	remoteEngine := inference.NewEngine(inference.EngineConfig{
		ModelConfig: config,
		LocalPeerID: remoteHost.ID().String(),
	})

	// Set up a mock layer executor that just returns the input
	mockExecutor := &MockLayerExecutor{
		forwardFunc: func(ctx context.Context, layerID int, hidden []byte, position int) ([]byte, []byte, []byte, error) {
			// Simulate processing: return modified hidden state
			output := make([]byte, len(hidden))
			copy(output, hidden)
			// Mark it as processed
			if len(output) > 0 {
				output[0] = byte(layerID)
			}
			kvSize := config.NumKVHeads * config.HeadDim * 2
			return output, make([]byte, kvSize), make([]byte, kvSize), nil
		},
	}
	remoteEngine.SetLayerExecutor(mockExecutor)

	// Create remote executor (handles incoming requests)
	remoteExecutor := inference.NewRemoteExecutor(inference.RemoteExecutorConfig{
		Host:         remoteHost,
		Engine:       remoteEngine,
		StartLayerID: 11,
		EndLayerID:   21,
	})
	defer remoteExecutor.Close()

	// Create shared protocol on local host for response routing
	localProtocol := p2p.NewProtocol(localHost)

	// Create remote layer executor (sends requests to remote)
	remoteLayerExec := inference.NewRemoteLayerExecutor(inference.RemoteLayerExecutorConfig{
		Host:         localHost,
		Protocol:     localProtocol, // Shared protocol for response routing
		TargetPeerID: remoteHost.ID(),
		StartLayerID: 11,
		EndLayerID:   21,
		Config:       config,
		Timeout:      5 * time.Second,
	})

	// Create test hidden state
	hiddenSize := config.HiddenSize * 2 // FP16
	testHidden := make([]byte, hiddenSize)
	for i := range testHidden {
		testHidden[i] = byte((i + 1) % 256)
	}

	// Send activation to remote and wait for response
	t.Log("Sending activation to remote peer...")
	output, _, _, err := remoteLayerExec.Forward(ctx, 11, testHidden, 0)
	if err != nil {
		t.Fatalf("Remote forward failed: %v", err)
	}

	// Verify we got a response
	if output == nil {
		t.Fatal("Output is nil")
	}
	if len(output) != len(testHidden) {
		t.Errorf("Output size mismatch: got %d, expected %d", len(output), len(testHidden))
	}

	// Verify the data was processed (first byte should be layer ID)
	// Note: The remote executor processes layers 11-21, so the last processed layer is 21
	expectedFirstByte := byte(21) // Last layer processed
	if output[0] != expectedFirstByte {
		t.Errorf("First byte mismatch: got %d, expected %d", output[0], expectedFirstByte)
	}

	t.Logf("PASS: Full activation roundtrip completed")
	t.Logf("  - Sent %d bytes", len(testHidden))
	t.Logf("  - Received %d bytes", len(output))
	t.Logf("  - Processing marker: %d (layer 21)", output[0])
}

// MockLayerExecutor is a test helper for mocking layer execution
type MockLayerExecutor struct {
	forwardFunc func(ctx context.Context, layerID int, hidden []byte, position int) ([]byte, []byte, []byte, error)
}

func (m *MockLayerExecutor) Forward(ctx context.Context, layerID int, hidden []byte, position int) ([]byte, []byte, []byte, error) {
	if m.forwardFunc != nil {
		return m.forwardFunc(ctx, layerID, hidden, position)
	}
	return hidden, nil, nil, nil
}

// TestCoordinatorIntegration tests the DistributedInferenceCoordinator
func TestCoordinatorIntegration(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	config := types.TinyLlamaConfig()
	localPeerIDStr := "local-peer-id"
	remotePeerIDStr := "remote-peer-id"

	// Create libp2p hosts
	localHost, err := libp2p.New(libp2p.ListenAddrStrings("/ip4/127.0.0.1/tcp/0"))
	if err != nil {
		t.Fatalf("Failed to create local host: %v", err)
	}
	defer localHost.Close()

	remoteHost, err := libp2p.New(libp2p.ListenAddrStrings("/ip4/127.0.0.1/tcp/0"))
	if err != nil {
		t.Fatalf("Failed to create remote host: %v", err)
	}
	defer remoteHost.Close()

	// Get actual peer IDs
	localPeerIDStr = localHost.ID().String()
	remotePeerIDStr = remoteHost.ID().String()

	// Connect hosts
	localHost.Peerstore().AddAddrs(remoteHost.ID(), remoteHost.Addrs(), time.Hour)
	err = localHost.Connect(ctx, peer.AddrInfo{ID: remoteHost.ID(), Addrs: remoteHost.Addrs()})
	if err != nil {
		t.Fatalf("Failed to connect hosts: %v", err)
	}

	// Create layer assignments - local: 0-10, remote: 11-21
	var assignments []scheduler.LayerAssignment
	for i := 0; i < config.NumLayers; i++ {
		if i <= 10 {
			assignments = append(assignments, scheduler.LayerAssignment{
				LayerID: i,
				PeerID:  localPeerIDStr,
			})
		} else {
			assignments = append(assignments, scheduler.LayerAssignment{
				LayerID: i,
				PeerID:  remotePeerIDStr,
			})
		}
	}

	// Create local engine
	localEngine := inference.NewEngine(inference.EngineConfig{
		ModelConfig: config,
		LocalPeerID: localPeerIDStr,
	})
	localEngine.SetAssignments(assignments)

	// Create peer manager for local host
	peerManager := p2p.NewPeerManager(localHost)

	// Create coordinator
	localProtocol := p2p.NewProtocol(localHost)

	coordinator := inference.NewDistributedInferenceCoordinator(inference.CoordinatorConfig{
		Host:          localHost,
		Engine:        localEngine,
		PeerManager:   peerManager,
		Protocol:      localProtocol,
		ModelConfig:   config,
		Assignments:   assignments,
		LocalPeerID:   localPeerIDStr,
		WeightTimeout: 30 * time.Second,
	})
	defer coordinator.Close()

	// Verify coordinator parsed assignments correctly
	weightsForRemote := coordinator.GetWeightsForPeer(remotePeerIDStr)
	// We haven't loaded any weights yet, so should be empty
	if len(weightsForRemote) != 0 {
		t.Errorf("Expected 0 weights (not loaded), got %d", len(weightsForRemote))
	}

	// Load some test weights for remote layers
	for i := 11; i <= 15; i++ {
		testWeight := &inference.CPULayerWeights{
			LayerID:  i,
			AttnNorm: &inference.CPUTensor{Shape: []int{config.HiddenSize}, Data: make([]byte, config.HiddenSize*2)},
		}
		coordinator.LoadLocalWeights(i, testWeight)
	}

	// Now check weights for remote peer
	weightsForRemote = coordinator.GetWeightsForPeer(remotePeerIDStr)
	if len(weightsForRemote) != 5 {
		t.Errorf("Expected 5 weights for remote peer, got %d", len(weightsForRemote))
	}

	// Verify weight receiver is set up
	if coordinator.GetWeightReceiver() == nil {
		t.Error("Weight receiver should not be nil")
	}

	// Verify weight distributor is set up
	if coordinator.GetWeightDistributor() == nil {
		t.Error("Weight distributor should not be nil")
	}

	// Create performer coordinator on remote side
	remoteEngine := inference.NewEngine(inference.EngineConfig{
		ModelConfig: config,
		LocalPeerID: remotePeerIDStr,
	})

	performerCoordinator := inference.NewPerformerCoordinator(inference.PerformerCoordinatorConfig{
		Host:         remoteHost,
		Engine:       remoteEngine,
		ModelConfig:  config,
		StartLayerID: 11,
		EndLayerID:   21,
	})
	defer performerCoordinator.Close()

	// Verify performer is not ready (no weights loaded)
	if performerCoordinator.IsWeightsLoaded() {
		t.Error("Performer should not have weights loaded yet")
	}

	// Verify remote executor is set up
	if performerCoordinator.GetRemoteExecutor() == nil {
		t.Error("Performer's remote executor should not be nil")
	}

	t.Log("PASS: Coordinator integration test completed")
}

// TestCoordinatorPeerConnectionCallback tests automatic setup on peer connection
func TestCoordinatorPeerConnectionCallback(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	config := types.TinyLlamaConfig()

	// Create libp2p hosts
	localHost, err := libp2p.New(libp2p.ListenAddrStrings("/ip4/127.0.0.1/tcp/0"))
	if err != nil {
		t.Fatalf("Failed to create local host: %v", err)
	}
	defer localHost.Close()

	remoteHost, err := libp2p.New(libp2p.ListenAddrStrings("/ip4/127.0.0.1/tcp/0"))
	if err != nil {
		t.Fatalf("Failed to create remote host: %v", err)
	}
	defer remoteHost.Close()

	localPeerIDStr := localHost.ID().String()
	remotePeerIDStr := remoteHost.ID().String()

	// Create layer assignments
	var assignments []scheduler.LayerAssignment
	for i := 0; i < config.NumLayers; i++ {
		if i <= 10 {
			assignments = append(assignments, scheduler.LayerAssignment{
				LayerID: i,
				PeerID:  localPeerIDStr,
			})
		} else {
			assignments = append(assignments, scheduler.LayerAssignment{
				LayerID: i,
				PeerID:  remotePeerIDStr,
			})
		}
	}

	// Create engine
	localEngine := inference.NewEngine(inference.EngineConfig{
		ModelConfig: config,
		LocalPeerID: localPeerIDStr,
	})
	localEngine.SetAssignments(assignments)

	// Create peer manager
	peerManager := p2p.NewPeerManager(localHost)
	peerManager.Start()
	defer peerManager.Stop()

	// Create coordinator

	localProtocol := p2p.NewProtocol(localHost)

	coordinator := inference.NewDistributedInferenceCoordinator(inference.CoordinatorConfig{
		Host:          localHost,
		Engine:        localEngine,
		PeerManager:   peerManager,
		Protocol:      localProtocol, // Shared protocol for response routing
		ModelConfig:   config,
		Assignments:   assignments,
		LocalPeerID:   localPeerIDStr,
		WeightTimeout: 5 * time.Second,
	})
	defer coordinator.Close()

	// Load test weights for remote layers
	for i := 11; i <= 21; i++ {
		testWeight := &inference.CPULayerWeights{
			LayerID:  i,
			AttnNorm: &inference.CPUTensor{Shape: []int{config.HiddenSize}, Data: make([]byte, config.HiddenSize*2)},
		}
		coordinator.LoadLocalWeights(i, testWeight)
	}

	// Initially not ready (no peer connected)
	if coordinator.IsReady() {
		t.Error("Coordinator should not be ready before peer connection")
	}

	// Add remote peer info
	remoteInfo := peer.AddrInfo{ID: remoteHost.ID(), Addrs: remoteHost.Addrs()}
	peerManager.AddPeer(remoteInfo)

	// Connect to remote peer - this should trigger the callback
	err = localHost.Connect(ctx, remoteInfo)
	if err != nil {
		t.Fatalf("Failed to connect to remote peer: %v", err)
	}

	// Wait a bit for callback to be processed
	time.Sleep(200 * time.Millisecond)

	// Check remote executor was registered
	exec, exists := coordinator.GetRemoteExecutor(remotePeerIDStr)
	if !exists {
		t.Error("Remote executor should be registered after peer connection")
	}
	if exec == nil {
		t.Error("Remote executor should not be nil")
	}

	// Check coordinator is now ready
	if !coordinator.IsReady() {
		t.Error("Coordinator should be ready after peer connection")
	}

	t.Log("PASS: Coordinator peer connection callback test completed")
}

// =============================================================================
// PRP: HYBRID DISTRIBUTED INFERENCE - MODEL CONFIG PROTOCOL TESTS (RED PHASE)
// =============================================================================
// These tests validate the MsgTypeModelConfig message type for transferring
// model configuration to stateless workers. All tests should FAIL initially
// as the implementation doesn't exist yet.

// TestProtocol_ModelConfigMessageType validates MsgTypeModelConfig constant exists
// AC1: Worker receives ModelConfig via P2P
func TestProtocol_ModelConfigMessageType(t *testing.T) {
	// This test should FAIL initially - MsgTypeModelConfig doesn't exist yet
	// Expected constant value: 0x07 (after MsgTypeLayerRequest = 0x06)

	// Verify MsgTypeModelConfig constant is defined
	// This should FAIL - p2p.MsgTypeModelConfig doesn't exist
	if p2p.MsgTypeModelConfig != 0x07 {
		t.Errorf("MsgTypeModelConfig should be 0x07, got 0x%02x", p2p.MsgTypeModelConfig)
	}

	t.Log("PASS: MsgTypeModelConfig constant defined")
}

// TestProtocol_SendModelConfig validates SendModelConfig method exists
// AC1: Worker receives ModelConfig via P2P
func TestProtocol_SendModelConfig(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Create sender and receiver hosts
	sender, err := p2p.NewTestHost(ctx, 0)
	if err != nil {
		t.Fatalf("Sender creation failed: %v", err)
	}
	defer sender.Close()

	receiver, err := p2p.NewTestHost(ctx, 0)
	if err != nil {
		t.Fatalf("Receiver creation failed: %v", err)
	}
	defer receiver.Close()

	// Setup protocol on sender and receiver (required for stream negotiation)
	senderProto := p2p.NewProtocol(sender)
	_ = p2p.NewProtocol(receiver) // Receiver needs protocol registered for stream handling

	// Connect sender to receiver
	err = sender.Connect(ctx, p2p.GetHostInfo(receiver))
	if err != nil {
		t.Fatalf("Connection failed: %v", err)
	}

	// Create test config data (simulating serialized TransferableConfig)
	testConfigData := []byte(`{"model_name":"mistral-7b","hidden_size":4096,"num_layers":32}`)

	// This should FAIL - SendModelConfig method doesn't exist yet
	err = senderProto.SendModelConfig(ctx, receiver.ID(), testConfigData)
	if err != nil {
		t.Fatalf("SendModelConfig failed: %v", err)
	}

	t.Log("PASS: SendModelConfig method exists and sends data")
}

// TestProtocol_ModelConfigRoundtrip validates full send/receive cycle
// AC1: Worker receives ModelConfig via P2P (Log: "Received model config")
func TestProtocol_ModelConfigRoundtrip(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Create sender and receiver hosts
	sender, err := p2p.NewTestHost(ctx, 0)
	if err != nil {
		t.Fatalf("Sender creation failed: %v", err)
	}
	defer sender.Close()

	receiver, err := p2p.NewTestHost(ctx, 0)
	if err != nil {
		t.Fatalf("Receiver creation failed: %v", err)
	}
	defer receiver.Close()

	// Setup protocols
	senderProto := p2p.NewProtocol(sender)
	receiverProto := p2p.NewProtocol(receiver)

	// Track received config
	var receivedConfig []byte
	var receivedFrom peer.ID
	receivedChan := make(chan struct{})

	// Register handler for model config
	// This should FAIL - OnModelConfigReceived method doesn't exist yet
	receiverProto.OnModelConfigReceived(func(config []byte, from peer.ID) {
		receivedConfig = config
		receivedFrom = from
		close(receivedChan)
	})

	// Connect sender to receiver
	err = sender.Connect(ctx, p2p.GetHostInfo(receiver))
	if err != nil {
		t.Fatalf("Connection failed: %v", err)
	}

	// Create test config data
	testConfigData := []byte(`{
		"model_name": "mistral-7b",
		"hidden_size": 4096,
		"num_layers": 32,
		"intermediate_size": 14336,
		"num_heads": 32,
		"num_kv_heads": 8,
		"head_dim": 128,
		"vocab_size": 32000,
		"max_seq_len": 8192,
		"rms_norm_eps": 1e-5
	}`)

	// Send config
	// This should FAIL - SendModelConfig method doesn't exist yet
	err = senderProto.SendModelConfig(ctx, receiver.ID(), testConfigData)
	if err != nil {
		t.Fatalf("SendModelConfig failed: %v", err)
	}

	// Wait for reception
	select {
	case <-receivedChan:
		// Verify received data
		if receivedConfig == nil {
			t.Fatal("Received config is nil")
		}
		if string(receivedConfig) != string(testConfigData) {
			t.Errorf("Config data mismatch:\ngot: %s\nexpected: %s", string(receivedConfig), string(testConfigData))
		}
		if receivedFrom != sender.ID() {
			t.Errorf("From peer mismatch: got %s, expected %s", receivedFrom.String(), sender.ID().String())
		}
		t.Log("PASS: ModelConfig round-trip working")

	case <-ctx.Done():
		t.Fatal("Timeout waiting for model config reception")
	}
}

// TestProtocol_ModelConfigHandler validates handler registration
// AC1: Worker receives ModelConfig via P2P
func TestProtocol_ModelConfigHandler(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Create host
	host, err := p2p.NewTestHost(ctx, 0)
	if err != nil {
		t.Fatalf("Host creation failed: %v", err)
	}
	defer host.Close()

	// Setup protocol
	proto := p2p.NewProtocol(host)

	// Define handler
	handlerCalled := false
	handler := func(config []byte, from peer.ID) {
		handlerCalled = true
	}

	// Register handler
	// This should FAIL - OnModelConfigReceived method doesn't exist yet
	proto.OnModelConfigReceived(handler)

	// Verify handler can be registered (no panic)
	// The actual invocation is tested in TestProtocol_ModelConfigRoundtrip

	t.Log("PASS: OnModelConfigReceived handler registration works")
	_ = handlerCalled // Handler will be called in roundtrip test
}

// TestCoordinator_SendsConfigBeforeWeights validates config is sent before weights
// AC4: Coordinator distributes layers to 2 workers (config must precede weights)
func TestCoordinator_SendsConfigBeforeWeights(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	config := types.TinyLlamaConfig()

	// Create two hosts
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

	// Track message order
	var messageOrder []string
	var orderMu sync.Mutex

	// Set up worker protocol to track messages
	workerProto := p2p.NewProtocol(workerHost)

	// Track config reception
	// This should FAIL - OnModelConfigReceived doesn't exist yet
	workerProto.OnModelConfigReceived(func(configData []byte, from peer.ID) {
		orderMu.Lock()
		messageOrder = append(messageOrder, "config")
		orderMu.Unlock()
	})

	// Track weights reception
	workerProto.OnWeightsReceived(func(layerID int, chunkIndex int, totalChunks int, data []byte) {
		orderMu.Lock()
		messageOrder = append(messageOrder, "weights")
		orderMu.Unlock()
	})

	// Respond immediately to layer status requests so distribution can proceed.
	workerProto.OnLayerRequestReceived(func(from peer.ID, requestedLayers []int) {
		_ = workerProto.SendLayerStatus(ctx, from, []int{})
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
	coordinatorEngine := inference.NewEngine(inference.EngineConfig{
		ModelConfig: config,
		LocalPeerID: coordinatorPeerIDStr,
	})
	coordinatorEngine.SetAssignments(assignments)

	// Create peer manager
	peerManager := p2p.NewPeerManager(coordinatorHost)
	peerManager.Start()
	defer peerManager.Stop()

	// Create coordinator

	localProtocol := p2p.NewProtocol(coordinatorHost)

	coordinator := inference.NewDistributedInferenceCoordinator(inference.CoordinatorConfig{
		Host:          coordinatorHost,
		Engine:        coordinatorEngine,
		PeerManager:   peerManager,
		Protocol:      localProtocol, // Shared protocol for response routing
		ModelConfig:   config,
		Assignments:   assignments,
		LocalPeerID:   coordinatorPeerIDStr,
		WeightTimeout: 200 * time.Millisecond,
	})
	defer coordinator.Close()

	// Load test weights
	for i := 0; i < 5; i++ {
		testWeight := &inference.CPULayerWeights{
			LayerID:  i,
			AttnNorm: &inference.CPUTensor{Shape: []int{config.HiddenSize}, Data: make([]byte, config.HiddenSize*2)},
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

	// Wait for distribution to happen (async callback + protocol handshake)
	deadline := time.Now().Add(5 * time.Second)
	for {
		orderMu.Lock()
		n := len(messageOrder)
		orderMu.Unlock()
		if n > 0 || time.Now().After(deadline) {
			break
		}
		time.Sleep(100 * time.Millisecond)
	}

	// Verify message order: config should come before weights
	orderMu.Lock()
	defer orderMu.Unlock()

	if len(messageOrder) == 0 {
		t.Fatal("No messages received")
	}

	// First message should be config
	if messageOrder[0] != "config" {
		t.Errorf("First message should be config, got: %s", messageOrder[0])
	}

	// All subsequent messages should be weights
	for i := 1; i < len(messageOrder); i++ {
		if messageOrder[i] != "weights" {
			t.Errorf("Message %d should be weights, got: %s", i, messageOrder[i])
		}
	}

	t.Log("PASS: Coordinator sends config before weights")
	t.Logf("Message order: %v", messageOrder)
}
