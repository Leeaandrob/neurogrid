// Package inference provides remote layer execution handling for distributed inference.
package inference

import (
	"context"
	"fmt"
	"log"
	"sync"
	"sync/atomic"
	"time"

	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/neurogrid/engine/p2p"
	"github.com/neurogrid/engine/pkg/types"
)

// RemoteExecutor handles incoming activation requests from remote peers.
// It executes transformer layers locally and sends responses back.
type RemoteExecutor struct {
	host          host.Host
	protocol      *p2p.Protocol
	engine        *Engine
	startLayerID  int // First layer this executor handles
	endLayerID    int // Last layer this executor handles (inclusive)
	requestCount  uint64
	mu            sync.RWMutex
	closed        bool
}

// RemoteExecutorConfig holds configuration for creating a RemoteExecutor.
type RemoteExecutorConfig struct {
	Host         host.Host
	Engine       *Engine
	StartLayerID int
	EndLayerID   int
}

// NewRemoteExecutor creates a new remote layer executor.
func NewRemoteExecutor(config RemoteExecutorConfig) *RemoteExecutor {
	re := &RemoteExecutor{
		host:         config.Host,
		engine:       config.Engine,
		startLayerID: config.StartLayerID,
		endLayerID:   config.EndLayerID,
	}

	// Create protocol and register handlers
	re.protocol = p2p.NewProtocol(config.Host)
	re.protocol.OnActivationReceived(re.handleActivation)

	return re
}

// handleActivation processes incoming activation requests.
func (re *RemoteExecutor) handleActivation(msg *p2p.TensorMessage) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	atomic.AddUint64(&re.requestCount, 1)

	// Validate layer is in our range
	if msg.LayerID < re.startLayerID || msg.LayerID > re.endLayerID {
		log.Printf("[RemoteExecutor] Layer %d not in range [%d, %d]",
			msg.LayerID, re.startLayerID, re.endLayerID)
		return
	}

	// Execute layers from msg.LayerID through endLayerID
	hidden := msg.Data
	var err error

	for layerID := msg.LayerID; layerID <= re.endLayerID; layerID++ {
		// Use engine's layer executor for local processing
		re.engine.mu.RLock()
		executor := re.engine.layerExecutor
		re.engine.mu.RUnlock()

		if executor == nil {
			log.Printf("[RemoteExecutor] No layer executor available")
			return
		}

		// Position comes from SeqID (we use it as position for simplicity)
		position := int(msg.SeqID)

		output, _, _, err := executor.Forward(ctx, layerID, hidden, position)
		if err != nil {
			log.Printf("[RemoteExecutor] Layer %d forward failed: %v", layerID, err)
			return
		}

		hidden = output
	}

	// Send response back to sender
	err = re.protocol.SendResponse(ctx, msg.From, re.endLayerID, msg.SeqID, msg.RequestID, hidden)
	if err != nil {
		log.Printf("[RemoteExecutor] Failed to send response: %v", err)
	}
}

// GetProtocol returns the underlying protocol for direct access.
func (re *RemoteExecutor) GetProtocol() *p2p.Protocol {
	return re.protocol
}

// RequestCount returns the number of requests processed.
func (re *RemoteExecutor) RequestCount() uint64 {
	return atomic.LoadUint64(&re.requestCount)
}

// Close stops the remote executor.
func (re *RemoteExecutor) Close() error {
	re.mu.Lock()
	defer re.mu.Unlock()

	if re.closed {
		return nil
	}

	re.closed = true
	return nil
}

// RemoteLayerExecutor implements LayerExecutor for remote layer execution.
// It sends activations to remote peers and waits for responses.
type RemoteLayerExecutor struct {
	host           host.Host
	protocol       *p2p.Protocol
	targetPeerID   peer.ID
	startLayerID   int
	endLayerID     int
	config         *types.LlamaConfig
	requestCounter uint64
	defaultTimeout time.Duration
}

// RemoteLayerExecutorConfig holds configuration for creating a RemoteLayerExecutor.
type RemoteLayerExecutorConfig struct {
	Host         host.Host
	TargetPeerID peer.ID
	StartLayerID int
	EndLayerID   int
	Config       *types.LlamaConfig
	Timeout      time.Duration
}

// NewRemoteLayerExecutor creates a new remote layer executor for forwarding to peers.
func NewRemoteLayerExecutor(config RemoteLayerExecutorConfig) *RemoteLayerExecutor {
	timeout := config.Timeout
	if timeout == 0 {
		timeout = 30 * time.Second
	}

	rle := &RemoteLayerExecutor{
		host:           config.Host,
		targetPeerID:   config.TargetPeerID,
		startLayerID:   config.StartLayerID,
		endLayerID:     config.EndLayerID,
		config:         config.Config,
		defaultTimeout: timeout,
	}

	// Create protocol
	rle.protocol = p2p.NewProtocol(config.Host)

	return rle
}

// Forward sends activation to remote peer and waits for response.
func (rle *RemoteLayerExecutor) Forward(ctx context.Context, layerID int, hidden []byte, position int) ([]byte, []byte, []byte, error) {
	// Validate layer is in our target range
	if layerID < rle.startLayerID {
		return nil, nil, nil, fmt.Errorf("layer %d below start layer %d", layerID, rle.startLayerID)
	}

	// Generate unique request ID
	requestID := atomic.AddUint64(&rle.requestCounter, 1)

	// Send activation to remote peer
	seqID := uint64(position) // Use position as seqID
	err := rle.protocol.SendActivation(ctx, rle.targetPeerID, layerID, seqID, requestID, hidden)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("send activation failed: %w", err)
	}

	// Wait for response
	response, err := rle.protocol.WaitForResponse(ctx, requestID, rle.defaultTimeout)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("wait response failed: %w", err)
	}

	// Return output hidden state (k, v are managed by remote KV cache)
	kvSize := 0
	if rle.config != nil {
		kvSize = rle.config.NumKVHeads * rle.config.HeadDim * 2 // FP16
	}
	k := make([]byte, kvSize)
	v := make([]byte, kvSize)

	return response.Data, k, v, nil
}

// Close releases resources.
func (rle *RemoteLayerExecutor) Close() error {
	return nil
}
