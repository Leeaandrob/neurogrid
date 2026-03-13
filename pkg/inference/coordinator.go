// Package inference provides distributed inference coordination.
package inference

import (
	"context"
	"log"
	"sync"
	"time"

	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/neurogrid/engine/p2p"
	"github.com/neurogrid/engine/pkg/scheduler"
	"github.com/neurogrid/engine/pkg/types"
)

// peerIDLogLength is the number of characters to show when logging peer IDs.
// Peer IDs are truncated for readability in log output.
const peerIDLogLength = 12

// shortPeerID returns a truncated peer ID string for logging.
// Example: "12D3KooWSttt" from a full peer ID.
func shortPeerID(peerID string) string {
	if len(peerID) < peerIDLogLength {
		return peerID
	}
	return peerID[:peerIDLogLength]
}

// DistributedInferenceCoordinator manages distributed inference across peers.
// It handles weight distribution, remote executor setup, and peer lifecycle.
type DistributedInferenceCoordinator struct {
	host        host.Host
	engine      *Engine
	peerManager *p2p.PeerManager
	config      *types.LlamaConfig
	protocol    *p2p.Protocol // Shared protocol for all remote executors

	// Weight distribution
	weightDistributor *WeightDistributor
	weightReceiver    *WeightReceiver

	// Layer assignments per peer
	peerAssignments map[string][]int // peerID -> list of layerIDs
	localLayers     []int            // layers assigned to local peer

	// Local layer weights ready to distribute (set by LoadLocalWeights)
	localWeights map[int]*CPULayerWeights // layerID -> weights

	// Peer layer status (which layers each peer has locally)
	peerLoadedLayers map[string][]int // peerID -> list of layerIDs already loaded

	// Config distribution tracking for stateless workers
	configSent map[string]bool // Track which peers received config
	modelName  string          // Model name for config serialization

	// Remote executors
	remoteExecutors map[string]*RemoteLayerExecutor // peerID -> executor

	// Timeout for weight distribution
	weightTimeout time.Duration

	mu     sync.RWMutex
	closed bool
}

// CoordinatorConfig holds configuration for creating a DistributedInferenceCoordinator.
type CoordinatorConfig struct {
	Host          host.Host
	Engine        *Engine
	PeerManager   *p2p.PeerManager
	Protocol      *p2p.Protocol // Shared protocol for tensor transfer (required)
	ModelConfig   *types.LlamaConfig
	Assignments   []scheduler.LayerAssignment
	LocalPeerID   string
	WeightTimeout time.Duration
	ModelName     string // Model name for config transfer to stateless workers
}

// NewDistributedInferenceCoordinator creates a new coordinator.
func NewDistributedInferenceCoordinator(config CoordinatorConfig) *DistributedInferenceCoordinator {
	timeout := config.WeightTimeout
	if timeout == 0 {
		timeout = 60 * time.Second
	}

	if config.Protocol == nil {
		panic("CoordinatorConfig.Protocol is required - must provide shared protocol")
	}

	// Use model name from config, or default if not provided
	modelName := config.ModelName
	if modelName == "" {
		modelName = "unknown"
	}

	dic := &DistributedInferenceCoordinator{
		host:             config.Host,
		engine:           config.Engine,
		peerManager:      config.PeerManager,
		config:           config.ModelConfig,
		protocol:         config.Protocol,
		peerAssignments:  make(map[string][]int),
		localLayers:      make([]int, 0),
		localWeights:     make(map[int]*CPULayerWeights),
		peerLoadedLayers: make(map[string][]int),
		configSent:       make(map[string]bool),
		modelName:        modelName,
		remoteExecutors:  make(map[string]*RemoteLayerExecutor),
		weightTimeout:    timeout,
	}

	// Create weight distributor
	dic.weightDistributor = NewWeightDistributor(WeightDistributorConfig{
		Host:        config.Host,
		ModelConfig: config.ModelConfig,
	})

	// Create weight receiver with callback
	dic.weightReceiver = NewWeightReceiver(WeightReceiverConfig{
		Host:            config.Host,
		OnLayerReceived: dic.onLayerWeightsReceived,
	})

	// Register layer status handler to receive worker's loaded layers info
	config.Protocol.OnLayerStatusReceived(dic.onLayerStatusReceived)

	// Parse layer assignments
	for _, a := range config.Assignments {
		// Skip embedding and output layers
		if a.LayerID < 0 || a.LayerID >= config.ModelConfig.NumLayers {
			continue
		}

		if a.PeerID == config.LocalPeerID {
			dic.localLayers = append(dic.localLayers, a.LayerID)
		} else {
			dic.peerAssignments[a.PeerID] = append(dic.peerAssignments[a.PeerID], a.LayerID)
		}
	}

	// Register peer connection callbacks
	if config.PeerManager != nil {
		config.PeerManager.SetCallbacks(dic.onPeerConnected, dic.onPeerDisconnected)
	}

	log.Printf("[Coordinator] Initialized with %d local layers, %d remote peers",
		len(dic.localLayers), len(dic.peerAssignments))

	return dic
}

// LoadLocalWeights loads layer weights that will be distributed to remote peers.
// This should be called after loading the model but before peers connect.
func (dic *DistributedInferenceCoordinator) LoadLocalWeights(layerID int, weights *CPULayerWeights) {
	dic.mu.Lock()
	defer dic.mu.Unlock()
	dic.localWeights[layerID] = weights
	log.Printf("[Coordinator] Loaded weights for layer %d", layerID)
}

// GetWeightsForPeer returns the weights that should be distributed to a peer.
func (dic *DistributedInferenceCoordinator) GetWeightsForPeer(peerID string) []*CPULayerWeights {
	dic.mu.RLock()
	defer dic.mu.RUnlock()

	layerIDs, ok := dic.peerAssignments[peerID]
	if !ok {
		return nil
	}

	var weights []*CPULayerWeights
	for _, layerID := range layerIDs {
		if w, exists := dic.localWeights[layerID]; exists {
			weights = append(weights, w)
		}
	}
	return weights
}

// onPeerConnected is called when a peer connects.
func (dic *DistributedInferenceCoordinator) onPeerConnected(peerID peer.ID) {
	dic.mu.Lock()
	if dic.closed {
		dic.mu.Unlock()
		return
	}
	dic.mu.Unlock()

	peerIDStr := peerID.String()
	log.Printf("[Coordinator] Peer connected: %s", shortPeerID(peerIDStr))

	// Check if this peer has layer assignments
	dic.mu.RLock()
	layerIDs, hasAssignments := dic.peerAssignments[peerIDStr]
	dic.mu.RUnlock()

	if !hasAssignments {
		log.Printf("[Coordinator] No layer assignments for peer %s", shortPeerID(peerIDStr))
		return
	}

	// Create remote executor for this peer
	dic.setupRemoteExecutor(peerID, layerIDs)

	// Request layer status from peer before distributing weights
	// The actual weight distribution will happen in onLayerStatusReceived
	go dic.requestAndDistributeWeights(peerID)
}

// onLayerStatusReceived is called when a peer reports its loaded layers.
func (dic *DistributedInferenceCoordinator) onLayerStatusReceived(peerID peer.ID, loadedLayers []int) {
	peerIDStr := peerID.String()
	log.Printf("[Coordinator] Received layer status from %s: %d layers loaded locally",
		shortPeerID(peerIDStr), len(loadedLayers))

	dic.mu.Lock()
	dic.peerLoadedLayers[peerIDStr] = loadedLayers
	dic.mu.Unlock()
}

// requestAndDistributeWeights requests layer status and distributes only missing weights.
func (dic *DistributedInferenceCoordinator) requestAndDistributeWeights(peerID peer.ID) {
	peerIDStr := peerID.String()

	// Wait a moment for the worker to send its layer status after connection
	time.Sleep(500 * time.Millisecond)

	// Check if we already received layer status
	dic.mu.RLock()
	loadedLayers, hasStatus := dic.peerLoadedLayers[peerIDStr]
	dic.mu.RUnlock()

	if !hasStatus {
		// Request layer status explicitly
		log.Printf("[Coordinator] Requesting layer status from peer %s", shortPeerID(peerIDStr))
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		var err error
		loadedLayers, err = dic.protocol.RequestLayerStatus(ctx, peerID, dic.weightTimeout)
		cancel()

		if err != nil {
			log.Printf("[Coordinator] Failed to get layer status from %s: %v, will send all weights",
				shortPeerID(peerIDStr), err)
			loadedLayers = nil
		} else {
			dic.mu.Lock()
			dic.peerLoadedLayers[peerIDStr] = loadedLayers
			dic.mu.Unlock()
		}
	}

	// Distribute only missing weights
	dic.distributeWeightsToPeer(peerID)
}

// onPeerDisconnected is called when a peer disconnects.
func (dic *DistributedInferenceCoordinator) onPeerDisconnected(peerID peer.ID) {
	peerIDStr := peerID.String()
	log.Printf("[Coordinator] Peer disconnected: %s", shortPeerID(peerIDStr))

	dic.mu.Lock()
	defer dic.mu.Unlock()

	// Remove remote executor
	if exec, exists := dic.remoteExecutors[peerIDStr]; exists {
		exec.Close()
		delete(dic.remoteExecutors, peerIDStr)
	}

	// Unregister from engine
	if dic.engine != nil {
		dic.engine.UnregisterRemoteExecutor(peerIDStr)
	}
}

// setupRemoteExecutor creates and registers a remote executor for a peer.
func (dic *DistributedInferenceCoordinator) setupRemoteExecutor(peerID peer.ID, layerIDs []int) {
	if len(layerIDs) == 0 {
		return
	}

	// Find layer range
	startLayer := layerIDs[0]
	endLayer := layerIDs[0]
	for _, id := range layerIDs {
		if id < startLayer {
			startLayer = id
		}
		if id > endLayer {
			endLayer = id
		}
	}

	peerIDStr := peerID.String()

	// Create remote layer executor with shared protocol
	exec := NewRemoteLayerExecutor(RemoteLayerExecutorConfig{
		Host:         dic.host,
		Protocol:     dic.protocol, // Use shared protocol for response routing
		TargetPeerID: peerID,
		StartLayerID: startLayer,
		EndLayerID:   endLayer,
		Config:       dic.config,
		Timeout:      30 * time.Second,
	})

	dic.mu.Lock()
	dic.remoteExecutors[peerIDStr] = exec
	dic.mu.Unlock()

	// Register with engine
	if dic.engine != nil {
		dic.engine.RegisterRemoteExecutor(peerIDStr, exec)
	}

	log.Printf("[Coordinator] Registered remote executor for peer %s (layers %d-%d)",
		shortPeerID(peerIDStr), startLayer, endLayer)
}

// sendConfigToPeer sends model config to a peer if not already sent.
// Returns true if config was sent (or already sent), false on error.
// Config must be sent before weights so stateless workers can process incoming weight data.
func (dic *DistributedInferenceCoordinator) sendConfigToPeer(peerID peer.ID) bool {
	peerIDStr := peerID.String()

	dic.mu.RLock()
	alreadySent := dic.configSent[peerIDStr]
	dic.mu.RUnlock()

	if alreadySent || dic.config == nil {
		return true
	}

	configData, err := SerializeConfig(dic.config, dic.modelName)
	if err != nil {
		log.Printf("[Coordinator] Failed to serialize config for %s: %v", shortPeerID(peerIDStr), err)
		return false
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := dic.protocol.SendModelConfig(ctx, peerID, configData); err != nil {
		log.Printf("[Coordinator] Failed to send config to %s: %v", shortPeerID(peerIDStr), err)
		return false
	}

	dic.mu.Lock()
	dic.configSent[peerIDStr] = true
	dic.mu.Unlock()

	log.Printf("[Coordinator] Sent model config to peer %s", shortPeerID(peerIDStr))

	// Brief pause to ensure config is processed before weights
	time.Sleep(100 * time.Millisecond)
	return true
}

// distributeWeightsToPeer sends layer weights to a connected peer.
// Only sends weights for layers the peer doesn't already have locally.
// Config is sent before weights via sendConfigToPeer.
func (dic *DistributedInferenceCoordinator) distributeWeightsToPeer(peerID peer.ID) {
	peerIDStr := peerID.String()

	// Get all weights assigned to this peer
	allWeights := dic.GetWeightsForPeer(peerIDStr)
	if len(allWeights) == 0 {
		log.Printf("[Coordinator] No weights to distribute to peer %s", shortPeerID(peerIDStr))
		return
	}

	// Send config before weights (critical for stateless workers)
	dic.sendConfigToPeer(peerID)

	// Check which layers peer already has locally
	dic.mu.RLock()
	loadedLayers := dic.peerLoadedLayers[peerIDStr]
	dic.mu.RUnlock()

	// Build set of loaded layer IDs for quick lookup
	loadedSet := make(map[int]bool)
	for _, layerID := range loadedLayers {
		loadedSet[layerID] = true
	}

	// Filter to only weights the peer doesn't have
	var missingWeights []*CPULayerWeights
	var missingLayerIDs []int
	for _, w := range allWeights {
		if !loadedSet[w.LayerID] {
			missingWeights = append(missingWeights, w)
			missingLayerIDs = append(missingLayerIDs, w.LayerID)
		}
	}

	// Log the optimization
	if len(loadedLayers) > 0 {
		log.Printf("[Coordinator] Peer %s has %d/%d layers locally cached, sending %d missing layers",
			shortPeerID(peerIDStr), len(loadedLayers), len(allWeights), len(missingWeights))
	}

	if len(missingWeights) == 0 {
		log.Printf("[Coordinator] Peer %s has all %d layers locally, no transfer needed!",
			shortPeerID(peerIDStr), len(allWeights))
		return
	}

	log.Printf("[Coordinator] Distributing %d layer weights to peer %s (layers: %v)",
		len(missingWeights), shortPeerID(peerIDStr), missingLayerIDs)

	ctx, cancel := context.WithTimeout(context.Background(), dic.weightTimeout)
	defer cancel()

	err := dic.weightDistributor.DistributeLayersToPerformer(ctx, peerID, missingWeights, 30*time.Second)
	if err != nil {
		log.Printf("[Coordinator] Weight distribution to %s failed: %v", shortPeerID(peerIDStr), err)
		return
	}

	log.Printf("[Coordinator] Successfully distributed %d weights to peer %s", len(missingWeights), shortPeerID(peerIDStr))
}

// onLayerWeightsReceived is called when layer weights are received from a peer.
func (dic *DistributedInferenceCoordinator) onLayerWeightsReceived(layerID int, weights *CPULayerWeights) {
	log.Printf("[Coordinator] Received weights for layer %d", layerID)

	// Store received weights
	dic.mu.Lock()
	dic.localWeights[layerID] = weights
	dic.mu.Unlock()

	// TODO: Load weights into local layer executor (GPU memory)
	// This would involve converting CPULayerWeights to GPU tensors
	// and loading them into the CUDA layer executor
}

// ConnectToPeer initiates connection to a remote peer.
func (dic *DistributedInferenceCoordinator) ConnectToPeer(ctx context.Context, peerInfo peer.AddrInfo) error {
	if dic.peerManager != nil {
		return dic.peerManager.ConnectToPeer(ctx, peerInfo)
	}
	return dic.host.Connect(ctx, peerInfo)
}

// GetRemoteExecutor returns the remote executor for a peer.
func (dic *DistributedInferenceCoordinator) GetRemoteExecutor(peerID string) (*RemoteLayerExecutor, bool) {
	dic.mu.RLock()
	defer dic.mu.RUnlock()
	exec, ok := dic.remoteExecutors[peerID]
	return exec, ok
}

// GetWeightReceiver returns the weight receiver for handling incoming weights.
func (dic *DistributedInferenceCoordinator) GetWeightReceiver() *WeightReceiver {
	return dic.weightReceiver
}

// GetWeightDistributor returns the weight distributor.
func (dic *DistributedInferenceCoordinator) GetWeightDistributor() *WeightDistributor {
	return dic.weightDistributor
}

// IsReady checks if all remote peers are connected and ready.
func (dic *DistributedInferenceCoordinator) IsReady() bool {
	dic.mu.RLock()
	defer dic.mu.RUnlock()

	// Check that all assigned peers have remote executors
	for peerID := range dic.peerAssignments {
		if _, exists := dic.remoteExecutors[peerID]; !exists {
			return false
		}
	}
	return true
}

// WaitForReady blocks until all remote peers are connected or context is cancelled.
func (dic *DistributedInferenceCoordinator) WaitForReady(ctx context.Context) error {
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			if dic.IsReady() {
				return nil
			}
		}
	}
}

// Close releases resources.
func (dic *DistributedInferenceCoordinator) Close() error {
	dic.mu.Lock()
	defer dic.mu.Unlock()

	if dic.closed {
		return nil
	}
	dic.closed = true

	// Close remote executors
	for peerID, exec := range dic.remoteExecutors {
		exec.Close()
		if dic.engine != nil {
			dic.engine.UnregisterRemoteExecutor(peerID)
		}
	}

	// Close weight components
	if dic.weightDistributor != nil {
		dic.weightDistributor.Close()
	}
	if dic.weightReceiver != nil {
		dic.weightReceiver.Close()
	}

	return nil
}

// PerformerCoordinator manages the performer side (receiving weights, executing layers).
type PerformerCoordinator struct {
	host           host.Host
	engine         *Engine
	weightReceiver *WeightReceiver
	remoteExecutor *RemoteExecutor
	config         *types.LlamaConfig
	startLayerID   int
	endLayerID     int

	mu     sync.RWMutex
	closed bool
}

// PerformerCoordinatorConfig holds configuration for creating a PerformerCoordinator.
type PerformerCoordinatorConfig struct {
	Host         host.Host
	Engine       *Engine
	ModelConfig  *types.LlamaConfig
	StartLayerID int
	EndLayerID   int
}

// NewPerformerCoordinator creates a coordinator for the performer side.
func NewPerformerCoordinator(config PerformerCoordinatorConfig) *PerformerCoordinator {
	pc := &PerformerCoordinator{
		host:         config.Host,
		engine:       config.Engine,
		config:       config.ModelConfig,
		startLayerID: config.StartLayerID,
		endLayerID:   config.EndLayerID,
	}

	// Create weight receiver
	pc.weightReceiver = NewWeightReceiver(WeightReceiverConfig{
		Host:            config.Host,
		OnLayerReceived: pc.onLayerWeightsReceived,
	})

	// Create remote executor to handle incoming activation requests
	pc.remoteExecutor = NewRemoteExecutor(RemoteExecutorConfig{
		Host:         config.Host,
		Engine:       config.Engine,
		StartLayerID: config.StartLayerID,
		EndLayerID:   config.EndLayerID,
	})

	log.Printf("[PerformerCoordinator] Initialized for layers %d-%d",
		config.StartLayerID, config.EndLayerID)

	return pc
}

// onLayerWeightsReceived is called when layer weights are received.
func (pc *PerformerCoordinator) onLayerWeightsReceived(layerID int, weights *CPULayerWeights) {
	log.Printf("[PerformerCoordinator] Received weights for layer %d", layerID)

	// TODO: Load weights into local layer executor (GPU memory)
	// This would involve converting CPULayerWeights to GPU tensors
}

// GetWeightReceiver returns the weight receiver.
func (pc *PerformerCoordinator) GetWeightReceiver() *WeightReceiver {
	return pc.weightReceiver
}

// GetRemoteExecutor returns the remote executor handling incoming requests.
func (pc *PerformerCoordinator) GetRemoteExecutor() *RemoteExecutor {
	return pc.remoteExecutor
}

// IsWeightsLoaded checks if all assigned layer weights have been received.
func (pc *PerformerCoordinator) IsWeightsLoaded() bool {
	for layerID := pc.startLayerID; layerID <= pc.endLayerID; layerID++ {
		if _, ok := pc.weightReceiver.GetLayerWeights(layerID); !ok {
			return false
		}
	}
	return true
}

// WaitForWeights blocks until all layer weights are received or context is cancelled.
func (pc *PerformerCoordinator) WaitForWeights(ctx context.Context) error {
	ticker := time.NewTicker(100 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			if pc.IsWeightsLoaded() {
				return nil
			}
		}
	}
}

// Close releases resources.
func (pc *PerformerCoordinator) Close() error {
	pc.mu.Lock()
	defer pc.mu.Unlock()

	if pc.closed {
		return nil
	}
	pc.closed = true

	if pc.weightReceiver != nil {
		pc.weightReceiver.Close()
	}
	if pc.remoteExecutor != nil {
		pc.remoteExecutor.Close()
	}

	return nil
}
