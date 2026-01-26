// Package main provides the coordinator binary for NeuroGrid distributed inference.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/libp2p/go-libp2p"
	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/libp2p/go-libp2p/p2p/discovery/mdns"
	"github.com/multiformats/go-multiaddr"

	"github.com/neurogrid/engine/api"
	"github.com/neurogrid/engine/gpu/bindings"
	"github.com/neurogrid/engine/p2p"
	"github.com/neurogrid/engine/pkg/inference"
	"github.com/neurogrid/engine/pkg/model"
	"github.com/neurogrid/engine/pkg/scheduler"
	"github.com/neurogrid/engine/pkg/transport"
	"github.com/neurogrid/engine/pkg/types"
)

const (
	// ServiceTag for mDNS discovery
	ServiceTag = "neurogrid-worker"

	// PeerDiscoveryTimeout is the maximum time to wait for peers
	PeerDiscoveryTimeout = 300 * time.Second // 5 minutes for distributed setup
)

// CoordinatorConfig holds coordinator configuration
type CoordinatorConfig struct {
	HTTPPort           int
	P2PPort            int
	GPUID              int
	ModelPath          string
	ModelName          string
	MinPeers           int
	LogLevel           string
	EnableCORS         bool
	BootstrapPeers     []peer.AddrInfo // Direct peers to connect to (for WAN connections)
	Role               string          // "coordinator" or "performer"
	SkipWeightTransfer bool            // Skip sending weights to workers (they have local models)
	MaxSeqLen          int             // Maximum sequence length for KV cache (caps model's max_position_embeddings)
}

// Coordinator orchestrates distributed inference
type Coordinator struct {
	config          CoordinatorConfig
	host            host.Host
	ctx             context.Context
	cancel          context.CancelFunc
	scheduler       *scheduler.Scheduler
	router          *transport.TransportRouter
	engine          *inference.Engine
	server          *api.Server
	peers           []peer.AddrInfo
	peersMu         sync.RWMutex
	startTime       time.Time
	p2pProtocol     *p2p.Protocol
	remoteExecutors map[string]*inference.RemoteLayerExecutor // Peer ID -> executor
	modelConfig     *types.LlamaConfig
}

// NewCoordinator creates a new coordinator instance
func NewCoordinator(config CoordinatorConfig) (*Coordinator, error) {
	ctx, cancel := context.WithCancel(context.Background())

	// Create libp2p listen address
	listenAddr, err := multiaddr.NewMultiaddr(fmt.Sprintf("/ip4/0.0.0.0/tcp/%d", config.P2PPort))
	if err != nil {
		cancel()
		return nil, fmt.Errorf("failed to create listen address: %w", err)
	}

	// Create libp2p host
	h, err := libp2p.New(
		libp2p.ListenAddrs(listenAddr),
	)
	if err != nil {
		cancel()
		return nil, fmt.Errorf("failed to create libp2p host: %w", err)
	}

	return &Coordinator{
		config:          config,
		host:            h,
		ctx:             ctx,
		cancel:          cancel,
		peers:           make([]peer.AddrInfo, 0),
		startTime:       time.Now(),
		remoteExecutors: make(map[string]*inference.RemoteLayerExecutor),
	}, nil
}

// Start starts the coordinator
func (c *Coordinator) Start() error {
	log.Println("Starting NeuroGrid coordinator...")
	log.Printf("Role: %s", c.config.Role)
	log.Printf("Coordinator peer ID: %s", c.host.ID())

	// Print multiaddrs for other peers to connect
	for _, addr := range c.host.Addrs() {
		fullAddr := fmt.Sprintf("%s/p2p/%s", addr.String(), c.host.ID())
		log.Printf("Listening on: %s", fullAddr)
	}

	// Subscribe to network connection events to detect incoming worker connections
	c.host.Network().Notify(&networkNotifee{coordinator: c})

	// Connect to bootstrap peers first (for WAN connections)
	if len(c.config.BootstrapPeers) > 0 {
		log.Printf("Connecting to %d bootstrap peers...", len(c.config.BootstrapPeers))
		for _, peerInfo := range c.config.BootstrapPeers {
			log.Printf("Connecting to bootstrap peer: %s", peerInfo.ID)
			ctx, cancel := context.WithTimeout(c.ctx, 30*time.Second)
			err := c.host.Connect(ctx, peerInfo)
			cancel()
			if err != nil {
				log.Printf("Warning: Failed to connect to bootstrap peer %s: %v", peerInfo.ID, err)
			} else {
				log.Printf("Connected to bootstrap peer: %s", peerInfo.ID)
				c.peersMu.Lock()
				c.peers = append(c.peers, peerInfo)
				c.peersMu.Unlock()
			}
		}
	}

	// Setup mDNS discovery (for LAN connections)
	notifee := &coordinatorNotifee{coordinator: c}
	mdnsService := mdns.NewMdnsService(c.host, ServiceTag, notifee)
	if err := mdnsService.Start(); err != nil {
		return fmt.Errorf("failed to start mDNS: %w", err)
	}

	log.Printf("Discovering workers (minimum %d required)...", c.config.MinPeers)

	// Wait for minimum peers
	if err := c.waitForPeers(); err != nil {
		return fmt.Errorf("failed to discover peers: %w", err)
	}

	// Initialize components
	if err := c.initializeComponents(); err != nil {
		return fmt.Errorf("failed to initialize components: %w", err)
	}

	// Start HTTP server
	if err := c.startHTTPServer(); err != nil {
		return fmt.Errorf("failed to start HTTP server: %w", err)
	}

	log.Println("NeuroGrid coordinator started successfully")
	return nil
}

// waitForPeers waits for the minimum number of peers to be discovered
func (c *Coordinator) waitForPeers() error {
	timeout := time.After(PeerDiscoveryTimeout)
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-timeout:
			c.peersMu.RLock()
			peerCount := len(c.peers)
			c.peersMu.RUnlock()

			if peerCount >= c.config.MinPeers {
				log.Printf("Discovered %d peers (timeout reached)", peerCount)
				return nil
			}

			// If no minimum specified, continue without peers (local mode)
			if c.config.MinPeers <= 0 {
				log.Println("No minimum peers required, continuing in local mode")
				return nil
			}

			return fmt.Errorf("timeout waiting for peers (found %d, need %d)", peerCount, c.config.MinPeers)

		case <-ticker.C:
			c.peersMu.RLock()
			peerCount := len(c.peers)
			c.peersMu.RUnlock()

			if peerCount >= c.config.MinPeers {
				log.Printf("Discovered %d peers", peerCount)
				return nil
			}
		}
	}
}

// initializeComponents initializes the inference components
func (c *Coordinator) initializeComponents() error {
	// Get model configuration - try auto-detection from config.json first
	var modelConfig scheduler.ModelConfig
	if c.config.ModelPath != "" {
		cfg, err := getModelConfigFromPath(c.config.ModelPath)
		if err != nil {
			log.Printf("Could not auto-detect model config from %s: %v, using --model-name", c.config.ModelPath, err)
			modelConfig = getModelConfig(c.config.ModelName)
		} else {
			log.Printf("Auto-detected model config from %s/config.json", c.config.ModelPath)
			modelConfig = cfg
			// Also update ModelName from path for correct chat template
			// e.g., "models/tinyllama" -> "tinyllama"
			modelNameFromPath := filepath.Base(c.config.ModelPath)
			if modelNameFromPath != "" && modelNameFromPath != "." {
				c.config.ModelName = modelNameFromPath
				log.Printf("Using model name '%s' for chat template", c.config.ModelName)
			}
		}
	} else {
		modelConfig = getModelConfig(c.config.ModelName)
	}

	// Cap MaxSeqLen to save VRAM (models may support 128k+ but we cap for practical inference)
	if c.config.MaxSeqLen > 0 && modelConfig.MaxSeqLen > c.config.MaxSeqLen {
		log.Printf("Capping MaxSeqLen from %d to %d (use --max-seq-len to adjust)", modelConfig.MaxSeqLen, c.config.MaxSeqLen)
		modelConfig.MaxSeqLen = c.config.MaxSeqLen
	}

	// Create scheduler
	c.scheduler = scheduler.NewScheduler(modelConfig)

	// Set GPU device and query memory
	if err := bindings.SetDevice(c.config.GPUID); err != nil {
		return fmt.Errorf("failed to set GPU device %d: %w", c.config.GPUID, err)
	}
	deviceInfo, err := bindings.GetDeviceInfo()
	if err != nil {
		return fmt.Errorf("failed to get GPU info: %w", err)
	}
	usedVRAM, err := bindings.GetMemoryUsed()
	if err != nil {
		return fmt.Errorf("failed to get GPU memory usage: %w", err)
	}
	totalVRAM := deviceInfo.TotalMemory
	log.Printf("GPU %d: %s, Total VRAM: %.2f GB, Used: %.2f GB, Available: %.2f GB",
		c.config.GPUID, deviceInfo.Name,
		float64(totalVRAM)/(1024*1024*1024),
		float64(usedVRAM)/(1024*1024*1024),
		float64(totalVRAM-usedVRAM)/(1024*1024*1024))

	c.scheduler.RegisterPeer(c.host.ID().String(), totalVRAM, usedVRAM)

	// Register discovered peers only in distributed mode (MinPeers > 0)
	// In local-only mode, all layers run on this node
	if c.config.MinPeers > 0 {
		// TODO: Request actual GPU info from each peer via P2P protocol
		// For now, assume peers have similar GPU specs
		c.peersMu.RLock()
		for _, p := range c.peers {
			// Use same memory values as local GPU (approximation)
			c.scheduler.RegisterPeer(p.ID.String(), totalVRAM, usedVRAM)
		}
		c.peersMu.RUnlock()
	} else {
		log.Printf("Running in local-only mode: all layers assigned to this node")
	}

	// Compute layer assignments
	assignments, err := c.scheduler.ComputeAssignments()
	if err != nil {
		return fmt.Errorf("failed to compute assignments: %w", err)
	}
	log.Printf("Layer assignments computed: %d layers distributed", len(assignments))
	// Debug: Print detailed assignments
	for _, a := range assignments {
		isLocal := a.PeerID == c.host.ID().String()
		log.Printf("[DEBUG] Layer %d -> Peer %s (local=%v)", a.LayerID, a.PeerID[:16]+"...", isLocal)
	}

	// Create transport router
	c.router = transport.NewTransportRouter()

	// Register local transport
	localTransport, err := transport.NewCUDATransport(c.config.GPUID, c.config.GPUID)
	if err != nil {
		return fmt.Errorf("failed to create local transport: %w", err)
	}
	c.router.RegisterLocalTransport(c.config.GPUID, localTransport)
	c.router.RegisterRemoteTransport(c.host.ID().String(), localTransport)

	// Assign layers to peers in router
	for _, assignment := range assignments {
		c.router.AssignLayerToPeer(assignment.LayerID, assignment.PeerID)
	}

	// Create inference engine config using types.LlamaConfig
	ropeTheta := float32(modelConfig.RopeTheta)
	if ropeTheta == 0 {
		ropeTheta = 10000.0 // Default for Llama 2
	}
	llamaConfig := &types.LlamaConfig{
		HiddenSize:       int(modelConfig.HiddenSize),
		NumLayers:        modelConfig.NumLayers,
		IntermediateSize: int(modelConfig.IntermediateSize),
		NumHeads:         modelConfig.NumHeads,
		NumKVHeads:       modelConfig.NumKVHeads,
		HeadDim:          modelConfig.HeadDim,
		VocabSize:        int(modelConfig.VocabSize),
		MaxSeqLen:        modelConfig.MaxSeqLen,
		RMSNormEps:       modelConfig.RMSNormEps,
		RopeTheta:        ropeTheta,
	}

	engineConfig := inference.EngineConfig{
		ModelConfig: llamaConfig,
		LocalPeerID: c.host.ID().String(),
	}

	// Store model config for later use
	c.modelConfig = llamaConfig

	// Create inference engine
	c.engine = inference.NewEngine(engineConfig)
	c.engine.SetScheduler(c.scheduler)
	c.engine.SetRouter(c.router)
	c.engine.SetAssignments(assignments)

	// Create P2P protocol for tensor transfer
	c.p2pProtocol = p2p.NewProtocol(c.host)

	// Create RemoteLayerExecutors for each remote peer
	// Group layers by peer to determine layer ranges
	peerLayers := make(map[string][]int) // Peer ID -> list of layer IDs
	for _, assignment := range assignments {
		if assignment.PeerID != c.host.ID().String() && assignment.LayerID >= 0 && assignment.LayerID < llamaConfig.NumLayers {
			peerLayers[assignment.PeerID] = append(peerLayers[assignment.PeerID], assignment.LayerID)
		}
	}

	// Create executor for each remote peer
	for peerIDStr, layers := range peerLayers {
		if len(layers) == 0 {
			continue
		}

		// Find the peer.ID from our peers list
		var targetPeerID peer.ID
		c.peersMu.RLock()
		for _, p := range c.peers {
			if p.ID.String() == peerIDStr {
				targetPeerID = p.ID
				break
			}
		}
		c.peersMu.RUnlock()

		if targetPeerID == "" {
			log.Printf("Warning: Cannot find peer %s in connected peers list", peerIDStr)
			continue
		}

		// Determine layer range (min, max)
		startLayer, endLayer := layers[0], layers[0]
		for _, l := range layers {
			if l < startLayer {
				startLayer = l
			}
			if l > endLayer {
				endLayer = l
			}
		}

		// Create RemoteLayerExecutor with shared protocol
		// All executors must share the same protocol to ensure response routing works correctly
		remoteExec := inference.NewRemoteLayerExecutor(inference.RemoteLayerExecutorConfig{
			Host:         c.host,
			Protocol:     c.p2pProtocol, // Share protocol across all executors
			TargetPeerID: targetPeerID,
			StartLayerID: startLayer,
			EndLayerID:   endLayer,
			Config:       llamaConfig,
			Timeout:      60 * time.Second,
		})

		c.remoteExecutors[peerIDStr] = remoteExec
		c.engine.RegisterRemoteExecutor(peerIDStr, remoteExec)

		log.Printf("Created RemoteLayerExecutor for peer %s: layers %d-%d", peerIDStr, startLayer, endLayer)

		// Send layer assignment to this peer (for Weight Distributed Memory)
		go func(pID peer.ID, layerList []int) {
			ctx, cancel := context.WithTimeout(c.ctx, 30*time.Second)
			defer cancel()
			log.Printf("Sending layer assignment to peer %s: %d layers", pID, len(layerList))
			if err := c.p2pProtocol.SendLayerRequest(ctx, pID, layerList); err != nil {
				log.Printf("Warning: Failed to send layer assignment to %s: %v", pID, err)
			} else {
				log.Printf("Layer assignment sent to peer %s: layers %v", pID, layerList)
			}
		}(targetPeerID, layers)
	}

	// Load tokenizer from model directory
	// Try tokenizer.json first (HuggingFace/GPT-2 style), then fall back to tokenizer.model (SentencePiece)
	var tokenizer inference.Tokenizer
	if tok, err := model.NewTokenizer(c.config.ModelPath); err == nil {
		log.Printf("Loaded BPE tokenizer from %s (vocab size: %d)", c.config.ModelPath, tok.VocabSize())
		tokenizer = tok
	} else if spTok, spErr := model.NewSentencePieceTokenizer(c.config.ModelPath); spErr == nil {
		log.Printf("Loaded SentencePiece tokenizer from %s (vocab size: %d)", c.config.ModelPath, spTok.VocabSize())
		tokenizer = spTok
	} else {
		log.Printf("Warning: Failed to load tokenizer from %s: %v (BPE) / %v (SentencePiece)", c.config.ModelPath, err, spErr)
		log.Printf("Using mock tokenizer for testing")
		tokenizer = model.NewMockSentencePieceTokenizer()
	}
	c.engine.SetTokenizer(tokenizer)

	// Load model weights to GPU
	log.Printf("Loading model weights from %s...", c.config.ModelPath)
	weightLoader, err := model.NewWeightLoader(c.config.ModelPath)
	if err != nil {
		log.Printf("Warning: Failed to create weight loader: %v", err)
		log.Printf("Running in mock mode - inference will return placeholder data")
	} else {
		defer weightLoader.Close()

		// Initialize GPU and load all weights
		gpuComponents, err := c.engine.InitializeGPU(weightLoader, c.config.GPUID)
		if err != nil {
			log.Printf("Warning: GPU initialization failed: %v", err)
			log.Printf("Falling back to CPU mode with mock inference")

			// Fallback: load embeddings to CPU for basic operation
			embeddings, embedInfo, err := weightLoader.LoadEmbeddings()
			if err != nil {
				log.Printf("Warning: Failed to load embeddings: %v", err)
			} else {
				c.engine.LoadEmbeddings(embeddings)
				log.Printf("Loaded embeddings to CPU: %v (%.2f MB)", embedInfo.Shape, float64(len(embeddings))/(1024*1024))
			}
		} else {
			log.Printf("GPU inference pipeline initialized successfully")
			// Store GPU components for cleanup
			_ = gpuComponents // TODO: store for cleanup on shutdown
		}

		log.Printf("Model weights loaded successfully")

		// Distribute weights to remote peers (unless workers have local models)
		if len(peerLayers) > 0 {
			if c.config.SkipWeightTransfer {
				log.Printf("Skipping weight distribution to %d remote peers (workers have local models)", len(peerLayers))
			} else {
				log.Printf("Distributing weights to %d remote peers...", len(peerLayers))
				if err := c.distributeWeightsToRemotePeers(weightLoader, assignments); err != nil {
					log.Printf("Warning: Weight distribution failed: %v", err)
					log.Printf("Remote peers will need to load weights locally")
				} else {
					log.Printf("Weight distribution complete")
				}
			}
		}
	}

	log.Printf("Inference engine initialized with %d peers", len(c.peers)+1)
	return nil
}

// distributeWeightsToRemotePeers sends layer weights to remote peers via P2P.
func (c *Coordinator) distributeWeightsToRemotePeers(weightLoader *model.WeightLoader, assignments []scheduler.LayerAssignment) error {
	ctx, cancel := context.WithTimeout(c.ctx, 10*time.Minute)
	defer cancel()

	// Group assignments by peer
	peerAssignments := make(map[string][]scheduler.LayerAssignment)
	for _, a := range assignments {
		if a.PeerID != c.host.ID().String() && a.LayerID >= 0 && a.LayerID < c.modelConfig.NumLayers {
			peerAssignments[a.PeerID] = append(peerAssignments[a.PeerID], a)
		}
	}

	// Send weights to each peer
	for peerIDStr, layerAssignments := range peerAssignments {
		// Find peer.ID
		var targetPeerID peer.ID
		c.peersMu.RLock()
		for _, p := range c.peers {
			if p.ID.String() == peerIDStr {
				targetPeerID = p.ID
				break
			}
		}
		c.peersMu.RUnlock()

		if targetPeerID == "" {
			log.Printf("Warning: Skipping weight distribution to unknown peer %s", peerIDStr)
			continue
		}

		log.Printf("Sending %d layer weights to peer %s...", len(layerAssignments), peerIDStr)

		for _, a := range layerAssignments {
			// Load layer weights
			weights, err := weightLoader.LoadLayerWeights(a.LayerID)
			if err != nil {
				log.Printf("Warning: Failed to load weights for layer %d: %v", a.LayerID, err)
				continue
			}

			// Serialize weights
			data, err := model.SerializeLayerWeights(weights)
			if err != nil {
				log.Printf("Warning: Failed to serialize weights for layer %d: %v", a.LayerID, err)
				continue
			}

			// Send via P2P protocol
			err = c.p2pProtocol.SendWeights(ctx, targetPeerID, a.LayerID, data)
			if err != nil {
				log.Printf("Warning: Failed to send weights for layer %d to peer %s: %v", a.LayerID, peerIDStr, err)
				continue
			}

			log.Printf("  Layer %d: %.2f MB sent", a.LayerID, float64(len(data))/(1024*1024))
		}
	}

	return nil
}

// GetClusterInfo implements api.ClusterInfoProvider
func (c *Coordinator) GetClusterInfo() api.ClusterInfo {
	c.peersMu.RLock()
	numPeers := len(c.peers)
	c.peersMu.RUnlock()

	// Get layer assignments from router
	layerAssignments := make(map[int]string)
	if c.router != nil {
		layerAssignments = c.router.GetAllLayerAssignments()
	}

	// Get total and loaded layers
	totalLayers := 0
	loadedLayers := 0
	if c.modelConfig != nil {
		totalLayers = c.modelConfig.NumLayers
		loadedLayers = totalLayers // Assume all loaded for now
	}

	return api.ClusterInfo{
		PeerID:           c.host.ID().String(),
		ConnectedPeers:   numPeers,
		LayerAssignments: layerAssignments,
		MemoryUsage:      make(map[string]uint64), // TODO: implement
		TotalLayers:      totalLayers,
		LoadedLayers:     loadedLayers,
	}
}

// startHTTPServer starts the HTTP API server
func (c *Coordinator) startHTTPServer() error {
	serverConfig := api.ServerConfig{
		Addr:         fmt.Sprintf(":%d", c.config.HTTPPort),
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 600 * time.Second, // 10 minutes for slow inference
		ModelName:    c.config.ModelName,
		EnableCORS:   c.config.EnableCORS,
	}

	c.server = api.NewServer(c.engine, serverConfig)

	// Set cluster info provider
	c.server.SetClusterInfoProvider(c)

	// Start server in goroutine
	go func() {
		log.Printf("HTTP API server starting on :%d", c.config.HTTPPort)
		if err := c.server.Start(); err != nil {
			log.Printf("HTTP server error: %v", err)
		}
	}()

	return nil
}

// Shutdown gracefully shuts down the coordinator
func (c *Coordinator) Shutdown() error {
	log.Println("Shutting down coordinator...")

	// Shutdown HTTP server
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if c.server != nil {
		if err := c.server.Shutdown(ctx); err != nil {
			log.Printf("HTTP server shutdown error: %v", err)
		}
	}

	// Cancel context
	c.cancel()

	// Close libp2p host
	if err := c.host.Close(); err != nil {
		return fmt.Errorf("failed to close host: %w", err)
	}

	log.Println("Coordinator shutdown complete")
	return nil
}

// GetStats returns coordinator statistics
func (c *Coordinator) GetStats() map[string]interface{} {
	c.peersMu.RLock()
	peerCount := len(c.peers)
	c.peersMu.RUnlock()

	return map[string]interface{}{
		"peer_id":     c.host.ID().String(),
		"http_port":   c.config.HTTPPort,
		"p2p_port":    c.config.P2PPort,
		"model":       c.config.ModelName,
		"peer_count":  peerCount,
		"uptime_secs": time.Since(c.startTime).Seconds(),
	}
}

// coordinatorNotifee handles peer discovery notifications
type coordinatorNotifee struct {
	coordinator *Coordinator
}

// networkNotifee handles network connection events
type networkNotifee struct {
	coordinator *Coordinator
}

// Connected is called when a peer connects
func (n *networkNotifee) Connected(net network.Network, conn network.Conn) {
	peerID := conn.RemotePeer()

	// Don't add self
	if peerID == n.coordinator.host.ID() {
		return
	}

	// Atomic check-and-add to prevent race condition duplicates
	n.coordinator.peersMu.Lock()
	defer n.coordinator.peersMu.Unlock()

	// Check if already in peers list
	for _, p := range n.coordinator.peers {
		if p.ID == peerID {
			return // Already registered
		}
	}

	// Add new peer
	pi := peer.AddrInfo{
		ID:    peerID,
		Addrs: []multiaddr.Multiaddr{conn.RemoteMultiaddr()},
	}

	n.coordinator.peers = append(n.coordinator.peers, pi)
	peerCount := len(n.coordinator.peers)

	log.Printf("Worker connected: %s (total workers: %d)", peerID, peerCount)
}

// Disconnected is called when a peer disconnects
func (n *networkNotifee) Disconnected(net network.Network, conn network.Conn) {
	peerID := conn.RemotePeer()
	log.Printf("Worker disconnected: %s", peerID)
}

// Listen is called when we start listening on an address (required by interface)
func (n *networkNotifee) Listen(net network.Network, ma multiaddr.Multiaddr) {}

// ListenClose is called when we stop listening on an address (required by interface)
func (n *networkNotifee) ListenClose(net network.Network, ma multiaddr.Multiaddr) {}

// HandlePeerFound is called when a peer is discovered
func (n *coordinatorNotifee) HandlePeerFound(pi peer.AddrInfo) {
	// Don't add self
	if pi.ID == n.coordinator.host.ID() {
		return
	}

	log.Printf("Discovered worker: %s", pi.ID)

	// Connect to the peer
	if err := n.coordinator.host.Connect(n.coordinator.ctx, pi); err != nil {
		log.Printf("Failed to connect to peer %s: %v", pi.ID, err)
		return
	}

	// Atomic check-and-add to prevent duplicate peers
	n.coordinator.peersMu.Lock()
	defer n.coordinator.peersMu.Unlock()

	// Check if already in peers list
	for _, p := range n.coordinator.peers {
		if p.ID == pi.ID {
			return // Already registered
		}
	}

	// Add new peer
	n.coordinator.peers = append(n.coordinator.peers, pi)
	log.Printf("Connected to worker: %s (total: %d)", pi.ID, len(n.coordinator.peers))
}

// getModelConfig returns configuration for the specified model
// getModelConfigFromPath auto-detects model configuration from config.json
func getModelConfigFromPath(modelPath string) (scheduler.ModelConfig, error) {
	cfg, err := model.LoadModelConfig(modelPath)
	if err != nil {
		return scheduler.ModelConfig{}, err
	}

	// HeadDim = HiddenSize / NumAttentionHeads
	headDim := cfg.HiddenSize / cfg.NumAttentionHeads
	if headDim == 0 {
		headDim = 128 // Default
	}

	return scheduler.ModelConfig{
		HiddenSize:       int64(cfg.HiddenSize),
		IntermediateSize: int64(cfg.IntermediateSize),
		NumLayers:        cfg.NumHiddenLayers,
		NumHeads:         cfg.NumAttentionHeads,
		NumKVHeads:       cfg.NumKeyValueHeads,
		HeadDim:          headDim,
		MaxSeqLen:        cfg.MaxPositionEmbeddings,
		VocabSize:        int64(cfg.VocabSize),
		RMSNormEps:       float32(cfg.RMSNormEps),
	}, nil
}

func getModelConfig(modelName string) scheduler.ModelConfig {
	switch modelName {
	case "llama-7b", "llama-2-7b":
		return scheduler.DefaultLlama7BConfig()
	case "llama-13b", "llama-2-13b":
		return scheduler.DefaultLlama13BConfig()
	case "llama-70b", "llama-2-70b", "llama3-70b", "llama-3.3-70b":
		return scheduler.DefaultLlama3_70BConfig()
	case "tinyllama", "tinyllama-1.1b":
		return scheduler.DefaultTinyLlamaConfig()
	case "mistral-7b", "mistral-7b-instruct":
		return scheduler.DefaultMistral7BConfig()
	default:
		// Default to 7B config
		log.Printf("Unknown model %s, defaulting to llama-7b config", modelName)
		return scheduler.DefaultLlama7BConfig()
	}
}

func main() {
	// Parse command line flags
	httpPort := flag.Int("http-port", 8090, "HTTP API port")
	p2pPort := flag.Int("p2p-port", 9000, "libp2p port for peer communication")
	gpuID := flag.Int("gpu", 0, "Local GPU device ID")
	modelPath := flag.String("model", "", "Path to model weights")
	modelName := flag.String("model-name", "llama-7b", "Model name (llama-7b, llama-13b)")
	minPeers := flag.Int("min-peers", 0, "Minimum number of workers to wait for (0 = local only)")
	logLevel := flag.String("log-level", "info", "Log level (debug, info, warn, error)")
	enableCORS := flag.Bool("cors", true, "Enable CORS headers")
	bootstrapStr := flag.String("bootstrap", "", "Bootstrap peer multiaddr (e.g., /ip4/192.168.1.100/tcp/9000/p2p/12D3KooW...)")
	role := flag.String("role", "coordinator", "Node role: coordinator (orchestrates inference) or performer (executes layers)")
	skipWeightTransfer := flag.Bool("skip-weight-transfer", false, "Skip sending weights to workers (use when workers have local models)")
	maxSeqLen := flag.Int("max-seq-len", 4096, "Maximum sequence length for KV cache (caps model's max_position_embeddings to save VRAM)")
	flag.Parse()

	// Parse bootstrap peers
	var bootstrapPeers []peer.AddrInfo
	if *bootstrapStr != "" {
		for _, addrStr := range strings.Split(*bootstrapStr, ",") {
			addrStr = strings.TrimSpace(addrStr)
			if addrStr == "" {
				continue
			}
			maddr, err := multiaddr.NewMultiaddr(addrStr)
			if err != nil {
				log.Fatalf("Invalid bootstrap multiaddr %q: %v", addrStr, err)
			}
			peerInfo, err := peer.AddrInfoFromP2pAddr(maddr)
			if err != nil {
				log.Fatalf("Failed to parse peer info from %q: %v", addrStr, err)
			}
			bootstrapPeers = append(bootstrapPeers, *peerInfo)
			log.Printf("Bootstrap peer: %s", peerInfo.ID)
		}
	}

	// Create coordinator config
	config := CoordinatorConfig{
		HTTPPort:           *httpPort,
		P2PPort:            *p2pPort,
		GPUID:              *gpuID,
		ModelPath:          *modelPath,
		ModelName:          *modelName,
		MinPeers:           *minPeers,
		LogLevel:           *logLevel,
		EnableCORS:         *enableCORS,
		BootstrapPeers:     bootstrapPeers,
		Role:               *role,
		SkipWeightTransfer: *skipWeightTransfer,
		MaxSeqLen:          *maxSeqLen,
	}

	log.Println("================================================")
	log.Println("    NeuroGrid Distributed Inference Engine      ")
	log.Println("================================================")
	log.Printf("Role: %s", config.Role)
	log.Printf("Model: %s", config.ModelName)
	log.Printf("HTTP Port: %d", config.HTTPPort)
	log.Printf("P2P Port: %d", config.P2PPort)
	log.Printf("GPU: %d", config.GPUID)
	log.Printf("Max Seq Len: %d", config.MaxSeqLen)
	if config.ModelPath != "" {
		log.Printf("Model Path: %s", config.ModelPath)
	}
	if len(config.BootstrapPeers) > 0 {
		log.Printf("Bootstrap Peers: %d", len(config.BootstrapPeers))
	}

	// Create and start coordinator
	coordinator, err := NewCoordinator(config)
	if err != nil {
		log.Fatalf("Failed to create coordinator: %v", err)
	}

	if err := coordinator.Start(); err != nil {
		log.Fatalf("Failed to start coordinator: %v", err)
	}

	log.Println("================================================")
	log.Printf("API available at http://localhost:%d", config.HTTPPort)
	log.Printf("OpenAI-compatible endpoint: http://localhost:%d/v1/chat/completions", config.HTTPPort)
	log.Println("================================================")

	// Wait for shutdown signal
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	sig := <-sigChan
	log.Printf("Received signal: %v", sig)

	if err := coordinator.Shutdown(); err != nil {
		log.Fatalf("Shutdown failed: %v", err)
	}
}
