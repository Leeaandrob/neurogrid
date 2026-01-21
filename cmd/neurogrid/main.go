// Package main provides the coordinator binary for NeuroGrid distributed inference.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	"github.com/libp2p/go-libp2p"
	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/libp2p/go-libp2p/p2p/discovery/mdns"
	"github.com/multiformats/go-multiaddr"

	"github.com/neurogrid/engine/api"
	"github.com/neurogrid/engine/gpu/bindings"
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
	PeerDiscoveryTimeout = 60 * time.Second
)

// CoordinatorConfig holds coordinator configuration
type CoordinatorConfig struct {
	HTTPPort   int
	P2PPort    int
	GPUID      int
	ModelPath  string
	ModelName  string
	MinPeers   int
	LogLevel   string
	EnableCORS bool
}

// Coordinator orchestrates distributed inference
type Coordinator struct {
	config    CoordinatorConfig
	host      host.Host
	ctx       context.Context
	cancel    context.CancelFunc
	scheduler *scheduler.Scheduler
	router    *transport.TransportRouter
	engine    *inference.Engine
	server    *api.Server
	peers     []peer.AddrInfo
	peersMu   sync.RWMutex
	startTime time.Time
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
		config:    config,
		host:      h,
		ctx:       ctx,
		cancel:    cancel,
		peers:     make([]peer.AddrInfo, 0),
		startTime: time.Now(),
	}, nil
}

// Start starts the coordinator
func (c *Coordinator) Start() error {
	log.Println("Starting NeuroGrid coordinator...")

	// Setup mDNS discovery
	notifee := &coordinatorNotifee{coordinator: c}
	mdnsService := mdns.NewMdnsService(c.host, ServiceTag, notifee)
	if err := mdnsService.Start(); err != nil {
		return fmt.Errorf("failed to start mDNS: %w", err)
	}

	log.Printf("Coordinator peer ID: %s", c.host.ID())
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
	// Get model configuration
	modelConfig := getModelConfig(c.config.ModelName)

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
	}

	engineConfig := inference.EngineConfig{
		ModelConfig: llamaConfig,
		LocalPeerID: c.host.ID().String(),
	}

	// Create inference engine
	c.engine = inference.NewEngine(engineConfig)
	c.engine.SetScheduler(c.scheduler)
	c.engine.SetRouter(c.router)
	c.engine.SetAssignments(assignments)

	// Load tokenizer from model directory
	tokenizer, err := model.NewSentencePieceTokenizer(c.config.ModelPath)
	if err != nil {
		log.Printf("Warning: Failed to load tokenizer from %s: %v", c.config.ModelPath, err)
		log.Printf("Using mock tokenizer for testing")
		tokenizer = model.NewMockSentencePieceTokenizer()
	} else {
		log.Printf("Loaded tokenizer from %s (vocab size: %d)", c.config.ModelPath, tokenizer.VocabSize())
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
	}

	log.Printf("Inference engine initialized with %d peers", len(c.peers)+1)
	return nil
}

// startHTTPServer starts the HTTP API server
func (c *Coordinator) startHTTPServer() error {
	serverConfig := api.ServerConfig{
		Addr:         fmt.Sprintf(":%d", c.config.HTTPPort),
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 120 * time.Second,
		ModelName:    c.config.ModelName,
		EnableCORS:   c.config.EnableCORS,
	}

	c.server = api.NewServer(c.engine, serverConfig)

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

// HandlePeerFound is called when a peer is discovered
func (n *coordinatorNotifee) HandlePeerFound(pi peer.AddrInfo) {
	log.Printf("Discovered worker: %s", pi.ID)

	// Connect to the peer
	if err := n.coordinator.host.Connect(n.coordinator.ctx, pi); err != nil {
		log.Printf("Failed to connect to peer %s: %v", pi.ID, err)
		return
	}

	// Add to peers list
	n.coordinator.peersMu.Lock()
	n.coordinator.peers = append(n.coordinator.peers, pi)
	n.coordinator.peersMu.Unlock()

	log.Printf("Connected to worker: %s (total: %d)", pi.ID, len(n.coordinator.peers))
}

// getModelConfig returns configuration for the specified model
func getModelConfig(modelName string) scheduler.ModelConfig {
	switch modelName {
	case "llama-7b":
		return scheduler.DefaultLlama7BConfig()
	case "llama-13b":
		return scheduler.DefaultLlama13BConfig()
	case "tinyllama", "tinyllama-1.1b":
		return scheduler.DefaultTinyLlamaConfig()
	default:
		// Default to 7B config
		log.Printf("Unknown model %s, defaulting to llama-7b config", modelName)
		return scheduler.DefaultLlama7BConfig()
	}
}

func main() {
	// Parse command line flags
	httpPort := flag.Int("http-port", 8080, "HTTP API port")
	p2pPort := flag.Int("p2p-port", 9000, "libp2p port for peer communication")
	gpuID := flag.Int("gpu", 0, "Local GPU device ID")
	modelPath := flag.String("model", "", "Path to model weights")
	modelName := flag.String("model-name", "llama-7b", "Model name (llama-7b, llama-13b)")
	minPeers := flag.Int("min-peers", 0, "Minimum number of workers to wait for (0 = local only)")
	logLevel := flag.String("log-level", "info", "Log level (debug, info, warn, error)")
	enableCORS := flag.Bool("cors", true, "Enable CORS headers")
	flag.Parse()

	// Create coordinator config
	config := CoordinatorConfig{
		HTTPPort:   *httpPort,
		P2PPort:    *p2pPort,
		GPUID:      *gpuID,
		ModelPath:  *modelPath,
		ModelName:  *modelName,
		MinPeers:   *minPeers,
		LogLevel:   *logLevel,
		EnableCORS: *enableCORS,
	}

	log.Println("================================================")
	log.Println("    NeuroGrid Distributed Inference Engine      ")
	log.Println("================================================")
	log.Printf("Model: %s", config.ModelName)
	log.Printf("HTTP Port: %d", config.HTTPPort)
	log.Printf("P2P Port: %d", config.P2PPort)
	log.Printf("GPU: %d", config.GPUID)
	if config.ModelPath != "" {
		log.Printf("Model Path: %s", config.ModelPath)
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
