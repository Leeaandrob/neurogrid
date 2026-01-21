// Package main provides the worker node binary for distributed inference.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/libp2p/go-libp2p"
	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/libp2p/go-libp2p/p2p/discovery/mdns"
	"github.com/multiformats/go-multiaddr"
)

const (
	// ProtocolID is the protocol identifier for tensor transfer
	ProtocolID = "/neurogrid/worker/1.0.0"

	// ServiceTag for mDNS discovery
	ServiceTag = "neurogrid-worker"
)

// WorkerConfig holds worker configuration
type WorkerConfig struct {
	Port      int
	GPUID     int
	ModelPath string
	LogLevel  string
}

// Worker represents a distributed inference worker node
type Worker struct {
	config     WorkerConfig
	host       host.Host
	ctx        context.Context
	cancel     context.CancelFunc
	layerData  map[int][]byte // Layer weights loaded on this worker
	peerCount  int
	startTime  time.Time
	reqCounter uint64
}

// NewWorker creates a new worker instance
func NewWorker(config WorkerConfig) (*Worker, error) {
	ctx, cancel := context.WithCancel(context.Background())

	// Create libp2p listen address
	listenAddr, err := multiaddr.NewMultiaddr(fmt.Sprintf("/ip4/0.0.0.0/tcp/%d", config.Port))
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

	return &Worker{
		config:    config,
		host:      h,
		ctx:       ctx,
		cancel:    cancel,
		layerData: make(map[int][]byte),
		startTime: time.Now(),
	}, nil
}

// Start starts the worker and begins listening for requests
func (w *Worker) Start() error {
	// Setup stream handler for layer requests
	w.host.SetStreamHandler(ProtocolID, w.handleStream)

	// Setup mDNS discovery
	notifee := &discoveryNotifee{worker: w}
	mdnsService := mdns.NewMdnsService(w.host, ServiceTag, notifee)
	if err := mdnsService.Start(); err != nil {
		return fmt.Errorf("failed to start mDNS: %w", err)
	}

	// Log startup info
	log.Printf("Worker started successfully")
	log.Printf("  Peer ID: %s", w.host.ID())
	log.Printf("  GPU: %d", w.config.GPUID)
	log.Printf("  Port: %d", w.config.Port)
	log.Printf("  Model: %s", w.config.ModelPath)

	// Log all listen addresses
	for _, addr := range w.host.Addrs() {
		log.Printf("  Listening: %s/p2p/%s", addr, w.host.ID())
	}

	return nil
}

// handleStream handles incoming stream requests
func (w *Worker) handleStream(s network.Stream) {
	defer s.Close()

	w.reqCounter++
	log.Printf("[%d] Incoming request from %s", w.reqCounter, s.Conn().RemotePeer())

	// Read request header (simplified protocol)
	header := make([]byte, 8)
	if _, err := s.Read(header); err != nil {
		log.Printf("Failed to read header: %v", err)
		return
	}

	// Parse layer ID from header (first 4 bytes)
	layerID := int(header[0])<<24 | int(header[1])<<16 | int(header[2])<<8 | int(header[3])

	// Read activation size (next 4 bytes)
	size := int(header[4])<<24 | int(header[5])<<16 | int(header[6])<<8 | int(header[7])

	// Read activation data
	activation := make([]byte, size)
	if _, err := s.Read(activation); err != nil {
		log.Printf("Failed to read activation: %v", err)
		return
	}

	// Execute layer (mock forward pass)
	output, err := w.executeLayer(layerID, activation)
	if err != nil {
		log.Printf("Layer execution failed: %v", err)
		s.Write([]byte{0xFF, 0xFF, 0xFF, 0xFF}) // Error marker
		return
	}

	// Send response header (size)
	respHeader := make([]byte, 4)
	respHeader[0] = byte(len(output) >> 24)
	respHeader[1] = byte(len(output) >> 16)
	respHeader[2] = byte(len(output) >> 8)
	respHeader[3] = byte(len(output))
	s.Write(respHeader)

	// Send output
	s.Write(output)

	log.Printf("[%d] Completed layer %d forward pass (%d -> %d bytes)",
		w.reqCounter, layerID, size, len(output))
}

// executeLayer executes a forward pass through the specified layer
func (w *Worker) executeLayer(layerID int, activation []byte) ([]byte, error) {
	// Mock output (same size as input for transformer layers)
	output := make([]byte, len(activation))
	copy(output, activation)

	// Simulate some computation time
	time.Sleep(time.Millisecond)

	return output, nil
}

// LoadLayerWeights loads weights for a specific layer
func (w *Worker) LoadLayerWeights(layerID int, weights []byte) error {
	w.layerData[layerID] = weights
	log.Printf("Loaded weights for layer %d (%d bytes)", layerID, len(weights))
	return nil
}

// Shutdown gracefully shuts down the worker
func (w *Worker) Shutdown() error {
	log.Println("Shutting down worker...")

	w.cancel()

	if err := w.host.Close(); err != nil {
		return fmt.Errorf("failed to close host: %w", err)
	}

	log.Println("Worker shutdown complete")
	return nil
}

// GetStats returns worker statistics
func (w *Worker) GetStats() map[string]interface{} {
	return map[string]interface{}{
		"peer_id":       w.host.ID().String(),
		"gpu_id":        w.config.GPUID,
		"port":          w.config.Port,
		"uptime_secs":   time.Since(w.startTime).Seconds(),
		"requests":      w.reqCounter,
		"peer_count":    w.peerCount,
		"layers_loaded": len(w.layerData),
	}
}

// discoveryNotifee handles peer discovery notifications
type discoveryNotifee struct {
	worker *Worker
}

// HandlePeerFound is called when a peer is discovered
func (n *discoveryNotifee) HandlePeerFound(pi peer.AddrInfo) {
	n.worker.peerCount++
	log.Printf("Discovered peer: %s", pi.ID)

	// Connect to the peer
	if err := n.worker.host.Connect(n.worker.ctx, pi); err != nil {
		log.Printf("Failed to connect to peer %s: %v", pi.ID, err)
		return
	}

	log.Printf("Connected to peer: %s", pi.ID)
}

func main() {
	// Parse command line flags
	port := flag.Int("port", 9000, "libp2p listen port")
	gpuID := flag.Int("gpu", 0, "GPU device ID")
	modelPath := flag.String("model", "", "Path to model weights")
	logLevel := flag.String("log-level", "info", "Log level (debug, info, warn, error)")
	flag.Parse()

	// Validate flags
	if *modelPath == "" {
		log.Println("Warning: No model path specified")
	}

	// Create worker config
	config := WorkerConfig{
		Port:      *port,
		GPUID:     *gpuID,
		ModelPath: *modelPath,
		LogLevel:  *logLevel,
	}

	// Create and start worker
	worker, err := NewWorker(config)
	if err != nil {
		log.Fatalf("Failed to create worker: %v", err)
	}

	if err := worker.Start(); err != nil {
		log.Fatalf("Failed to start worker: %v", err)
	}

	// Wait for shutdown signal
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	sig := <-sigChan
	log.Printf("Received signal: %v", sig)

	if err := worker.Shutdown(); err != nil {
		log.Fatalf("Shutdown failed: %v", err)
	}
}
