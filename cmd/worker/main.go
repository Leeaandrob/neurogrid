// Package main provides the worker node binary for distributed inference.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"runtime"
	"strings"
	"sync"
	"syscall"
	"time"
	"unsafe"

	"github.com/libp2p/go-libp2p"
	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/libp2p/go-libp2p/p2p/discovery/mdns"
	"github.com/multiformats/go-multiaddr"

	"github.com/neurogrid/engine/gpu/bindings"
	"github.com/neurogrid/engine/p2p"
	"github.com/neurogrid/engine/pkg/inference"
	"github.com/neurogrid/engine/pkg/model"
	"github.com/neurogrid/engine/pkg/types"
)

const (
	// ServiceTag for mDNS discovery
	ServiceTag = "neurogrid-worker"
)

// WorkerConfig holds worker configuration
type WorkerConfig struct {
	Port           int
	GPUID          int
	ModelPath      string
	ModelName      string
	LogLevel       string
	BootstrapPeers []peer.AddrInfo // Coordinator address for WAN connections
}

// chunkBuffer accumulates weight chunks for a layer
type chunkBuffer struct {
	chunks      [][]byte
	totalChunks int
	received    int
}

// Worker represents a distributed inference worker node (performer role)
type Worker struct {
	config       WorkerConfig
	host         host.Host
	ctx          context.Context
	cancel       context.CancelFunc
	protocol     *p2p.Protocol
	engine       *inference.Engine
	layerWeights map[int]*model.LayerWeights    // Layer ID -> weights (CPU)
	gpuWeights   map[int]*bindings.LayerWeights // Layer ID -> weights (GPU)
	gpuKVCaches  map[int]*bindings.KVCache      // Layer ID -> KV cache (GPU)
	kvCaches     *inference.KVCacheManager
	modelConfig  *types.LlamaConfig
	startLayerID int
	endLayerID   int
	peersMu      sync.RWMutex
	peers        []peer.AddrInfo
	startTime    time.Time
	reqCounter   uint64
	weightsReady bool
	weightsMu    sync.RWMutex
	chunkBuffers map[int]*chunkBuffer // Layer ID -> chunk buffer
	chunkMu      sync.Mutex
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
		config:       config,
		host:         h,
		ctx:          ctx,
		cancel:       cancel,
		layerWeights: make(map[int]*model.LayerWeights),
		gpuWeights:   make(map[int]*bindings.LayerWeights),
		gpuKVCaches:  make(map[int]*bindings.KVCache),
		kvCaches:     inference.NewKVCacheManager(),
		peers:        make([]peer.AddrInfo, 0),
		startTime:    time.Now(),
		startLayerID: -1,
		endLayerID:   -1,
		chunkBuffers: make(map[int]*chunkBuffer),
	}, nil
}

// Start starts the worker and begins listening for requests
func (w *Worker) Start() error {
	log.Println("Starting NeuroGrid worker (performer)...")
	log.Printf("Worker peer ID: %s", w.host.ID())

	// Print multiaddrs for coordinator to connect
	for _, addr := range w.host.Addrs() {
		fullAddr := fmt.Sprintf("%s/p2p/%s", addr.String(), w.host.ID())
		log.Printf("Listening on: %s", fullAddr)
	}

	// Initialize GPU
	if err := w.initializeGPU(); err != nil {
		log.Printf("Warning: GPU initialization failed: %v", err)
		log.Printf("Running in CPU mode")
	}

	// Create P2P protocol and register handlers
	w.protocol = p2p.NewProtocol(w.host)
	w.protocol.OnActivationReceived(w.handleActivation)
	w.protocol.OnWeightsReceived(w.handleWeights)
	w.protocol.OnLayerRequestReceived(w.handleLayerRequest)

	// Connect to bootstrap peers (coordinator)
	if len(w.config.BootstrapPeers) > 0 {
		log.Printf("Connecting to %d bootstrap peers (coordinators)...", len(w.config.BootstrapPeers))
		for _, peerInfo := range w.config.BootstrapPeers {
			log.Printf("Connecting to coordinator: %s", peerInfo.ID)
			ctx, cancel := context.WithTimeout(w.ctx, 30*time.Second)
			err := w.host.Connect(ctx, peerInfo)
			cancel()
			if err != nil {
				log.Printf("Warning: Failed to connect to coordinator %s: %v", peerInfo.ID, err)
			} else {
				log.Printf("Connected to coordinator: %s", peerInfo.ID)
				w.peersMu.Lock()
				w.peers = append(w.peers, peerInfo)
				w.peersMu.Unlock()

				// Send layer status to coordinator after connecting
				go w.sendLayerStatus(peerInfo.ID)
			}
		}
	}

	// Setup mDNS discovery (for LAN connections)
	notifee := &discoveryNotifee{worker: w}
	mdnsService := mdns.NewMdnsService(w.host, ServiceTag, notifee)
	if err := mdnsService.Start(); err != nil {
		return fmt.Errorf("failed to start mDNS: %w", err)
	}

	// If model path provided, load weights locally
	if w.config.ModelPath != "" {
		if err := w.loadLocalWeights(); err != nil {
			log.Printf("Warning: Failed to load local weights: %v", err)
			log.Printf("Worker will wait for weights from coordinator")
		}
	} else {
		log.Printf("No model path specified, waiting for weights from coordinator...")
	}

	log.Printf("Worker started successfully")
	log.Printf("  GPU: %d", w.config.GPUID)
	log.Printf("  Port: %d", w.config.Port)
	log.Printf("  Waiting for activation requests...")

	return nil
}

// initializeGPU sets up the GPU device
func (w *Worker) initializeGPU() error {
	if err := bindings.SetDevice(w.config.GPUID); err != nil {
		return fmt.Errorf("failed to set GPU device %d: %w", w.config.GPUID, err)
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
		w.config.GPUID, deviceInfo.Name,
		float64(totalVRAM)/(1024*1024*1024),
		float64(usedVRAM)/(1024*1024*1024),
		float64(totalVRAM-usedVRAM)/(1024*1024*1024))

	return nil
}

// loadLocalWeights loads weights from local model path
func (w *Worker) loadLocalWeights() error {
	log.Printf("Loading weights from %s...", w.config.ModelPath)

	loader, err := model.NewWeightLoader(w.config.ModelPath)
	if err != nil {
		return fmt.Errorf("failed to create weight loader: %w", err)
	}
	defer loader.Close()

	// Get model config
	modelConfig := getModelConfig(w.config.ModelName)
	w.modelConfig = &types.LlamaConfig{
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

	// Load all layers (in distributed mode, we only load assigned layers)
	numLayers := loader.CountLayers()
	if numLayers == 0 {
		numLayers = w.modelConfig.NumLayers
	}

	log.Printf("Model has %d layers", numLayers)

	// For now, load all layers - coordinator will tell us which ones to keep
	for layerID := 0; layerID < numLayers; layerID++ {
		weights, err := loader.LoadLayerWeights(layerID)
		if err != nil {
			log.Printf("Warning: Failed to load layer %d: %v", layerID, err)
			continue
		}
		w.layerWeights[layerID] = weights

		// Upload weights to GPU
		gpuWeights, err := bindings.CreateLayerWeightsFromHost(
			weights.QWeight,
			weights.KWeight,
			weights.VWeight,
			weights.OWeight,
			weights.GateWeight,
			weights.UpWeight,
			weights.DownWeight,
			weights.AttnNorm,
			weights.FFNNorm,
			w.modelConfig,
		)
		if err != nil {
			log.Printf("Warning: Failed to upload layer %d to GPU: %v", layerID, err)
			continue
		}
		w.gpuWeights[layerID] = gpuWeights

		// Initialize KV cache for this layer
		cache := inference.NewDistributedKVCache(
			inference.KVCacheConfig{
				LayerID:    layerID,
				NumKVHeads: w.modelConfig.NumKVHeads,
				HeadDim:    w.modelConfig.HeadDim,
				MaxSeqLen:  w.modelConfig.MaxSeqLen,
			},
			w.host.ID().String(),
			w.config.GPUID,
			true, // local
		)
		w.kvCaches.RegisterCache(cache)

		// Update layer range
		if w.startLayerID == -1 || layerID < w.startLayerID {
			w.startLayerID = layerID
		}
		if layerID > w.endLayerID {
			w.endLayerID = layerID
		}

		log.Printf("Loaded layer %d to GPU", layerID)
	}

	w.weightsMu.Lock()
	w.weightsReady = true
	w.weightsMu.Unlock()

	log.Printf("Loaded %d layers from local storage to GPU", len(w.gpuWeights))
	return nil
}

// handleActivation processes incoming activation requests from coordinator
func (w *Worker) handleActivation(msg *p2p.TensorMessage) {
	w.reqCounter++
	log.Printf("[%d] Activation request: layer=%d, seqID=%d, reqID=%d, from=%s",
		w.reqCounter, msg.LayerID, msg.SeqID, msg.RequestID, msg.From)

	// Check if we have weights ready
	w.weightsMu.RLock()
	ready := w.weightsReady
	w.weightsMu.RUnlock()

	if !ready {
		log.Printf("Warning: Weights not ready, returning passthrough")
		w.sendResponse(msg.From, msg.LayerID, msg.SeqID, msg.RequestID, msg.Data)
		return
	}

	// Execute the layer
	output, err := w.executeLayer(msg.LayerID, msg.Data, int(msg.SeqID))
	if err != nil {
		log.Printf("Error executing layer %d: %v", msg.LayerID, err)
		// Send back input as fallback
		w.sendResponse(msg.From, msg.LayerID, msg.SeqID, msg.RequestID, msg.Data)
		return
	}

	// Send response back
	w.sendResponse(msg.From, msg.LayerID, msg.SeqID, msg.RequestID, output)
	log.Printf("[%d] Completed layer %d: %d -> %d bytes", w.reqCounter, msg.LayerID, len(msg.Data), len(output))
}

// sendResponse sends activation response back to coordinator
func (w *Worker) sendResponse(peerID peer.ID, layerID int, seqID, requestID uint64, data []byte) {
	ctx, cancel := context.WithTimeout(w.ctx, 30*time.Second)
	defer cancel()

	err := w.protocol.SendResponse(ctx, peerID, layerID, seqID, requestID, data)
	if err != nil {
		log.Printf("Error sending response: %v", err)
	}
}

// executeLayer runs a forward pass through a transformer layer
func (w *Worker) executeLayer(layerID int, hidden []byte, position int) ([]byte, error) {
	// Lock this goroutine to its current OS thread to ensure CUDA context consistency.
	// CUDA contexts are per-thread, so without this lock, the goroutine could be
	// rescheduled to a different thread between CUDA calls, causing context errors.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	// Ensure CUDA context is set for this thread (P2P handlers may run on different goroutines)
	if err := bindings.SetDevice(w.config.GPUID); err != nil {
		return nil, fmt.Errorf("failed to set CUDA device: %w", err)
	}

	// Get GPU weights
	gpuWeights, ok := w.gpuWeights[layerID]
	if !ok {
		return nil, fmt.Errorf("GPU weights not available for layer %d", layerID)
	}

	// Get or create KV cache for this layer
	kvCache, ok := w.gpuKVCaches[layerID]
	if !ok {
		// Create KV cache for this layer
		var err error
		kvCache, err = bindings.NewKVCache(
			1, // batch size
			w.modelConfig.NumKVHeads,
			w.modelConfig.HeadDim,
			w.modelConfig.MaxSeqLen,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to create KV cache: %w", err)
		}
		w.gpuKVCaches[layerID] = kvCache
	}

	// Convert hidden bytes to tensor format
	// Input shape: [batch, seq_len, hidden_size]
	batchSize := 1
	seqLen := len(hidden) / (w.modelConfig.HiddenSize * 2) // FP16 = 2 bytes per element
	if seqLen == 0 {
		seqLen = 1
	}

	// Create input tensor
	inputTensor := &types.Tensor{
		Shape:  []int{batchSize, seqLen, w.modelConfig.HiddenSize},
		Dtype:  types.DtypeFP16,
		Device: w.config.GPUID,
	}

	// Allocate and upload input to GPU
	if err := bindings.AllocateTensor(inputTensor); err != nil {
		return nil, fmt.Errorf("failed to allocate input tensor: %w", err)
	}
	defer bindings.FreeTensor(inputTensor)

	// Copy input bytes to GPU
	if err := bindings.CopyToDeviceRaw(inputTensor.Data, unsafe.Pointer(&hidden[0]), uint64(len(hidden))); err != nil {
		return nil, fmt.Errorf("failed to copy input to GPU: %w", err)
	}

	// Create output tensor (same shape as input)
	outputTensor := &types.Tensor{
		Shape:  inputTensor.Shape,
		Dtype:  types.DtypeFP16,
		Device: w.config.GPUID,
	}
	if err := bindings.AllocateTensor(outputTensor); err != nil {
		return nil, fmt.Errorf("failed to allocate output tensor: %w", err)
	}
	defer bindings.FreeTensor(outputTensor)

	// Create positions array
	positions := make([]int32, seqLen)
	for i := range positions {
		positions[i] = int32(position + i)
	}

	// Execute layer forward pass
	if err := bindings.LayerForward(outputTensor, inputTensor, gpuWeights, kvCache, positions, w.modelConfig); err != nil {
		return nil, fmt.Errorf("layer forward failed: %w", err)
	}

	// Copy output back to CPU
	output := make([]byte, len(hidden))
	if err := bindings.CopyFromDeviceRaw(unsafe.Pointer(&output[0]), outputTensor.Data, uint64(len(output))); err != nil {
		return nil, fmt.Errorf("failed to copy output from GPU: %w", err)
	}

	return output, nil
}

// handleWeights receives layer weights from coordinator
func (w *Worker) handleWeights(layerID int, chunkIndex int, totalChunks int, data []byte) {
	log.Printf("Received weight chunk %d/%d for layer %d (%d bytes)",
		chunkIndex+1, totalChunks, layerID, len(data))

	// Accumulate chunks before deserializing
	w.chunkMu.Lock()

	// Initialize buffer for this layer if needed
	buf, exists := w.chunkBuffers[layerID]
	if !exists {
		buf = &chunkBuffer{
			chunks:      make([][]byte, totalChunks),
			totalChunks: totalChunks,
			received:    0,
		}
		w.chunkBuffers[layerID] = buf
	}

	// Store chunk data (make a copy since data may be reused)
	buf.chunks[chunkIndex] = make([]byte, len(data))
	copy(buf.chunks[chunkIndex], data)
	buf.received++

	// Check if all chunks received
	allReceived := buf.received == buf.totalChunks
	w.chunkMu.Unlock()

	if !allReceived {
		log.Printf("Waiting for more chunks for layer %d (%d/%d received)",
			layerID, buf.received, buf.totalChunks)
		return
	}

	// All chunks received - concatenate and deserialize
	log.Printf("All chunks received for layer %d, concatenating %d chunks...", layerID, totalChunks)

	// Calculate total size and concatenate
	w.chunkMu.Lock()
	buf = w.chunkBuffers[layerID]
	totalSize := 0
	for _, chunk := range buf.chunks {
		totalSize += len(chunk)
	}

	fullData := make([]byte, 0, totalSize)
	for _, chunk := range buf.chunks {
		fullData = append(fullData, chunk...)
	}

	// Clean up buffer
	delete(w.chunkBuffers, layerID)
	w.chunkMu.Unlock()

	log.Printf("Layer %d: concatenated %d bytes from %d chunks", layerID, len(fullData), totalChunks)

	// Deserialize weights from full data
	weights, err := model.DeserializeLayerWeights(layerID, fullData)
	if err != nil {
		log.Printf("Error deserializing weights for layer %d: %v", layerID, err)
		return
	}

	w.layerWeights[layerID] = weights
	log.Printf("Loaded weights for layer %d from coordinator (CPU)", layerID)

	// Upload weights to GPU
	if w.modelConfig != nil {
		gpuWeights, err := bindings.CreateLayerWeightsFromHost(
			weights.QWeight,
			weights.KWeight,
			weights.VWeight,
			weights.OWeight,
			weights.GateWeight,
			weights.UpWeight,
			weights.DownWeight,
			weights.AttnNorm,
			weights.FFNNorm,
			w.modelConfig,
		)
		if err != nil {
			log.Printf("Error uploading weights for layer %d to GPU: %v", layerID, err)
			// Continue without GPU weights - will fail at execution time
		} else {
			w.gpuWeights[layerID] = gpuWeights
			log.Printf("Loaded weights for layer %d to GPU", layerID)
		}
	}

	// Update layer range
	if w.startLayerID == -1 || layerID < w.startLayerID {
		w.startLayerID = layerID
	}
	if layerID > w.endLayerID {
		w.endLayerID = layerID
	}

	// Initialize KV cache for this layer if we have config
	if w.modelConfig != nil {
		cache := inference.NewDistributedKVCache(
			inference.KVCacheConfig{
				LayerID:    layerID,
				NumKVHeads: w.modelConfig.NumKVHeads,
				HeadDim:    w.modelConfig.HeadDim,
				MaxSeqLen:  w.modelConfig.MaxSeqLen,
			},
			w.host.ID().String(),
			w.config.GPUID,
			true, // local
		)
		w.kvCaches.RegisterCache(cache)
	}

	// Mark weights ready once we have at least one layer
	w.weightsMu.Lock()
	w.weightsReady = true
	w.weightsMu.Unlock()

	// Send acknowledgment
	ctx, cancel := context.WithTimeout(w.ctx, 10*time.Second)
	defer cancel()

	w.peersMu.RLock()
	for _, p := range w.peers {
		w.protocol.SendWeightsAck(ctx, p.ID, layerID)
	}
	w.peersMu.RUnlock()
}

// sendLayerStatus sends the list of locally loaded layers to the coordinator
func (w *Worker) sendLayerStatus(coordinatorID peer.ID) {
	// Get list of loaded layer IDs
	loadedLayers := w.getLoadedLayerIDs()

	log.Printf("Sending layer status to coordinator: %d layers loaded locally", len(loadedLayers))

	ctx, cancel := context.WithTimeout(w.ctx, 10*time.Second)
	defer cancel()

	if err := w.protocol.SendLayerStatus(ctx, coordinatorID, loadedLayers); err != nil {
		log.Printf("Warning: Failed to send layer status to coordinator: %v", err)
	}
}

// getLoadedLayerIDs returns a list of layer IDs that are loaded in GPU memory
func (w *Worker) getLoadedLayerIDs() []int {
	w.weightsMu.RLock()
	defer w.weightsMu.RUnlock()

	layers := make([]int, 0, len(w.gpuWeights))
	for layerID := range w.gpuWeights {
		layers = append(layers, layerID)
	}
	return layers
}

// handleLayerRequest handles layer request messages from the coordinator
func (w *Worker) handleLayerRequest(coordinatorID peer.ID, requestedLayers []int) {
	// If empty request, just send back our status (query mode)
	if len(requestedLayers) == 0 {
		log.Printf("Received layer status query from coordinator")
		w.sendLayerStatus(coordinatorID)
		return
	}

	log.Printf("Received request to load %d layers from coordinator", len(requestedLayers))

	// Check which layers we need to load vs already have
	loadedLayers := w.getLoadedLayerIDs()
	loadedSet := make(map[int]bool)
	for _, id := range loadedLayers {
		loadedSet[id] = true
	}

	needToLoad := make([]int, 0)
	for _, layerID := range requestedLayers {
		if !loadedSet[layerID] {
			needToLoad = append(needToLoad, layerID)
		}
	}

	if len(needToLoad) == 0 {
		log.Printf("All requested layers already loaded locally")
		w.sendLayerStatus(coordinatorID)
		return
	}

	// Try to load missing layers from local model if available
	if w.config.ModelPath != "" {
		log.Printf("Loading %d layers from local storage: %v", len(needToLoad), needToLoad)
		w.loadSpecificLayers(needToLoad)
	} else {
		log.Printf("No local model path, waiting for %d layers from coordinator", len(needToLoad))
	}

	// Send updated status
	w.sendLayerStatus(coordinatorID)
}

// loadSpecificLayers loads specific layers from local model storage
func (w *Worker) loadSpecificLayers(layerIDs []int) {
	loader, err := model.NewWeightLoader(w.config.ModelPath)
	if err != nil {
		log.Printf("Warning: Failed to create weight loader: %v", err)
		return
	}
	defer loader.Close()

	for _, layerID := range layerIDs {
		// Skip if already loaded
		w.weightsMu.RLock()
		_, exists := w.gpuWeights[layerID]
		w.weightsMu.RUnlock()
		if exists {
			continue
		}

		weights, err := loader.LoadLayerWeights(layerID)
		if err != nil {
			log.Printf("Warning: Failed to load layer %d: %v", layerID, err)
			continue
		}

		w.weightsMu.Lock()
		w.layerWeights[layerID] = weights
		w.weightsMu.Unlock()

		// Upload weights to GPU
		if w.modelConfig != nil {
			gpuWeights, err := bindings.CreateLayerWeightsFromHost(
				weights.QWeight,
				weights.KWeight,
				weights.VWeight,
				weights.OWeight,
				weights.GateWeight,
				weights.UpWeight,
				weights.DownWeight,
				weights.AttnNorm,
				weights.FFNNorm,
				w.modelConfig,
			)
			if err != nil {
				log.Printf("Warning: Failed to upload layer %d to GPU: %v", layerID, err)
				continue
			}
			w.weightsMu.Lock()
			w.gpuWeights[layerID] = gpuWeights
			w.weightsMu.Unlock()

			log.Printf("Loaded layer %d from local storage to GPU", layerID)
		}

		// Update layer range
		w.weightsMu.Lock()
		if w.startLayerID == -1 || layerID < w.startLayerID {
			w.startLayerID = layerID
		}
		if layerID > w.endLayerID {
			w.endLayerID = layerID
		}
		w.weightsReady = true
		w.weightsMu.Unlock()
	}
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
		"layers_loaded": len(w.layerWeights),
		"layer_range":   fmt.Sprintf("%d-%d", w.startLayerID, w.endLayerID),
	}
}

// discoveryNotifee handles peer discovery notifications
type discoveryNotifee struct {
	worker *Worker
}

// HandlePeerFound is called when a peer is discovered
func (n *discoveryNotifee) HandlePeerFound(pi peer.AddrInfo) {
	log.Printf("Discovered peer: %s", pi.ID)

	// Connect to the peer
	if err := n.worker.host.Connect(n.worker.ctx, pi); err != nil {
		log.Printf("Failed to connect to peer %s: %v", pi.ID, err)
		return
	}

	n.worker.peersMu.Lock()
	n.worker.peers = append(n.worker.peers, pi)
	n.worker.peersMu.Unlock()

	log.Printf("Connected to peer: %s", pi.ID)
}

// getModelConfig returns configuration for the specified model
func getModelConfig(modelName string) *types.LlamaConfig {
	switch modelName {
	case "llama-13b", "llama-2-13b":
		return types.Llama13BConfig()
	case "llama-70b", "llama3-70b", "llama-3.3-70b":
		return types.Llama70BConfig()
	case "tinyllama", "tinyllama-1.1b":
		return types.TinyLlamaConfig()
	default:
		// Default to 7B config
		return types.Llama7BConfig()
	}
}

func main() {
	// Parse command line flags
	port := flag.Int("port", 9000, "libp2p listen port")
	gpuID := flag.Int("gpu", 0, "GPU device ID")
	modelPath := flag.String("model", "", "Path to model weights (optional - coordinator can send weights)")
	modelName := flag.String("model-name", "llama-7b", "Model name (llama-7b, llama-13b, llama-70b)")
	logLevel := flag.String("log-level", "info", "Log level (debug, info, warn, error)")
	bootstrapStr := flag.String("bootstrap", "", "Bootstrap peer multiaddr (coordinator address)")
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
		}
	}

	// Create worker config
	config := WorkerConfig{
		Port:           *port,
		GPUID:          *gpuID,
		ModelPath:      *modelPath,
		ModelName:      *modelName,
		LogLevel:       *logLevel,
		BootstrapPeers: bootstrapPeers,
	}

	log.Println("================================================")
	log.Println("    NeuroGrid Worker (Performer Node)           ")
	log.Println("================================================")
	log.Printf("GPU: %d", config.GPUID)
	log.Printf("Port: %d", config.Port)
	log.Printf("Model: %s", config.ModelName)
	if config.ModelPath != "" {
		log.Printf("Model Path: %s", config.ModelPath)
	}
	if len(config.BootstrapPeers) > 0 {
		log.Printf("Bootstrap Peers (coordinators): %d", len(config.BootstrapPeers))
	}

	// Create and start worker
	worker, err := NewWorker(config)
	if err != nil {
		log.Fatalf("Failed to create worker: %v", err)
	}

	if err := worker.Start(); err != nil {
		log.Fatalf("Failed to start worker: %v", err)
	}

	log.Println("================================================")
	log.Printf("Worker ready to receive requests")
	log.Println("================================================")

	// Wait for shutdown signal
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)

	sig := <-sigChan
	log.Printf("Received signal: %v", sig)

	if err := worker.Shutdown(); err != nil {
		log.Fatalf("Shutdown failed: %v", err)
	}
}
