// Package scheduler provides VRAM-aware layer scheduling for distributed inference.
package scheduler

import (
	"fmt"
	"sort"
	"sync"
)

// ModelConfig contains configuration for memory estimation.
type ModelConfig struct {
	HiddenSize       int64   // Hidden dimension (e.g., 4096 for 7B)
	IntermediateSize int64   // FFN intermediate size (e.g., 11008 for 7B)
	NumLayers        int     // Number of transformer layers (e.g., 32 for 7B)
	NumHeads         int     // Number of attention heads (e.g., 32 for 7B)
	NumKVHeads       int     // Number of KV heads (for GQA)
	HeadDim          int     // Dimension per head (e.g., 128)
	MaxSeqLen        int     // Maximum sequence length for KV cache
	VocabSize        int64   // Vocabulary size for embedding
	RMSNormEps       float32 // RMSNorm epsilon (typically 1e-6)
}

// DefaultLlama7BConfig returns the default configuration for Llama 7B.
func DefaultLlama7BConfig() ModelConfig {
	return ModelConfig{
		HiddenSize:       4096,
		IntermediateSize: 11008,
		NumLayers:        32,
		NumHeads:         32,
		NumKVHeads:       32,
		HeadDim:          128,
		MaxSeqLen:        2048,
		VocabSize:        32000,
		RMSNormEps:       1e-6,
	}
}

// DefaultLlama13BConfig returns the default configuration for Llama 13B.
func DefaultLlama13BConfig() ModelConfig {
	return ModelConfig{
		HiddenSize:       5120,
		IntermediateSize: 13824,
		NumLayers:        40,
		NumHeads:         40,
		NumKVHeads:       40,
		HeadDim:          128,
		MaxSeqLen:        2048,
		VocabSize:        32000,
		RMSNormEps:       1e-6,
	}
}

// DefaultTinyLlamaConfig returns the default configuration for TinyLlama 1.1B.
func DefaultTinyLlamaConfig() ModelConfig {
	return ModelConfig{
		HiddenSize:       2048,
		IntermediateSize: 5632,
		NumLayers:        22,
		NumHeads:         32, // 32 attention heads
		NumKVHeads:       4,  // GQA: 4 KV heads
		HeadDim:          64, // 2048 / 32 heads
		MaxSeqLen:        2048,
		VocabSize:        32000,
		RMSNormEps:       1e-6,
	}
}

// DefaultLlama3_70BConfig returns the default configuration for Llama 3.3 70B.
func DefaultLlama3_70BConfig() ModelConfig {
	return ModelConfig{
		HiddenSize:       8192,
		IntermediateSize: 28672,
		NumLayers:        80,
		NumHeads:         64,
		NumKVHeads:       8,   // GQA: 8 KV heads
		HeadDim:          128, // 8192 / 64 heads
		MaxSeqLen:        8192,
		VocabSize:        128256,
		RMSNormEps:       1e-5,
	}
}

// LayerAssignment represents a layer-to-peer assignment.
type LayerAssignment struct {
	LayerID      int
	PeerID       string
	MemoryNeeded uint64
}

// Scheduler manages layer assignment across peers.
type Scheduler struct {
	config  ModelConfig
	tracker *VRAMTracker
	peers   []string // Ordered list of peer IDs
	mu      sync.RWMutex
}

// NewScheduler creates a new scheduler with the given configuration.
func NewScheduler(config ModelConfig) *Scheduler {
	return &Scheduler{
		config:  config,
		tracker: NewVRAMTracker(),
		peers:   make([]string, 0),
	}
}

// RegisterPeer adds a peer to the scheduler.
func (s *Scheduler) RegisterPeer(peerID string, totalVRAM, usedVRAM uint64) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if err := s.tracker.RegisterPeer(peerID, totalVRAM, usedVRAM); err != nil {
		return err
	}

	s.peers = append(s.peers, peerID)
	return nil
}

// UnregisterPeer removes a peer from the scheduler.
func (s *Scheduler) UnregisterPeer(peerID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if err := s.tracker.UnregisterPeer(peerID); err != nil {
		return err
	}

	// Remove from peer list
	for i, p := range s.peers {
		if p == peerID {
			s.peers = append(s.peers[:i], s.peers[i+1:]...)
			break
		}
	}

	return nil
}

// EstimateLayerMemory calculates the estimated memory for a single transformer layer.
func (s *Scheduler) EstimateLayerMemory() uint64 {
	cfg := s.config

	// Weights (INT8): Q, K, V, O projections
	attnWeights := uint64(cfg.HiddenSize * cfg.HiddenSize * 4) // 4 projections

	// FFN weights (INT8): gate, up, down projections
	ffnWeights := uint64(cfg.HiddenSize * cfg.IntermediateSize * 3) // 3 projections

	// Scales (FP32): one per column for each weight matrix
	numScaleParams := cfg.HiddenSize*4 + cfg.IntermediateSize*3
	scales := uint64(numScaleParams * 4) // 4 bytes per float32

	// Norms (FP16): RMSNorm weights for attention and FFN
	norms := uint64(cfg.HiddenSize * 2 * 2) // 2 norms, 2 bytes per fp16

	// KV cache (FP16): 2 * seq_len * num_kv_heads * head_dim * 2 bytes
	kvCache := uint64(cfg.MaxSeqLen * cfg.NumKVHeads * cfg.HeadDim * 2 * 2) // K and V

	// Activation buffer (FP16): working memory for forward pass
	activations := uint64(4096 * cfg.HiddenSize * 2) // batch * hidden * 2 bytes

	total := attnWeights + ffnWeights + scales + norms + kvCache + activations

	// 20% overhead for alignment, fragmentation, and cuBLAS workspace
	return uint64(float64(total) * 1.2)
}

// EstimateEmbeddingMemory calculates memory for embedding/output layers.
func (s *Scheduler) EstimateEmbeddingMemory() uint64 {
	cfg := s.config

	// Token embedding (FP16): vocab_size * hidden_size * 2 bytes
	embedding := uint64(cfg.VocabSize * cfg.HiddenSize * 2)

	// Output head (FP16): shares weights with embedding
	// Only need activation buffer
	outputBuffer := uint64(cfg.VocabSize * 4) // logits in FP32

	// 10% overhead
	return uint64(float64(embedding+outputBuffer) * 1.1)
}

// TotalModelMemory returns the total estimated memory for the model.
func (s *Scheduler) TotalModelMemory() uint64 {
	layerMem := s.EstimateLayerMemory()
	embedMem := s.EstimateEmbeddingMemory()

	return embedMem + uint64(s.config.NumLayers)*layerMem
}

// ComputeAssignments calculates layer-to-peer assignments.
// Uses a greedy algorithm that assigns layers to peers with most available VRAM.
func (s *Scheduler) ComputeAssignments() ([]LayerAssignment, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if len(s.peers) == 0 {
		return nil, fmt.Errorf("no peers registered")
	}

	layerMemory := s.EstimateLayerMemory()
	embedMemory := s.EstimateEmbeddingMemory()

	// Get peer info sorted by available VRAM (descending)
	peerInfos := s.tracker.GetAllPeers()
	sort.Slice(peerInfos, func(i, j int) bool {
		return peerInfos[i].Available() > peerInfos[j].Available()
	})

	assignments := make([]LayerAssignment, 0, s.config.NumLayers+2)

	// Create a local copy for tracking during assignment
	available := make(map[string]uint64)
	for _, p := range peerInfos {
		available[p.PeerID] = p.Available()
	}

	// Helper to find best peer for given memory requirement
	findBestPeer := func(needed uint64) (string, error) {
		// Sort peers by available (descending) each time to account for updates
		var candidates []string
		for peerID, avail := range available {
			if avail >= needed {
				candidates = append(candidates, peerID)
			}
		}

		if len(candidates) == 0 {
			return "", fmt.Errorf("no peer has %d bytes available", needed)
		}

		// Pick the one with most available
		sort.Slice(candidates, func(i, j int) bool {
			return available[candidates[i]] > available[candidates[j]]
		})

		return candidates[0], nil
	}

	// Assign embedding layer (layer -1)
	embedPeer, err := findBestPeer(embedMemory)
	if err != nil {
		return nil, fmt.Errorf("cannot assign embedding layer: %w", err)
	}
	assignments = append(assignments, LayerAssignment{
		LayerID:      -1, // Embedding layer
		PeerID:       embedPeer,
		MemoryNeeded: embedMemory,
	})
	available[embedPeer] -= embedMemory

	// Assign transformer layers
	for i := 0; i < s.config.NumLayers; i++ {
		peer, err := findBestPeer(layerMemory)
		if err != nil {
			return nil, fmt.Errorf("cannot assign layer %d: %w", i, err)
		}
		assignments = append(assignments, LayerAssignment{
			LayerID:      i,
			PeerID:       peer,
			MemoryNeeded: layerMemory,
		})
		available[peer] -= layerMemory
	}

	// Assign output layer (layer NumLayers, shares embedding weights)
	// Just need logits buffer
	outputMemory := uint64(s.config.VocabSize * 4) // logits in FP32
	outputPeer, err := findBestPeer(outputMemory)
	if err != nil {
		return nil, fmt.Errorf("cannot assign output layer: %w", err)
	}
	assignments = append(assignments, LayerAssignment{
		LayerID:      s.config.NumLayers, // Output layer
		PeerID:       outputPeer,
		MemoryNeeded: outputMemory,
	})

	return assignments, nil
}

// GetPeerLayers returns which layers are assigned to a specific peer.
func (s *Scheduler) GetPeerLayers(assignments []LayerAssignment, peerID string) []int {
	var layers []int
	for _, a := range assignments {
		if a.PeerID == peerID {
			layers = append(layers, a.LayerID)
		}
	}
	return layers
}

// ValidateAssignments checks if assignments are feasible given current VRAM.
func (s *Scheduler) ValidateAssignments(assignments []LayerAssignment) error {
	// Group by peer
	peerMemory := make(map[string]uint64)
	for _, a := range assignments {
		peerMemory[a.PeerID] += a.MemoryNeeded
	}

	// Check each peer
	for peerID, needed := range peerMemory {
		info, err := s.tracker.GetPeerInfo(peerID)
		if err != nil {
			return err
		}
		if needed > info.Available() {
			return fmt.Errorf("peer %s needs %d bytes but only has %d available",
				peerID, needed, info.Available())
		}
	}

	return nil
}

// VRAMTracker returns the underlying VRAM tracker.
func (s *Scheduler) VRAMTracker() *VRAMTracker {
	return s.tracker
}

// Config returns the model configuration.
func (s *Scheduler) Config() ModelConfig {
	return s.config
}
