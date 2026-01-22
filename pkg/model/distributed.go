// Package model provides model loading and weight management for LLM inference.
package model

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"sync"
	"sync/atomic"

	"github.com/neurogrid/engine/pkg/scheduler"
	"github.com/neurogrid/engine/pkg/transport"
	"github.com/neurogrid/engine/pkg/types"
)

// DistributedModel manages model weights distributed across multiple peers.
type DistributedModel struct {
	config      *types.LlamaConfig
	scheduler   *scheduler.Scheduler
	loader      *WeightLoader
	router      *transport.TransportRouter
	localPeerID string

	// Embeddings and output head (on coordinator)
	embeddings []byte
	lmHead     []byte

	// Layer loading status
	layerStatus map[int]LoadStatus
	statusMu    sync.RWMutex

	// Progress tracking
	totalLayers  int
	loadedLayers int32
	progressChan chan LoadProgress
}

// LoadStatus represents the loading status of a layer.
type LoadStatus int

const (
	LoadStatusPending LoadStatus = iota
	LoadStatusLoading
	LoadStatusLoaded
	LoadStatusFailed
)

// LoadProgress represents loading progress for a layer.
type LoadProgress struct {
	LayerID   int
	PeerID    string
	Status    LoadStatus
	BytesSize int64
	Error     error
}

// DistributedModelConfig holds configuration for distributed model loading.
type DistributedModelConfig struct {
	ModelConfig *types.LlamaConfig
	ModelPath   string
	LocalPeerID string
}

// NewDistributedModel creates a new distributed model manager.
func NewDistributedModel(config DistributedModelConfig) (*DistributedModel, error) {
	loader, err := NewWeightLoader(config.ModelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to create weight loader: %w", err)
	}

	numLayers := loader.CountLayers()
	if numLayers == 0 {
		numLayers = config.ModelConfig.NumLayers
	}

	return &DistributedModel{
		config:       config.ModelConfig,
		loader:       loader,
		localPeerID:  config.LocalPeerID,
		layerStatus:  make(map[int]LoadStatus),
		totalLayers:  numLayers,
		progressChan: make(chan LoadProgress, 100),
	}, nil
}

// SetScheduler sets the scheduler for layer assignments.
func (m *DistributedModel) SetScheduler(s *scheduler.Scheduler) {
	m.scheduler = s
}

// SetRouter sets the transport router for remote layer transfer.
func (m *DistributedModel) SetRouter(r *transport.TransportRouter) {
	m.router = r
}

// LoadToCluster loads all model weights to the cluster according to assignments.
func (m *DistributedModel) LoadToCluster(ctx context.Context) error {
	if m.scheduler == nil {
		return fmt.Errorf("scheduler not set")
	}

	assignments, err := m.scheduler.ComputeAssignments()
	if err != nil {
		return fmt.Errorf("failed to compute assignments: %w", err)
	}

	// Initialize status for all layers
	m.statusMu.Lock()
	for _, a := range assignments {
		m.layerStatus[a.LayerID] = LoadStatusPending
	}
	m.statusMu.Unlock()

	// Load embeddings on coordinator
	if err := m.loadEmbeddings(ctx); err != nil {
		return fmt.Errorf("failed to load embeddings: %w", err)
	}

	// Load lm_head on coordinator
	if err := m.loadLMHead(ctx); err != nil {
		return fmt.Errorf("failed to load lm_head: %w", err)
	}

	// Load layers in parallel
	errChan := make(chan error, len(assignments))
	var wg sync.WaitGroup

	for _, assign := range assignments {
		// Skip embedding and output layers
		if assign.LayerID < 0 || assign.LayerID >= m.totalLayers {
			continue
		}

		wg.Add(1)
		go func(a scheduler.LayerAssignment) {
			defer wg.Done()

			m.statusMu.Lock()
			m.layerStatus[a.LayerID] = LoadStatusLoading
			m.statusMu.Unlock()

			var err error
			if a.PeerID == m.localPeerID {
				err = m.loadLocalLayer(ctx, a.LayerID)
			} else {
				err = m.sendLayerToPeer(ctx, a)
			}

			m.statusMu.Lock()
			if err != nil {
				m.layerStatus[a.LayerID] = LoadStatusFailed
				errChan <- fmt.Errorf("layer %d: %w", a.LayerID, err)
			} else {
				m.layerStatus[a.LayerID] = LoadStatusLoaded
				atomic.AddInt32(&m.loadedLayers, 1)
			}
			m.statusMu.Unlock()

			// Send progress update
			select {
			case m.progressChan <- LoadProgress{
				LayerID: a.LayerID,
				PeerID:  a.PeerID,
				Status:  m.layerStatus[a.LayerID],
				Error:   err,
			}:
			default:
			}
		}(assign)
	}

	// Wait for all goroutines
	wg.Wait()
	close(errChan)

	// Collect errors
	var errors []error
	for err := range errChan {
		errors = append(errors, err)
	}

	if len(errors) > 0 {
		return fmt.Errorf("loading failed: %v", errors)
	}

	return nil
}

// loadEmbeddings loads the embedding matrix.
func (m *DistributedModel) loadEmbeddings(ctx context.Context) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	data, _, err := m.loader.LoadEmbeddings()
	if err != nil {
		return err
	}

	m.embeddings = data
	return nil
}

// loadLMHead loads the output projection matrix.
func (m *DistributedModel) loadLMHead(ctx context.Context) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	data, _, err := m.loader.LoadLMHead()
	if err != nil {
		return err
	}

	m.lmHead = data
	return nil
}

// loadLocalLayer loads a layer to local GPU memory.
func (m *DistributedModel) loadLocalLayer(ctx context.Context, layerID int) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	_, err := m.loader.LoadLayerWeights(layerID)
	if err != nil {
		return err
	}

	// In a real implementation, we would upload to GPU here
	// For now, the weights are loaded into CPU memory

	return nil
}

// sendLayerToPeer sends layer weights to a remote peer.
func (m *DistributedModel) sendLayerToPeer(ctx context.Context, assign scheduler.LayerAssignment) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}

	// Load the layer weights first
	weights, err := m.loader.LoadLayerWeights(assign.LayerID)
	if err != nil {
		return fmt.Errorf("failed to load weights: %w", err)
	}

	// Serialize layer weights
	data, err := SerializeLayerWeights(weights)
	if err != nil {
		return fmt.Errorf("failed to serialize weights: %w", err)
	}

	// Send via router if available
	if m.router != nil {
		// Use layer ID as seqID for weight transfer
		err = m.router.RouteActivation(ctx, assign.LayerID, uint64(assign.LayerID), data)
		if err != nil {
			return fmt.Errorf("failed to send to peer %s: %w", assign.PeerID, err)
		}
	}

	return nil
}

// SerializeLayerWeights serializes all layer weights into a single byte buffer.
// Format: [count(uint32)][name_len(uint32)][name][data_len(uint32)][data]...
func SerializeLayerWeights(weights *LayerWeights) ([]byte, error) {
	var buf bytes.Buffer

	// Helper to write a tensor
	writeTensor := func(name string, data []byte) error {
		// Write name length and name
		nameBytes := []byte(name)
		if err := binary.Write(&buf, binary.LittleEndian, uint32(len(nameBytes))); err != nil {
			return err
		}
		buf.Write(nameBytes)

		// Write data length and data
		if err := binary.Write(&buf, binary.LittleEndian, uint32(len(data))); err != nil {
			return err
		}
		buf.Write(data)
		return nil
	}

	// Write all tensors
	tensors := map[string][]byte{
		"q_proj":    weights.QWeight,
		"k_proj":    weights.KWeight,
		"v_proj":    weights.VWeight,
		"o_proj":    weights.OWeight,
		"gate_proj": weights.GateWeight,
		"up_proj":   weights.UpWeight,
		"down_proj": weights.DownWeight,
		"attn_norm": weights.AttnNorm,
		"ffn_norm":  weights.FFNNorm,
	}

	// Write count
	if err := binary.Write(&buf, binary.LittleEndian, uint32(len(tensors))); err != nil {
		return nil, err
	}

	for name, data := range tensors {
		if err := writeTensor(name, data); err != nil {
			return nil, err
		}
	}

	return buf.Bytes(), nil
}

// DeserializeLayerWeights deserializes a byte buffer into layer weights.
func DeserializeLayerWeights(layerID int, data []byte) (*LayerWeights, error) {
	buf := bytes.NewReader(data)

	// Read count
	var count uint32
	if err := binary.Read(buf, binary.LittleEndian, &count); err != nil {
		return nil, err
	}

	weights := &LayerWeights{LayerID: layerID}
	tensors := make(map[string][]byte)

	for i := uint32(0); i < count; i++ {
		// Read name
		var nameLen uint32
		if err := binary.Read(buf, binary.LittleEndian, &nameLen); err != nil {
			return nil, err
		}
		nameBytes := make([]byte, nameLen)
		if _, err := buf.Read(nameBytes); err != nil {
			return nil, err
		}

		// Read data
		var dataLen uint32
		if err := binary.Read(buf, binary.LittleEndian, &dataLen); err != nil {
			return nil, err
		}
		tensorData := make([]byte, dataLen)
		if _, err := buf.Read(tensorData); err != nil {
			return nil, err
		}

		tensors[string(nameBytes)] = tensorData
	}

	// Map to struct fields
	weights.QWeight = tensors["q_proj"]
	weights.KWeight = tensors["k_proj"]
	weights.VWeight = tensors["v_proj"]
	weights.OWeight = tensors["o_proj"]
	weights.GateWeight = tensors["gate_proj"]
	weights.UpWeight = tensors["up_proj"]
	weights.DownWeight = tensors["down_proj"]
	weights.AttnNorm = tensors["attn_norm"]
	weights.FFNNorm = tensors["ffn_norm"]

	return weights, nil
}

// Embeddings returns the loaded embedding matrix.
func (m *DistributedModel) Embeddings() []byte {
	return m.embeddings
}

// LMHead returns the loaded LM head matrix.
func (m *DistributedModel) LMHead() []byte {
	return m.lmHead
}

// GetLayerStatus returns the loading status of a layer.
func (m *DistributedModel) GetLayerStatus(layerID int) LoadStatus {
	m.statusMu.RLock()
	defer m.statusMu.RUnlock()
	return m.layerStatus[layerID]
}

// AllLayersLoaded returns true if all transformer layers are loaded.
// This only checks layers 0 to totalLayers-1, excluding embedding (-1) and output (totalLayers).
func (m *DistributedModel) AllLayersLoaded() bool {
	m.statusMu.RLock()
	defer m.statusMu.RUnlock()

	// Check only transformer layers (0 to totalLayers-1)
	for layerID := 0; layerID < m.totalLayers; layerID++ {
		status, ok := m.layerStatus[layerID]
		if !ok || status != LoadStatusLoaded {
			return false
		}
	}
	return m.totalLayers > 0
}

// LoadedCount returns the number of loaded layers.
func (m *DistributedModel) LoadedCount() int {
	return int(atomic.LoadInt32(&m.loadedLayers))
}

// TotalLayers returns the total number of layers.
func (m *DistributedModel) TotalLayers() int {
	return m.totalLayers
}

// ProgressChan returns a channel for loading progress updates.
func (m *DistributedModel) ProgressChan() <-chan LoadProgress {
	return m.progressChan
}

// Close cleans up resources.
func (m *DistributedModel) Close() error {
	close(m.progressChan)
	return m.loader.Close()
}

// Config returns the model configuration.
func (m *DistributedModel) Config() *types.LlamaConfig {
	return m.config
}

// Loader returns the underlying weight loader.
func (m *DistributedModel) Loader() *WeightLoader {
	return m.loader
}
