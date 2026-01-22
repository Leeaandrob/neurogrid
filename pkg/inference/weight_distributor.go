// Package inference provides weight distribution for distributed inference.
package inference

import (
	"bytes"
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"sync"
	"time"

	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/neurogrid/engine/p2p"
	"github.com/neurogrid/engine/pkg/types"
)

// CPUTensor represents tensor data on CPU for network transfer.
// Unlike types.Tensor which uses unsafe.Pointer for GPU memory,
// CPUTensor stores data as a byte slice for serialization.
type CPUTensor struct {
	Shape []int       // Dimensions
	Data  []byte      // Raw tensor data
	Dtype types.Dtype // Element data type
}

// CPULayerWeights holds all weight tensors for a layer in CPU-accessible form.
// This is used for weight serialization and network transfer.
type CPULayerWeights struct {
	LayerID int

	// Attention weights
	AttnNorm *CPUTensor
	QProj    *CPUTensor
	KProj    *CPUTensor
	VProj    *CPUTensor
	OProj    *CPUTensor

	// INT8 quantization scales (optional)
	QScale *CPUTensor
	KScale *CPUTensor
	VScale *CPUTensor
	OScale *CPUTensor

	// FFN weights
	FFNNorm  *CPUTensor
	GateProj *CPUTensor
	UpProj   *CPUTensor
	DownProj *CPUTensor

	// INT8 quantization scales for FFN (optional)
	GateScale *CPUTensor
	UpScale   *CPUTensor
	DownScale *CPUTensor
}

// SerializedLayerWeights represents a single layer's weights in serialized form.
type SerializedLayerWeights struct {
	LayerID int
	Data    []byte   // Concatenated weight tensors
	Offsets []uint64 // Byte offsets for each tensor within Data
	Names   []string // Names of tensors in order
	Shapes  [][]int  // Shape of each tensor
}

// WeightDistributor handles distribution of model weights to remote peers.
type WeightDistributor struct {
	host        host.Host
	protocol    *p2p.Protocol
	modelConfig *types.LlamaConfig

	// Tracks which layers have been distributed to which peers
	distributed   map[string]map[int]bool // peerID -> layerID -> distributed
	distributedMu sync.RWMutex
}

// WeightDistributorConfig holds configuration for creating a WeightDistributor.
type WeightDistributorConfig struct {
	Host        host.Host
	ModelConfig *types.LlamaConfig
}

// NewWeightDistributor creates a new weight distributor.
func NewWeightDistributor(config WeightDistributorConfig) *WeightDistributor {
	wd := &WeightDistributor{
		host:        config.Host,
		modelConfig: config.ModelConfig,
		distributed: make(map[string]map[int]bool),
	}

	// Create protocol for weight transfer
	wd.protocol = p2p.NewProtocol(config.Host)

	return wd
}

// SerializeLayerWeights converts CPULayerWeights to serialized bytes.
// The format is: [count:4B][name_len:4B][name:N][shape_dims:4B][shapes:N*4B][data_len:4B][data:N]...
func SerializeLayerWeights(layerWeights *CPULayerWeights) (*SerializedLayerWeights, error) {
	var buf bytes.Buffer

	// Collect all tensors with names
	tensors := []struct {
		name   string
		tensor *CPUTensor
	}{
		{"attn_norm", layerWeights.AttnNorm},
		{"q_proj", layerWeights.QProj},
		{"k_proj", layerWeights.KProj},
		{"v_proj", layerWeights.VProj},
		{"o_proj", layerWeights.OProj},
		{"ffn_norm", layerWeights.FFNNorm},
		{"gate_proj", layerWeights.GateProj},
		{"up_proj", layerWeights.UpProj},
		{"down_proj", layerWeights.DownProj},
	}

	// Add quantization scales if present
	if layerWeights.QScale != nil {
		tensors = append(tensors, struct {
			name   string
			tensor *CPUTensor
		}{"q_scale", layerWeights.QScale})
	}
	if layerWeights.KScale != nil {
		tensors = append(tensors, struct {
			name   string
			tensor *CPUTensor
		}{"k_scale", layerWeights.KScale})
	}
	if layerWeights.VScale != nil {
		tensors = append(tensors, struct {
			name   string
			tensor *CPUTensor
		}{"v_scale", layerWeights.VScale})
	}
	if layerWeights.OScale != nil {
		tensors = append(tensors, struct {
			name   string
			tensor *CPUTensor
		}{"o_scale", layerWeights.OScale})
	}
	if layerWeights.GateScale != nil {
		tensors = append(tensors, struct {
			name   string
			tensor *CPUTensor
		}{"gate_scale", layerWeights.GateScale})
	}
	if layerWeights.UpScale != nil {
		tensors = append(tensors, struct {
			name   string
			tensor *CPUTensor
		}{"up_scale", layerWeights.UpScale})
	}
	if layerWeights.DownScale != nil {
		tensors = append(tensors, struct {
			name   string
			tensor *CPUTensor
		}{"down_scale", layerWeights.DownScale})
	}

	// Filter out nil tensors
	var validTensors []struct {
		name   string
		tensor *CPUTensor
	}
	for _, t := range tensors {
		if t.tensor != nil {
			validTensors = append(validTensors, t)
		}
	}

	// Write tensor count
	if err := binary.Write(&buf, binary.BigEndian, uint32(len(validTensors))); err != nil {
		return nil, fmt.Errorf("write count: %w", err)
	}

	var offsets []uint64
	var names []string
	var shapes [][]int
	currentOffset := uint64(0)

	// Write each tensor
	for _, t := range validTensors {
		names = append(names, t.name)
		shapes = append(shapes, t.tensor.Shape)

		// Write name length and name
		if err := binary.Write(&buf, binary.BigEndian, uint32(len(t.name))); err != nil {
			return nil, fmt.Errorf("write name length: %w", err)
		}
		if _, err := buf.WriteString(t.name); err != nil {
			return nil, fmt.Errorf("write name: %w", err)
		}

		// Write shape dimensions count and shape
		if err := binary.Write(&buf, binary.BigEndian, uint32(len(t.tensor.Shape))); err != nil {
			return nil, fmt.Errorf("write shape dims: %w", err)
		}
		for _, dim := range t.tensor.Shape {
			if err := binary.Write(&buf, binary.BigEndian, uint32(dim)); err != nil {
				return nil, fmt.Errorf("write shape: %w", err)
			}
		}

		// Write data length and data
		if err := binary.Write(&buf, binary.BigEndian, uint32(len(t.tensor.Data))); err != nil {
			return nil, fmt.Errorf("write data length: %w", err)
		}

		offsets = append(offsets, currentOffset)
		currentOffset += uint64(len(t.tensor.Data))

		if _, err := buf.Write(t.tensor.Data); err != nil {
			return nil, fmt.Errorf("write data: %w", err)
		}
	}

	return &SerializedLayerWeights{
		LayerID: layerWeights.LayerID,
		Data:    buf.Bytes(),
		Offsets: offsets,
		Names:   names,
		Shapes:  shapes,
	}, nil
}

// DeserializeLayerWeights reconstructs CPULayerWeights from serialized bytes.
func DeserializeLayerWeights(layerID int, data []byte) (*CPULayerWeights, error) {
	buf := bytes.NewReader(data)

	// Read tensor count
	var count uint32
	if err := binary.Read(buf, binary.BigEndian, &count); err != nil {
		return nil, fmt.Errorf("read count: %w", err)
	}

	weights := &CPULayerWeights{LayerID: layerID}

	// Read each tensor
	for i := uint32(0); i < count; i++ {
		// Read name length and name
		var nameLen uint32
		if err := binary.Read(buf, binary.BigEndian, &nameLen); err != nil {
			return nil, fmt.Errorf("read name length: %w", err)
		}
		nameBytes := make([]byte, nameLen)
		if _, err := io.ReadFull(buf, nameBytes); err != nil {
			return nil, fmt.Errorf("read name: %w", err)
		}
		name := string(nameBytes)

		// Read shape
		var shapeDims uint32
		if err := binary.Read(buf, binary.BigEndian, &shapeDims); err != nil {
			return nil, fmt.Errorf("read shape dims: %w", err)
		}
		shape := make([]int, shapeDims)
		for j := uint32(0); j < shapeDims; j++ {
			var dim uint32
			if err := binary.Read(buf, binary.BigEndian, &dim); err != nil {
				return nil, fmt.Errorf("read shape: %w", err)
			}
			shape[j] = int(dim)
		}

		// Read data length and data
		var dataLen uint32
		if err := binary.Read(buf, binary.BigEndian, &dataLen); err != nil {
			return nil, fmt.Errorf("read data length: %w", err)
		}
		tensorData := make([]byte, dataLen)
		if _, err := io.ReadFull(buf, tensorData); err != nil {
			return nil, fmt.Errorf("read data: %w", err)
		}

		// Create tensor
		tensor := &CPUTensor{
			Shape: shape,
			Data:  tensorData,
			Dtype: types.DtypeFP16, // Default, should be encoded if needed
		}

		// Assign to appropriate field
		switch name {
		case "attn_norm":
			weights.AttnNorm = tensor
		case "q_proj":
			weights.QProj = tensor
		case "k_proj":
			weights.KProj = tensor
		case "v_proj":
			weights.VProj = tensor
		case "o_proj":
			weights.OProj = tensor
		case "ffn_norm":
			weights.FFNNorm = tensor
		case "gate_proj":
			weights.GateProj = tensor
		case "up_proj":
			weights.UpProj = tensor
		case "down_proj":
			weights.DownProj = tensor
		case "q_scale":
			weights.QScale = tensor
		case "k_scale":
			weights.KScale = tensor
		case "v_scale":
			weights.VScale = tensor
		case "o_scale":
			weights.OScale = tensor
		case "gate_scale":
			weights.GateScale = tensor
		case "up_scale":
			weights.UpScale = tensor
		case "down_scale":
			weights.DownScale = tensor
		}
	}

	return weights, nil
}

// DistributeLayerWeights sends layer weights to a remote peer.
func (wd *WeightDistributor) DistributeLayerWeights(ctx context.Context, peerID peer.ID, layerWeights *CPULayerWeights) error {
	// Serialize the layer weights
	serialized, err := SerializeLayerWeights(layerWeights)
	if err != nil {
		return fmt.Errorf("serialize layer %d: %w", layerWeights.LayerID, err)
	}

	// Send via protocol
	if err := wd.protocol.SendWeights(ctx, peerID, layerWeights.LayerID, serialized.Data); err != nil {
		return fmt.Errorf("send weights for layer %d: %w", layerWeights.LayerID, err)
	}

	// Mark as distributed
	wd.distributedMu.Lock()
	if wd.distributed[peerID.String()] == nil {
		wd.distributed[peerID.String()] = make(map[int]bool)
	}
	wd.distributed[peerID.String()][layerWeights.LayerID] = true
	wd.distributedMu.Unlock()

	return nil
}

// DistributeLayersToPerformer distributes multiple layers to a peer and waits for acknowledgments.
func (wd *WeightDistributor) DistributeLayersToPerformer(ctx context.Context, peerID peer.ID, layerWeights []*CPULayerWeights, timeout time.Duration) error {
	for _, lw := range layerWeights {
		if err := wd.DistributeLayerWeights(ctx, peerID, lw); err != nil {
			return fmt.Errorf("distribute layer %d: %w", lw.LayerID, err)
		}

		// Wait for ack
		if err := wd.protocol.WaitForWeightsAck(ctx, peerID, lw.LayerID, timeout); err != nil {
			return fmt.Errorf("ack layer %d: %w", lw.LayerID, err)
		}
	}
	return nil
}

// IsLayerDistributed checks if a layer has been distributed to a peer.
func (wd *WeightDistributor) IsLayerDistributed(peerID peer.ID, layerID int) bool {
	wd.distributedMu.RLock()
	defer wd.distributedMu.RUnlock()
	if peerLayers, ok := wd.distributed[peerID.String()]; ok {
		return peerLayers[layerID]
	}
	return false
}

// Close releases resources.
func (wd *WeightDistributor) Close() error {
	if wd.protocol != nil {
		return wd.protocol.Close()
	}
	return nil
}

// WeightReceiver handles incoming weight transfers and stores them.
type WeightReceiver struct {
	host          host.Host
	protocol      *p2p.Protocol
	layerWeights  map[int]*CPULayerWeights
	pendingChunks map[int]map[int][]byte // layerID -> chunkIndex -> data
	chunkCounts   map[int]int            // layerID -> totalChunks
	mu            sync.RWMutex

	// Callback when a layer is fully received
	onLayerReceived func(layerID int, weights *CPULayerWeights)
}

// WeightReceiverConfig holds configuration for creating a WeightReceiver.
type WeightReceiverConfig struct {
	Host            host.Host
	OnLayerReceived func(layerID int, weights *CPULayerWeights)
}

// NewWeightReceiver creates a new weight receiver.
func NewWeightReceiver(config WeightReceiverConfig) *WeightReceiver {
	wr := &WeightReceiver{
		host:            config.Host,
		layerWeights:    make(map[int]*CPULayerWeights),
		pendingChunks:   make(map[int]map[int][]byte),
		chunkCounts:     make(map[int]int),
		onLayerReceived: config.OnLayerReceived,
	}

	// Create protocol and register handler
	wr.protocol = p2p.NewProtocol(config.Host)
	wr.protocol.OnWeightsReceived(wr.handleWeightChunk)

	return wr
}

// handleWeightChunk processes an incoming weight chunk.
func (wr *WeightReceiver) handleWeightChunk(layerID int, chunkIndex int, totalChunks int, data []byte) {
	wr.mu.Lock()
	defer wr.mu.Unlock()

	// Initialize chunk storage for this layer if needed
	if wr.pendingChunks[layerID] == nil {
		wr.pendingChunks[layerID] = make(map[int][]byte)
	}

	// Store chunk
	wr.pendingChunks[layerID][chunkIndex] = data
	wr.chunkCounts[layerID] = totalChunks

	// Check if all chunks received
	if len(wr.pendingChunks[layerID]) == totalChunks {
		// Reassemble data
		var fullData []byte
		for i := 0; i < totalChunks; i++ {
			chunk, ok := wr.pendingChunks[layerID][i]
			if !ok {
				return // Missing chunk
			}
			fullData = append(fullData, chunk...)
		}

		// Deserialize
		weights, err := DeserializeLayerWeights(layerID, fullData)
		if err != nil {
			return // Deserialization error
		}

		// Store
		wr.layerWeights[layerID] = weights

		// Clean up pending chunks
		delete(wr.pendingChunks, layerID)
		delete(wr.chunkCounts, layerID)

		// Notify callback
		if wr.onLayerReceived != nil {
			go wr.onLayerReceived(layerID, weights)
		}
	}
}

// GetLayerWeights retrieves received layer weights.
func (wr *WeightReceiver) GetLayerWeights(layerID int) (*CPULayerWeights, bool) {
	wr.mu.RLock()
	defer wr.mu.RUnlock()
	w, ok := wr.layerWeights[layerID]
	return w, ok
}

// GetProtocol returns the underlying protocol for sending acks.
func (wr *WeightReceiver) GetProtocol() *p2p.Protocol {
	return wr.protocol
}

// Close releases resources.
func (wr *WeightReceiver) Close() error {
	if wr.protocol != nil {
		return wr.protocol.Close()
	}
	return nil
}
