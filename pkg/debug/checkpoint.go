// Package debug provides debugging utilities for NeuroGrid inference engine.
// This includes tensor checkpointing for numerical validation.
package debug

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// CheckpointConfig holds configuration for the checkpoint writer.
type CheckpointConfig struct {
	BaseDir       string   // Base directory for checkpoints
	EnabledLayers []int    // Layers to checkpoint (empty = all)
	Enabled       bool     // Master switch for checkpointing
}

// CheckpointMetadata contains metadata about a checkpoint.
type CheckpointMetadata struct {
	RequestID   string             `json:"request_id"`
	Model       string             `json:"model"`
	Timestamp   time.Time          `json:"timestamp"`
	NumLayers   int                `json:"num_layers"`
	HiddenSize  int                `json:"hidden_size"`
	SeqLen      int                `json:"seq_len"`
	Position    int                `json:"position"`
	LayerTiming map[int]float64    `json:"layer_timing_ms"`
}

// CheckpointWriter saves tensor checkpoints in SafeTensors format.
type CheckpointWriter struct {
	baseDir       string
	enabledLayers map[int]bool
	enabled       bool
	mu            sync.Mutex
}

// NewCheckpointWriter creates a new checkpoint writer.
func NewCheckpointWriter(config CheckpointConfig) *CheckpointWriter {
	enabledLayers := make(map[int]bool)
	for _, layerID := range config.EnabledLayers {
		enabledLayers[layerID] = true
	}

	return &CheckpointWriter{
		baseDir:       config.BaseDir,
		enabledLayers: enabledLayers,
		enabled:       config.Enabled,
	}
}

// IsEnabled returns true if checkpointing is enabled.
func (c *CheckpointWriter) IsEnabled() bool {
	return c.enabled
}

// SetEnabled enables or disables checkpointing.
func (c *CheckpointWriter) SetEnabled(enabled bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.enabled = enabled
}

// ShouldCheckpointLayer returns true if the given layer should be checkpointed.
func (c *CheckpointWriter) ShouldCheckpointLayer(layerID int) bool {
	if !c.enabled {
		return false
	}
	// If no specific layers are configured, checkpoint all
	if len(c.enabledLayers) == 0 {
		return true
	}
	return c.enabledLayers[layerID]
}

// SaveLayerCheckpoint saves the input and output tensors for a layer.
// Tensors are saved in SafeTensors format for Python compatibility.
func (c *CheckpointWriter) SaveLayerCheckpoint(
	requestID string,
	layerID int,
	position int,
	input, output []byte,
	shape []int64,
) error {
	if !c.ShouldCheckpointLayer(layerID) {
		return nil
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	// Create directory structure: baseDir/requestID/layer_XX/pos_YYY/
	dir := filepath.Join(
		c.baseDir,
		requestID,
		fmt.Sprintf("layer_%02d", layerID),
		fmt.Sprintf("pos_%03d", position),
	)

	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create checkpoint directory: %w", err)
	}

	// Build tensors map
	tensors := map[string]TensorData{}
	if input != nil {
		tensors["input"] = TensorData{
			Dtype: "F16",
			Shape: shape,
			Data:  input,
		}
	}
	if output != nil {
		tensors["output"] = TensorData{
			Dtype: "F16",
			Shape: shape,
			Data:  output,
		}
	}

	// Save as SafeTensors
	tensorPath := filepath.Join(dir, "tensors.safetensors")
	if err := WriteSafeTensors(tensorPath, tensors); err != nil {
		return fmt.Errorf("failed to write SafeTensors: %w", err)
	}

	return nil
}

// SaveCheckpointWithMetadata saves tensors along with metadata.
func (c *CheckpointWriter) SaveCheckpointWithMetadata(
	metadata CheckpointMetadata,
	layerID int,
	input, output []byte,
	shape []int64,
) error {
	if !c.ShouldCheckpointLayer(layerID) {
		return nil
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	// Create directory
	dir := filepath.Join(
		c.baseDir,
		metadata.RequestID,
		fmt.Sprintf("layer_%02d", layerID),
		fmt.Sprintf("pos_%03d", metadata.Position),
	)

	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create checkpoint directory: %w", err)
	}

	// Save metadata
	metadataPath := filepath.Join(dir, "metadata.json")
	metadataBytes, err := json.MarshalIndent(metadata, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal metadata: %w", err)
	}
	if err := os.WriteFile(metadataPath, metadataBytes, 0644); err != nil {
		return fmt.Errorf("failed to write metadata: %w", err)
	}

	// Save tensors
	tensors := map[string]TensorData{}
	if input != nil {
		tensors["input"] = TensorData{
			Dtype: "F16",
			Shape: shape,
			Data:  input,
		}
	}
	if output != nil {
		tensors["output"] = TensorData{
			Dtype: "F16",
			Shape: shape,
			Data:  output,
		}
	}

	tensorPath := filepath.Join(dir, "tensors.safetensors")
	if err := WriteSafeTensors(tensorPath, tensors); err != nil {
		return fmt.Errorf("failed to write SafeTensors: %w", err)
	}

	return nil
}

// TensorData represents a tensor for SafeTensors serialization.
type TensorData struct {
	Dtype string  `json:"dtype"`
	Shape []int64 `json:"shape"`
	Data  []byte  `json:"-"`
}

// WriteSafeTensors writes tensors to a SafeTensors format file.
// This follows the SafeTensors specification for Python compatibility.
func WriteSafeTensors(path string, tensors map[string]TensorData) error {
	// Build header
	header := make(map[string]interface{})

	var dataBuffer bytes.Buffer
	for name, tensor := range tensors {
		start := dataBuffer.Len()
		dataBuffer.Write(tensor.Data)
		end := dataBuffer.Len()

		// CRITICAL: SafeTensors uses "F16" not "float16" for FP16 dtype
		header[name] = map[string]interface{}{
			"dtype":        tensor.Dtype,
			"shape":        tensor.Shape,
			"data_offsets": []int{start, end},
		}
	}

	// Serialize header
	headerBytes, err := json.Marshal(header)
	if err != nil {
		return fmt.Errorf("failed to marshal header: %w", err)
	}

	// Create file
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer f.Close()

	// Write header size (8 bytes, little-endian)
	if err := binary.Write(f, binary.LittleEndian, uint64(len(headerBytes))); err != nil {
		return fmt.Errorf("failed to write header size: %w", err)
	}

	// Write header
	if _, err := f.Write(headerBytes); err != nil {
		return fmt.Errorf("failed to write header: %w", err)
	}

	// Write data
	if _, err := f.Write(dataBuffer.Bytes()); err != nil {
		return fmt.Errorf("failed to write data: %w", err)
	}

	return nil
}

// ListCheckpoints lists all checkpoints in the base directory.
func (c *CheckpointWriter) ListCheckpoints() ([]string, error) {
	var checkpoints []string

	entries, err := os.ReadDir(c.baseDir)
	if err != nil {
		if os.IsNotExist(err) {
			return checkpoints, nil
		}
		return nil, err
	}

	for _, entry := range entries {
		if entry.IsDir() {
			checkpoints = append(checkpoints, entry.Name())
		}
	}

	return checkpoints, nil
}

// CleanupOldCheckpoints removes checkpoints older than the specified duration.
func (c *CheckpointWriter) CleanupOldCheckpoints(maxAge time.Duration) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	entries, err := os.ReadDir(c.baseDir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}

	cutoff := time.Now().Add(-maxAge)

	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}

		info, err := entry.Info()
		if err != nil {
			continue
		}

		if info.ModTime().Before(cutoff) {
			path := filepath.Join(c.baseDir, entry.Name())
			if err := os.RemoveAll(path); err != nil {
				return fmt.Errorf("failed to remove old checkpoint %s: %w", path, err)
			}
		}
	}

	return nil
}
