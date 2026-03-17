// Package model provides model loading and weight management for LLM inference.
package model

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	"github.com/neurogrid/engine/pkg/types"
)

// ConvertBF16ToFP16 converts bfloat16 data to float16 data.
// BF16 has 1 sign bit, 8 exponent bits, 7 mantissa bits
// FP16 has 1 sign bit, 5 exponent bits, 10 mantissa bits
func ConvertBF16ToFP16(bf16Data []byte) []byte {
	numElements := len(bf16Data) / 2
	fp16Data := make([]byte, len(bf16Data))

	for i := 0; i < numElements; i++ {
		// Read BF16 value (little-endian)
		bf16Bits := uint16(bf16Data[i*2]) | (uint16(bf16Data[i*2+1]) << 8)

		// Convert BF16 to FP32 (just shift left 16 bits)
		fp32Bits := uint32(bf16Bits) << 16
		fp32Val := math.Float32frombits(fp32Bits)

		// Convert FP32 to FP16
		fp16Bits := float32ToFloat16Bits(fp32Val)

		// Write FP16 value (little-endian)
		fp16Data[i*2] = byte(fp16Bits)
		fp16Data[i*2+1] = byte(fp16Bits >> 8)
	}

	return fp16Data
}

// float32ToFloat16Bits converts a float32 to float16 bit representation
func float32ToFloat16Bits(f float32) uint16 {
	bits := math.Float32bits(f)

	sign := (bits >> 31) & 1
	exp := int((bits >> 23) & 0xFF) - 127 // FP32 exponent bias
	mant := bits & 0x7FFFFF               // 23-bit mantissa

	// Handle special cases
	if exp == 128 { // Inf or NaN in FP32
		if mant == 0 {
			// Infinity
			return uint16((sign << 15) | 0x7C00)
		}
		// NaN
		return uint16((sign << 15) | 0x7E00)
	}

	// Rebias exponent for FP16 (bias 15)
	exp16 := exp + 15

	if exp16 <= 0 {
		// Denormalized or zero in FP16
		if exp16 < -10 {
			// Too small, becomes zero
			return uint16(sign << 15)
		}
		// Denormalized: shift mantissa
		mant = (mant | 0x800000) >> uint(1-exp16)
		return uint16((sign << 15) | (mant >> 13))
	}

	if exp16 >= 31 {
		// Overflow, becomes infinity
		return uint16((sign << 15) | 0x7C00)
	}

	// Normal case: truncate mantissa from 23 to 10 bits
	return uint16((sign << 15) | uint32(exp16<<10) | (mant >> 13))
}

// SafeTensorsHeader represents the header of a SafeTensors file.
type SafeTensorsHeader struct {
	Metadata map[string]interface{} `json:"__metadata__,omitempty"`
	Tensors  map[string]TensorInfo  `json:"-"` // Parsed from header
}

// TensorInfo describes a tensor's location and properties in a SafeTensors file.
type TensorInfo struct {
	File    string   // Which file contains this tensor
	Dtype   string   `json:"dtype"`
	Shape   []int64  `json:"shape"`
	Offsets [2]int64 `json:"data_offsets"` // [start, end] byte offsets
}

// WeightLoader loads model weights from SafeTensors format.
type WeightLoader struct {
	basePath string
	index    map[string]TensorInfo
	files    map[string]*os.File // Cached file handles
	mu       sync.RWMutex
	keepBF16 bool          // When true, skip BF16->FP16 conversion (for native BF16 models)
}

// NewWeightLoader creates a new weight loader for a SafeTensors model directory.
func NewWeightLoader(basePath string) (*WeightLoader, error) {
	loader := &WeightLoader{
		basePath: basePath,
		index:    make(map[string]TensorInfo),
		files:    make(map[string]*os.File),
	}

	// Try to load model.safetensors.index.json for sharded models
	indexPath := filepath.Join(basePath, "model.safetensors.index.json")
	if _, err := os.Stat(indexPath); err == nil {
		if err := loader.loadShardedIndex(indexPath); err != nil {
			return nil, fmt.Errorf("failed to load sharded index: %w", err)
		}
		return loader, nil
	}

	// Try to load single model.safetensors
	singlePath := filepath.Join(basePath, "model.safetensors")
	if _, err := os.Stat(singlePath); err == nil {
		if err := loader.loadSingleFile(singlePath); err != nil {
			return nil, fmt.Errorf("failed to load single file: %w", err)
		}
		return loader, nil
	}

	return nil, fmt.Errorf("no SafeTensors model found at %s", basePath)
}

// loadShardedIndex loads the index for a sharded SafeTensors model.
func (l *WeightLoader) loadShardedIndex(indexPath string) error {
	data, err := os.ReadFile(indexPath)
	if err != nil {
		return err
	}

	var shardIndex struct {
		Metadata  map[string]interface{} `json:"metadata"`
		WeightMap map[string]string      `json:"weight_map"` // tensor name -> filename
	}

	if err := json.Unmarshal(data, &shardIndex); err != nil {
		return fmt.Errorf("failed to parse shard index: %w", err)
	}

	// Group tensors by file
	fileToTensors := make(map[string][]string)
	for tensor, file := range shardIndex.WeightMap {
		fileToTensors[file] = append(fileToTensors[file], tensor)
	}

	// Load headers from each shard file
	for file := range fileToTensors {
		shardPath := filepath.Join(l.basePath, file)
		if err := l.loadSingleFile(shardPath); err != nil {
			return fmt.Errorf("failed to load shard %s: %w", file, err)
		}
	}

	return nil
}

// loadSingleFile loads tensor information from a single SafeTensors file.
func (l *WeightLoader) loadSingleFile(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}

	// Read header size (8 bytes, little-endian)
	var headerSize uint64
	if err := binary.Read(f, binary.LittleEndian, &headerSize); err != nil {
		f.Close()
		return fmt.Errorf("failed to read header size: %w", err)
	}

	// Validate header size (sanity check)
	if headerSize > 100*1024*1024 { // 100MB max header
		f.Close()
		return fmt.Errorf("header size too large: %d", headerSize)
	}

	// Read header JSON
	headerBytes := make([]byte, headerSize)
	if _, err := io.ReadFull(f, headerBytes); err != nil {
		f.Close()
		return fmt.Errorf("failed to read header: %w", err)
	}

	// Parse header
	var header map[string]json.RawMessage
	if err := json.Unmarshal(headerBytes, &header); err != nil {
		f.Close()
		return fmt.Errorf("failed to parse header: %w", err)
	}

	// Extract tensor information
	filename := filepath.Base(path)
	dataOffset := int64(8 + headerSize) // Header size field + header

	for name, raw := range header {
		if name == "__metadata__" {
			continue
		}

		var info TensorInfo
		if err := json.Unmarshal(raw, &info); err != nil {
			continue // Skip malformed entries
		}

		info.File = filename
		// Adjust offsets relative to data section
		info.Offsets[0] += dataOffset
		info.Offsets[1] += dataOffset

		l.index[name] = info
	}

	// Cache the file handle
	l.files[filename] = f

	return nil
}

// LoadTensor loads a tensor by name.
func (l *WeightLoader) LoadTensor(name string) ([]byte, *TensorInfo, error) {
	l.mu.RLock()
	info, ok := l.index[name]
	l.mu.RUnlock()

	if !ok {
		return nil, nil, fmt.Errorf("tensor not found: %s", name)
	}

	// Get file handle
	l.mu.RLock()
	f, ok := l.files[info.File]
	l.mu.RUnlock()

	if !ok {
		return nil, nil, fmt.Errorf("file not loaded: %s", info.File)
	}

	// Read tensor data
	size := info.Offsets[1] - info.Offsets[0]
	data := make([]byte, size)

	l.mu.Lock()
	_, err := f.Seek(info.Offsets[0], io.SeekStart)
	if err != nil {
		l.mu.Unlock()
		return nil, nil, fmt.Errorf("seek failed: %w", err)
	}

	_, err = io.ReadFull(f, data)
	l.mu.Unlock()

	if err != nil {
		return nil, nil, fmt.Errorf("read failed: %w", err)
	}

	// Convert BF16 to FP16 if needed (skip if keepBF16 is set on loader)
	if info.Dtype == "BF16" && !l.keepBF16 {
		fmt.Printf("[BF16->FP16] Converting tensor %s (dtype=%s, size=%d bytes)\n", name, info.Dtype, len(data))
		// Debug: show first 8 bytes before conversion
		if len(data) >= 8 {
			fmt.Printf("[BF16->FP16] Before: %02x %02x %02x %02x %02x %02x %02x %02x\n",
				data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7])
		}
		data = ConvertBF16ToFP16(data)
		// Debug: show first 8 bytes after conversion
		if len(data) >= 8 {
			fmt.Printf("[BF16->FP16] After:  %02x %02x %02x %02x %02x %02x %02x %02x\n",
				data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7])
		}
	} else {
		fmt.Printf("[LOAD] Tensor %s has dtype=%s (no conversion)\n", name, info.Dtype)
	}

	return data, &info, nil
}

// LoadLayerWeights loads all weights for a transformer layer.
func (l *WeightLoader) LoadLayerWeights(layerID int) (*LayerWeights, error) {
	prefix := fmt.Sprintf("model.layers.%d.", layerID)

	weights := &LayerWeights{
		LayerID: layerID,
	}

	// Load attention weights
	var err error
	weights.QWeight, _, err = l.LoadTensor(prefix + "self_attn.q_proj.weight")
	if err != nil {
		return nil, fmt.Errorf("failed to load Q weight: %w", err)
	}

	weights.KWeight, _, err = l.LoadTensor(prefix + "self_attn.k_proj.weight")
	if err != nil {
		return nil, fmt.Errorf("failed to load K weight: %w", err)
	}

	weights.VWeight, _, err = l.LoadTensor(prefix + "self_attn.v_proj.weight")
	if err != nil {
		return nil, fmt.Errorf("failed to load V weight: %w", err)
	}

	weights.OWeight, _, err = l.LoadTensor(prefix + "self_attn.o_proj.weight")
	if err != nil {
		return nil, fmt.Errorf("failed to load O weight: %w", err)
	}

	// Load FFN weights
	weights.GateWeight, _, err = l.LoadTensor(prefix + "mlp.gate_proj.weight")
	if err != nil {
		return nil, fmt.Errorf("failed to load gate weight: %w", err)
	}

	weights.UpWeight, _, err = l.LoadTensor(prefix + "mlp.up_proj.weight")
	if err != nil {
		return nil, fmt.Errorf("failed to load up weight: %w", err)
	}

	weights.DownWeight, _, err = l.LoadTensor(prefix + "mlp.down_proj.weight")
	if err != nil {
		return nil, fmt.Errorf("failed to load down weight: %w", err)
	}

	// Load norms
	weights.AttnNorm, _, err = l.LoadTensor(prefix + "input_layernorm.weight")
	if err != nil {
		return nil, fmt.Errorf("failed to load attention norm: %w", err)
	}

	weights.FFNNorm, _, err = l.LoadTensor(prefix + "post_attention_layernorm.weight")
	if err != nil {
		return nil, fmt.Errorf("failed to load FFN norm: %w", err)
	}

	return weights, nil
}

// ConvLayerWeights holds weights for an LFM2 conv layer.
type ConvLayerWeights struct {
	LayerID      int
	InProjWeight []byte // [3*hidden, hidden]
	ConvWeight   []byte // [hidden, kernel_size] (reshaped from [hidden, 1, kernel])
	OutProjWeight []byte // [hidden, hidden]
	OperatorNorm []byte // [hidden]
	FFNNorm      []byte // [hidden]
	GateWeight   []byte // [intermediate, hidden]
	UpWeight     []byte // [intermediate, hidden]
	DownWeight   []byte // [hidden, intermediate]
}

// LoadConvLayerWeights loads weights for an LFM2 conv layer.
func (l *WeightLoader) LoadConvLayerWeights(layerID int) (*ConvLayerWeights, error) {
	prefix := fmt.Sprintf("model.layers.%d.", layerID)
	weights := &ConvLayerWeights{LayerID: layerID}
	var err error

	weights.InProjWeight, _, err = l.LoadTensor(prefix + "conv.in_proj.weight")
	if err != nil {
		return nil, fmt.Errorf("load conv in_proj layer %d: %w", layerID, err)
	}

	convData, convInfo, err := l.LoadTensor(prefix + "conv.conv.weight")
	if err != nil {
		return nil, fmt.Errorf("load conv weight layer %d: %w", layerID, err)
	}
	// Reshape [hidden, 1, kernel] -> [hidden, kernel] by dropping middle dim
	if convInfo != nil && len(convInfo.Shape) == 3 && convInfo.Shape[1] == 1 {
		weights.ConvWeight = convData // Already contiguous, just different view
	} else {
		weights.ConvWeight = convData
	}

	weights.OutProjWeight, _, err = l.LoadTensor(prefix + "conv.out_proj.weight")
	if err != nil {
		return nil, fmt.Errorf("load conv out_proj layer %d: %w", layerID, err)
	}

	weights.OperatorNorm, _, err = l.LoadTensor(prefix + "operator_norm.weight")
	if err != nil {
		return nil, fmt.Errorf("load operator_norm layer %d: %w", layerID, err)
	}

	weights.FFNNorm, _, err = l.LoadTensor(prefix + "ffn_norm.weight")
	if err != nil {
		return nil, fmt.Errorf("load ffn_norm layer %d: %w", layerID, err)
	}

	weights.GateWeight, _, err = l.LoadTensor(prefix + "feed_forward.w1.weight")
	if err != nil {
		return nil, fmt.Errorf("load gate weight layer %d: %w", layerID, err)
	}

	weights.UpWeight, _, err = l.LoadTensor(prefix + "feed_forward.w3.weight")
	if err != nil {
		return nil, fmt.Errorf("load up weight layer %d: %w", layerID, err)
	}

	weights.DownWeight, _, err = l.LoadTensor(prefix + "feed_forward.w2.weight")
	if err != nil {
		return nil, fmt.Errorf("load down weight layer %d: %w", layerID, err)
	}

	return weights, nil
}

// LoadAttentionLayerWeightsLFM2 loads weights for an LFM2 attention layer (with QK norms).
func (l *WeightLoader) LoadAttentionLayerWeightsLFM2(layerID int) (*LayerWeights, []byte, []byte, error) {
	prefix := fmt.Sprintf("model.layers.%d.", layerID)
	weights := &LayerWeights{LayerID: layerID}
	var err error

	weights.QWeight, _, err = l.LoadTensor(prefix + "self_attn.q_proj.weight")
	if err != nil {
		return nil, nil, nil, fmt.Errorf("load Q weight layer %d: %w", layerID, err)
	}

	weights.KWeight, _, err = l.LoadTensor(prefix + "self_attn.k_proj.weight")
	if err != nil {
		return nil, nil, nil, fmt.Errorf("load K weight layer %d: %w", layerID, err)
	}

	weights.VWeight, _, err = l.LoadTensor(prefix + "self_attn.v_proj.weight")
	if err != nil {
		return nil, nil, nil, fmt.Errorf("load V weight layer %d: %w", layerID, err)
	}

	weights.OWeight, _, err = l.LoadTensor(prefix + "self_attn.o_proj.weight")
	if err != nil {
		return nil, nil, nil, fmt.Errorf("load O weight layer %d: %w", layerID, err)
	}

	// QK LayerNorm weights (LFM2 specific)
	qLayerNorm, _, err := l.LoadTensor(prefix + "self_attn.q_layernorm.weight")
	if err != nil {
		return nil, nil, nil, fmt.Errorf("load q_layernorm layer %d: %w", layerID, err)
	}

	kLayerNorm, _, err := l.LoadTensor(prefix + "self_attn.k_layernorm.weight")
	if err != nil {
		return nil, nil, nil, fmt.Errorf("load k_layernorm layer %d: %w", layerID, err)
	}

	// LFM2 uses operator_norm instead of input_layernorm
	weights.AttnNorm, _, err = l.LoadTensor(prefix + "operator_norm.weight")
	if err != nil {
		return nil, nil, nil, fmt.Errorf("load operator_norm layer %d: %w", layerID, err)
	}

	weights.FFNNorm, _, err = l.LoadTensor(prefix + "ffn_norm.weight")
	if err != nil {
		return nil, nil, nil, fmt.Errorf("load ffn_norm layer %d: %w", layerID, err)
	}

	// LFM2 uses feed_forward instead of mlp
	weights.GateWeight, _, err = l.LoadTensor(prefix + "feed_forward.w1.weight")
	if err != nil {
		return nil, nil, nil, fmt.Errorf("load gate weight layer %d: %w", layerID, err)
	}

	weights.UpWeight, _, err = l.LoadTensor(prefix + "feed_forward.w3.weight")
	if err != nil {
		return nil, nil, nil, fmt.Errorf("load up weight layer %d: %w", layerID, err)
	}

	weights.DownWeight, _, err = l.LoadTensor(prefix + "feed_forward.w2.weight")
	if err != nil {
		return nil, nil, nil, fmt.Errorf("load down weight layer %d: %w", layerID, err)
	}

	return weights, qLayerNorm, kLayerNorm, nil
}

// LoadEmbeddings loads the token embedding matrix.
func (l *WeightLoader) LoadEmbeddings() ([]byte, *TensorInfo, error) {
	return l.LoadTensor("model.embed_tokens.weight")
}

// LoadLMHead loads the language model output projection.
func (l *WeightLoader) LoadLMHead() ([]byte, *TensorInfo, error) {
	// Try lm_head.weight first, then fall back to tied embeddings
	data, info, err := l.LoadTensor("lm_head.weight")
	if err == nil {
		return data, info, nil
	}

	// Llama / LFM2 with tied embeddings
	return l.LoadEmbeddings()
}

// SetKeepBF16 configures the loader to skip BF16->FP16 conversion.
// Used for models that run natively in BF16 (e.g., LFM2).
func (l *WeightLoader) SetKeepBF16(keep bool) {
	l.keepBF16 = keep
}

// ListTensors returns all tensor names in the model.
func (l *WeightLoader) ListTensors() []string {
	l.mu.RLock()
	defer l.mu.RUnlock()

	names := make([]string, 0, len(l.index))
	for name := range l.index {
		names = append(names, name)
	}
	sort.Strings(names)
	return names
}

// GetTensorInfo returns information about a tensor.
func (l *WeightLoader) GetTensorInfo(name string) (*TensorInfo, bool) {
	l.mu.RLock()
	defer l.mu.RUnlock()

	info, ok := l.index[name]
	if !ok {
		return nil, false
	}
	return &info, true
}

// CountLayers returns the number of transformer layers in the model.
func (l *WeightLoader) CountLayers() int {
	l.mu.RLock()
	defer l.mu.RUnlock()

	maxLayer := -1
	for name := range l.index {
		if strings.HasPrefix(name, "model.layers.") {
			// Extract layer number
			parts := strings.Split(name, ".")
			if len(parts) >= 3 {
				var layer int
				if _, err := fmt.Sscanf(parts[2], "%d", &layer); err == nil {
					if layer > maxLayer {
						maxLayer = layer
					}
				}
			}
		}
	}
	return maxLayer + 1
}

// Close closes all open file handles.
func (l *WeightLoader) Close() error {
	l.mu.Lock()
	defer l.mu.Unlock()

	var lastErr error
	for _, f := range l.files {
		if err := f.Close(); err != nil {
			lastErr = err
		}
	}
	l.files = make(map[string]*os.File)
	return lastErr
}

// ValidateShapes validates that the loaded tensor shapes match the expected
// model configuration. This helps catch issues early when loading models.
func (l *WeightLoader) ValidateShapes(config *types.LlamaConfig) error {
	l.mu.RLock()
	defer l.mu.RUnlock()

	var errors []string

	// Validate embedding shape: [vocab_size, hidden_size]
	if info, ok := l.index["model.embed_tokens.weight"]; ok {
		expectedShape := []int64{int64(config.VocabSize), int64(config.HiddenSize)}
		if !shapeMatches(info.Shape, expectedShape) {
			errors = append(errors, fmt.Sprintf(
				"embedding shape mismatch: got %v, expected %v", info.Shape, expectedShape))
		}
	}

	// Validate layer shapes
	numLayers := l.countLayersUnsafe()
	for layer := 0; layer < numLayers; layer++ {
		prefix := fmt.Sprintf("model.layers.%d.", layer)

		// Q projection: [num_heads * head_dim, hidden_size]
		if info, ok := l.index[prefix+"self_attn.q_proj.weight"]; ok {
			expectedQ := []int64{int64(config.NumHeads * config.HeadDim), int64(config.HiddenSize)}
			if !shapeMatches(info.Shape, expectedQ) {
				errors = append(errors, fmt.Sprintf(
					"layer %d Q projection shape mismatch: got %v, expected %v",
					layer, info.Shape, expectedQ))
			}
		}

		// K projection: [num_kv_heads * head_dim, hidden_size] (GQA support)
		if info, ok := l.index[prefix+"self_attn.k_proj.weight"]; ok {
			expectedK := []int64{int64(config.NumKVHeads * config.HeadDim), int64(config.HiddenSize)}
			if !shapeMatches(info.Shape, expectedK) {
				errors = append(errors, fmt.Sprintf(
					"layer %d K projection shape mismatch: got %v, expected %v",
					layer, info.Shape, expectedK))
			}
		}

		// V projection: [num_kv_heads * head_dim, hidden_size]
		if info, ok := l.index[prefix+"self_attn.v_proj.weight"]; ok {
			expectedV := []int64{int64(config.NumKVHeads * config.HeadDim), int64(config.HiddenSize)}
			if !shapeMatches(info.Shape, expectedV) {
				errors = append(errors, fmt.Sprintf(
					"layer %d V projection shape mismatch: got %v, expected %v",
					layer, info.Shape, expectedV))
			}
		}

		// O projection: [hidden_size, num_heads * head_dim]
		if info, ok := l.index[prefix+"self_attn.o_proj.weight"]; ok {
			expectedO := []int64{int64(config.HiddenSize), int64(config.NumHeads * config.HeadDim)}
			if !shapeMatches(info.Shape, expectedO) {
				errors = append(errors, fmt.Sprintf(
					"layer %d O projection shape mismatch: got %v, expected %v",
					layer, info.Shape, expectedO))
			}
		}

		// Gate projection: [intermediate_size, hidden_size]
		if info, ok := l.index[prefix+"mlp.gate_proj.weight"]; ok {
			expectedGate := []int64{int64(config.IntermediateSize), int64(config.HiddenSize)}
			if !shapeMatches(info.Shape, expectedGate) {
				errors = append(errors, fmt.Sprintf(
					"layer %d gate projection shape mismatch: got %v, expected %v",
					layer, info.Shape, expectedGate))
			}
		}

		// Up projection: [intermediate_size, hidden_size]
		if info, ok := l.index[prefix+"mlp.up_proj.weight"]; ok {
			expectedUp := []int64{int64(config.IntermediateSize), int64(config.HiddenSize)}
			if !shapeMatches(info.Shape, expectedUp) {
				errors = append(errors, fmt.Sprintf(
					"layer %d up projection shape mismatch: got %v, expected %v",
					layer, info.Shape, expectedUp))
			}
		}

		// Down projection: [hidden_size, intermediate_size]
		if info, ok := l.index[prefix+"mlp.down_proj.weight"]; ok {
			expectedDown := []int64{int64(config.HiddenSize), int64(config.IntermediateSize)}
			if !shapeMatches(info.Shape, expectedDown) {
				errors = append(errors, fmt.Sprintf(
					"layer %d down projection shape mismatch: got %v, expected %v",
					layer, info.Shape, expectedDown))
			}
		}

		// Input layernorm: [hidden_size]
		if info, ok := l.index[prefix+"input_layernorm.weight"]; ok {
			expectedNorm := []int64{int64(config.HiddenSize)}
			if !shapeMatches(info.Shape, expectedNorm) {
				errors = append(errors, fmt.Sprintf(
					"layer %d input layernorm shape mismatch: got %v, expected %v",
					layer, info.Shape, expectedNorm))
			}
		}

		// Post-attention layernorm: [hidden_size]
		if info, ok := l.index[prefix+"post_attention_layernorm.weight"]; ok {
			expectedNorm := []int64{int64(config.HiddenSize)}
			if !shapeMatches(info.Shape, expectedNorm) {
				errors = append(errors, fmt.Sprintf(
					"layer %d post-attention layernorm shape mismatch: got %v, expected %v",
					layer, info.Shape, expectedNorm))
			}
		}
	}

	if len(errors) > 0 {
		return fmt.Errorf("shape validation failed:\n  - %s", strings.Join(errors, "\n  - "))
	}

	return nil
}

// countLayersUnsafe counts layers without acquiring the lock (caller must hold lock).
func (l *WeightLoader) countLayersUnsafe() int {
	maxLayer := -1
	for name := range l.index {
		if strings.HasPrefix(name, "model.layers.") {
			parts := strings.Split(name, ".")
			if len(parts) >= 3 {
				var layer int
				if _, err := fmt.Sscanf(parts[2], "%d", &layer); err == nil {
					if layer > maxLayer {
						maxLayer = layer
					}
				}
			}
		}
	}
	return maxLayer + 1
}

// shapeMatches checks if two shapes are equal.
func shapeMatches(actual, expected []int64) bool {
	if len(actual) != len(expected) {
		return false
	}
	for i := range actual {
		if actual[i] != expected[i] {
			return false
		}
	}
	return true
}

// LayerWeights holds all weight tensors for a single transformer layer.
type LayerWeights struct {
	LayerID    int
	QWeight    []byte // Query projection
	KWeight    []byte // Key projection
	VWeight    []byte // Value projection
	OWeight    []byte // Output projection
	GateWeight []byte // FFN gate projection
	UpWeight   []byte // FFN up projection
	DownWeight []byte // FFN down projection
	AttnNorm   []byte // Attention layer norm
	FFNNorm    []byte // FFN layer norm
}

// CreateMockSafeTensors creates a mock SafeTensors file for testing.
func CreateMockSafeTensors(path string, tensors map[string][]byte) error {
	// Build header
	header := make(map[string]interface{})

	var dataBuffer bytes.Buffer
	for name, data := range tensors {
		start := dataBuffer.Len()
		dataBuffer.Write(data)
		end := dataBuffer.Len()

		header[name] = map[string]interface{}{
			"dtype":        "F16",
			"shape":        []int{len(data) / 2}, // Assume FP16
			"data_offsets": []int{start, end},
		}
	}

	// Serialize header
	headerBytes, err := json.Marshal(header)
	if err != nil {
		return err
	}

	// Write file
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	// Write header size
	headerSize := uint64(len(headerBytes))
	if err := binary.Write(f, binary.LittleEndian, headerSize); err != nil {
		return err
	}

	// Write header
	if _, err := f.Write(headerBytes); err != nil {
		return err
	}

	// Write data
	if _, err := f.Write(dataBuffer.Bytes()); err != nil {
		return err
	}

	return nil
}

// DtypeByteSize returns the byte size of a dtype string.
func DtypeByteSize(dtype string) int {
	switch dtype {
	case "F32":
		return 4
	case "F16", "BF16":
		return 2
	case "I8", "U8":
		return 1
	case "I16", "U16":
		return 2
	case "I32", "U32":
		return 4
	case "I64", "U64":
		return 8
	default:
		return 0
	}
}
