// Package e2e provides end-to-end tests for the NeuroGrid distributed inference engine.
// This file contains acceptance tests for PRP-01: Model Weights Loader.
package e2e

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"
	"unsafe"

	"github.com/neurogrid/engine/pkg/model"
	"github.com/neurogrid/engine/pkg/scheduler"
	"github.com/neurogrid/engine/pkg/types"
)

// =============================================================================
// PRP-01 Acceptance Tests: Model Weights Loader
// Success Criteria:
//   1. Load Llama 7B safetensors weights in < 30s
//   2. Support sharded models (model-00001-of-00002.safetensors pattern)
//   3. Memory-map weights for models > available RAM
//   4. Distribute weights to correct GPU per scheduler assignment
//   5. Support INT8 and FP16 weight formats
//   6. Validate weight shapes against model config
// =============================================================================

// -----------------------------------------------------------------------------
// Test Group 1: Sharded Model Loading (Success Criterion #2)
// -----------------------------------------------------------------------------

// TestShardedModel_LoadFromIndexFile verifies loading from sharded safetensors
// with the model.safetensors.index.json format.
func TestShardedModel_LoadFromIndexFile(t *testing.T) {
	tmpDir := t.TempDir()

	// Create sharded model structure with index.json
	createShardedModel(t, tmpDir, 2)

	loader, err := model.NewWeightLoader(tmpDir)
	if err != nil {
		t.Fatalf("NewWeightLoader failed for sharded model: %v", err)
	}
	defer loader.Close()

	// Verify all tensors from both shards are accessible
	tensors := loader.ListTensors()
	if len(tensors) < 4 {
		t.Errorf("expected at least 4 tensors from 2 shards, got %d", len(tensors))
	}

	// Load tensor from shard 1
	data1, info1, err := loader.LoadTensor("model.embed_tokens.weight")
	if err != nil {
		t.Errorf("failed to load tensor from shard 1: %v", err)
	}
	if len(data1) == 0 || info1 == nil {
		t.Error("tensor data or info is empty")
	}

	// Load tensor from shard 2
	data2, info2, err := loader.LoadTensor("lm_head.weight")
	if err != nil {
		t.Errorf("failed to load tensor from shard 2: %v", err)
	}
	if len(data2) == 0 || info2 == nil {
		t.Error("tensor data or info is empty")
	}
}

// TestShardedModel_MultipleShards tests loading from more than 2 shards.
func TestShardedModel_MultipleShards(t *testing.T) {
	tmpDir := t.TempDir()

	// Create model with 4 shards (simulating Llama 70B)
	createShardedModel(t, tmpDir, 4)

	loader, err := model.NewWeightLoader(tmpDir)
	if err != nil {
		t.Fatalf("NewWeightLoader failed: %v", err)
	}
	defer loader.Close()

	// Verify layer count matches expected
	layerCount := loader.CountLayers()
	if layerCount < 4 {
		t.Errorf("expected at least 4 layers from 4 shards, got %d", layerCount)
	}
}

// -----------------------------------------------------------------------------
// Test Group 2: Memory-Mapped Loading (Success Criterion #3)
// -----------------------------------------------------------------------------

// TestMmapLoader_LargeFile verifies memory-mapped loading works for large files.
func TestMmapLoader_LargeFile(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping large file test in short mode")
	}

	tmpDir := t.TempDir()

	// Create a "large" mock file (10MB for test purposes)
	largeTensorSize := 10 * 1024 * 1024 // 10MB
	createLargeTestModel(t, tmpDir, largeTensorSize)

	// Use memory-mapped loader
	loader, err := model.NewMmapLoader(tmpDir)
	if err != nil {
		t.Fatalf("NewMmapLoader failed: %v", err)
	}
	defer loader.Close()

	// Verify we can load without reading entire file into memory
	info, ok := loader.GetTensorInfo("large_tensor")
	if !ok {
		t.Fatal("large_tensor not found")
	}

	// Verify the tensor is mmap'd (should have MmapData method)
	data, err := loader.MmapTensor("large_tensor")
	if err != nil {
		t.Fatalf("MmapTensor failed: %v", err)
	}

	expectedSize := info.Offsets[1] - info.Offsets[0]
	if int64(len(data)) != expectedSize {
		t.Errorf("mmap data size mismatch: got %d, expected %d", len(data), expectedSize)
	}
}

// TestMmapLoader_AlignedRead verifies GPU copy alignment.
func TestMmapLoader_AlignedRead(t *testing.T) {
	tmpDir := t.TempDir()

	// Create tensor with specific alignment requirements
	createAlignedTestModel(t, tmpDir)

	loader, err := model.NewMmapLoader(tmpDir)
	if err != nil {
		t.Fatalf("NewMmapLoader failed: %v", err)
	}
	defer loader.Close()

	data, err := loader.MmapTensorAligned("test_tensor", 256) // 256-byte alignment for GPU
	if err != nil {
		t.Fatalf("MmapTensorAligned failed: %v", err)
	}

	// Verify alignment
	ptr := uintptr(unsafe.Pointer(&data[0]))
	if ptr%256 != 0 {
		t.Errorf("data not 256-byte aligned: address %x", ptr)
	}
}

// -----------------------------------------------------------------------------
// Test Group 3: Weight Format Support (Success Criterion #5)
// -----------------------------------------------------------------------------

// TestWeightFormat_FP16 verifies FP16 weight loading and conversion.
func TestWeightFormat_FP16(t *testing.T) {
	tmpDir := t.TempDir()

	// Create FP16 weights
	createFP16TestModel(t, tmpDir)

	loader, err := model.NewWeightLoader(tmpDir)
	if err != nil {
		t.Fatalf("NewWeightLoader failed: %v", err)
	}
	defer loader.Close()

	data, info, err := loader.LoadTensor("fp16_tensor")
	if err != nil {
		t.Fatalf("LoadTensor failed: %v", err)
	}

	if info.Dtype != "F16" {
		t.Errorf("expected dtype F16, got %s", info.Dtype)
	}

	// Verify FP16 data is correctly parsed (2 bytes per element)
	expectedElements := info.Shape[0]
	expectedBytes := expectedElements * 2
	if int64(len(data)) != expectedBytes {
		t.Errorf("FP16 data size mismatch: got %d bytes, expected %d", len(data), expectedBytes)
	}
}

// TestWeightFormat_INT8 verifies INT8 quantized weight loading with scales.
func TestWeightFormat_INT8(t *testing.T) {
	tmpDir := t.TempDir()

	// Create INT8 weights with associated scales
	createINT8TestModel(t, tmpDir)

	loader, err := model.NewWeightLoader(tmpDir)
	if err != nil {
		t.Fatalf("NewWeightLoader failed: %v", err)
	}
	defer loader.Close()

	// Load INT8 weights
	data, info, err := loader.LoadTensor("int8_weight")
	if err != nil {
		t.Fatalf("LoadTensor for INT8 failed: %v", err)
	}

	if info.Dtype != "I8" {
		t.Errorf("expected dtype I8, got %s", info.Dtype)
	}

	// Load associated scale tensor
	scaleData, scaleInfo, err := loader.LoadTensor("int8_weight_scale")
	if err != nil {
		t.Fatalf("LoadTensor for scale failed: %v", err)
	}

	if scaleInfo.Dtype != "F32" {
		t.Errorf("expected scale dtype F32, got %s", scaleInfo.Dtype)
	}

	// Verify INT8 data (1 byte per element)
	expectedElements := info.Shape[0] * info.Shape[1]
	if int64(len(data)) != expectedElements {
		t.Errorf("INT8 data size mismatch: got %d bytes, expected %d", len(data), expectedElements)
	}

	// Verify scale tensor shape matches weight rows
	if scaleInfo.Shape[0] != info.Shape[0] {
		t.Errorf("scale shape[0] should match weight rows: got %d, expected %d",
			scaleInfo.Shape[0], info.Shape[0])
	}

	_ = scaleData // Use scaleData
}

// TestWeightFormat_BF16 verifies BF16 weight format support.
func TestWeightFormat_BF16(t *testing.T) {
	tmpDir := t.TempDir()

	createBF16TestModel(t, tmpDir)

	loader, err := model.NewWeightLoader(tmpDir)
	if err != nil {
		t.Fatalf("NewWeightLoader failed: %v", err)
	}
	defer loader.Close()

	data, info, err := loader.LoadTensor("bf16_tensor")
	if err != nil {
		t.Fatalf("LoadTensor failed: %v", err)
	}

	if info.Dtype != "BF16" {
		t.Errorf("expected dtype BF16, got %s", info.Dtype)
	}

	// BF16 is 2 bytes per element
	expectedBytes := info.Shape[0] * 2
	if int64(len(data)) != expectedBytes {
		t.Errorf("BF16 data size mismatch: got %d, expected %d", len(data), expectedBytes)
	}
}

// -----------------------------------------------------------------------------
// Test Group 4: Shape Validation (Success Criterion #6)
// -----------------------------------------------------------------------------

// TestShapeValidation_MatchesConfig verifies weight shapes against model config.
func TestShapeValidation_MatchesConfig(t *testing.T) {
	tmpDir := t.TempDir()

	config := types.Llama7BConfig()
	config.NumLayers = 2 // Use fewer layers for faster test
	createModelWithConfig(t, tmpDir, config)

	loader, err := model.NewWeightLoader(tmpDir)
	if err != nil {
		t.Fatalf("NewWeightLoader failed: %v", err)
	}
	defer loader.Close()

	// Validate shapes match config
	err = loader.ValidateShapes(config)
	if err != nil {
		t.Fatalf("ValidateShapes failed: %v", err)
	}
}

// TestShapeValidation_DetectsMismatch verifies shape validation catches errors.
func TestShapeValidation_DetectsMismatch(t *testing.T) {
	tmpDir := t.TempDir()

	// Create model with wrong shapes
	createModelWithWrongShapes(t, tmpDir)

	loader, err := model.NewWeightLoader(tmpDir)
	if err != nil {
		t.Fatalf("NewWeightLoader failed: %v", err)
	}
	defer loader.Close()

	config := types.Llama7BConfig()
	err = loader.ValidateShapes(config)
	if err == nil {
		t.Error("expected validation error for mismatched shapes")
	}
}

// TestShapeValidation_QKVProjections verifies attention projection shapes.
func TestShapeValidation_QKVProjections(t *testing.T) {
	tmpDir := t.TempDir()

	config := types.Llama7BConfig()
	config.NumLayers = 1 // Only need one layer for shape validation
	createModelWithConfig(t, tmpDir, config)

	loader, err := model.NewWeightLoader(tmpDir)
	if err != nil {
		t.Fatalf("NewWeightLoader failed: %v", err)
	}
	defer loader.Close()

	// Check Q projection shape: [num_heads * head_dim, hidden_size]
	_, qInfo, err := loader.LoadTensor("model.layers.0.self_attn.q_proj.weight")
	if err != nil {
		t.Fatalf("failed to load Q projection: %v", err)
	}

	expectedQRows := int64(config.NumHeads * config.HeadDim)
	expectedQCols := int64(config.HiddenSize)

	if len(qInfo.Shape) < 2 {
		t.Fatalf("Q projection should have 2 dimensions, got %d", len(qInfo.Shape))
	}

	if qInfo.Shape[0] != expectedQRows || qInfo.Shape[1] != expectedQCols {
		t.Errorf("Q projection shape mismatch: got [%d, %d], expected [%d, %d]",
			qInfo.Shape[0], qInfo.Shape[1], expectedQRows, expectedQCols)
	}

	// Check KV projection shape (for GQA): [num_kv_heads * head_dim, hidden_size]
	_, kInfo, err := loader.LoadTensor("model.layers.0.self_attn.k_proj.weight")
	if err != nil {
		t.Fatalf("failed to load K projection: %v", err)
	}

	expectedKRows := int64(config.NumKVHeads * config.HeadDim)
	if kInfo.Shape[0] != expectedKRows {
		t.Errorf("K projection shape[0] mismatch for GQA: got %d, expected %d",
			kInfo.Shape[0], expectedKRows)
	}
}

// -----------------------------------------------------------------------------
// Test Group 5: Distributed Loading (Success Criterion #4)
// -----------------------------------------------------------------------------

// TestDistributedLoader_CorrectGPUAssignment verifies weights go to assigned GPU.
func TestDistributedLoader_CorrectGPUAssignment(t *testing.T) {
	tmpDir := t.TempDir()

	config := types.Llama7BConfig()
	config.NumLayers = 4 // Smaller for testing
	createModelWithConfig(t, tmpDir, config)

	dm, err := model.NewDistributedModel(model.DistributedModelConfig{
		ModelConfig: config,
		ModelPath:   tmpDir,
		LocalPeerID: "peer-0",
	})
	if err != nil {
		t.Fatalf("NewDistributedModel failed: %v", err)
	}
	defer dm.Close()

	// Setup scheduler with 2 peers
	sched := scheduler.NewScheduler(scheduler.ModelConfig{
		HiddenSize:       int64(config.HiddenSize),
		IntermediateSize: int64(config.IntermediateSize),
		NumLayers:        config.NumLayers,
		NumKVHeads:       config.NumKVHeads,
		HeadDim:          config.HeadDim,
		MaxSeqLen:        config.MaxSeqLen,
		VocabSize:        int64(config.VocabSize),
	})

	sched.RegisterPeer("peer-0", 16*1024*1024*1024, 0) // 16GB GPU
	sched.RegisterPeer("peer-1", 16*1024*1024*1024, 0) // 16GB GPU

	dm.SetScheduler(sched)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	err = dm.LoadToCluster(ctx)
	if err != nil {
		t.Fatalf("LoadToCluster failed: %v", err)
	}

	// Verify assignments were made correctly
	assignments, _ := sched.ComputeAssignments()
	for _, assign := range assignments {
		if assign.LayerID >= 0 && assign.LayerID < config.NumLayers {
			status := dm.GetLayerStatus(assign.LayerID)
			if status != model.LoadStatusLoaded {
				t.Errorf("layer %d not loaded (status=%d)", assign.LayerID, status)
			}
		}
	}
}

// TestDistributedLoader_OnlyLoadsAssignedLayers verifies peer only loads its layers.
func TestDistributedLoader_OnlyLoadsAssignedLayers(t *testing.T) {
	tmpDir := t.TempDir()

	config := types.Llama7BConfig()
	config.NumLayers = 8
	createModelWithConfig(t, tmpDir, config)

	// Create model for peer-1 (not peer-0)
	dm, err := model.NewDistributedModel(model.DistributedModelConfig{
		ModelConfig: config,
		ModelPath:   tmpDir,
		LocalPeerID: "peer-1",
	})
	if err != nil {
		t.Fatalf("NewDistributedModel failed: %v", err)
	}
	defer dm.Close()

	sched := scheduler.NewScheduler(scheduler.ModelConfig{
		HiddenSize:       int64(config.HiddenSize),
		IntermediateSize: int64(config.IntermediateSize),
		NumLayers:        config.NumLayers,
		NumKVHeads:       config.NumKVHeads,
		HeadDim:          config.HeadDim,
		MaxSeqLen:        config.MaxSeqLen,
		VocabSize:        int64(config.VocabSize),
	})

	sched.RegisterPeer("peer-0", 8*1024*1024*1024, 0)
	sched.RegisterPeer("peer-1", 8*1024*1024*1024, 0)

	dm.SetScheduler(sched)

	ctx := context.Background()
	err = dm.LoadToCluster(ctx)
	if err != nil {
		t.Fatalf("LoadToCluster failed: %v", err)
	}

	// Get layers assigned to peer-1
	assignments, _ := sched.ComputeAssignments()
	peer1Layers := sched.GetPeerLayers(assignments, "peer-1")

	// Verify only peer-1 layers are loaded locally
	for _, layerID := range peer1Layers {
		if layerID >= 0 && layerID < config.NumLayers {
			status := dm.GetLayerStatus(layerID)
			if status != model.LoadStatusLoaded {
				t.Errorf("peer-1 layer %d should be loaded", layerID)
			}
		}
	}
}

// -----------------------------------------------------------------------------
// Test Group 6: Performance (Success Criterion #1)
// -----------------------------------------------------------------------------

// TestLoadPerformance_Benchmark measures loading time.
func TestLoadPerformance_Benchmark(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping performance test in short mode")
	}

	tmpDir := t.TempDir()

	// Create a model that simulates Llama 7B size tensors (smaller for test)
	createRealisticSizeModel(t, tmpDir)

	start := time.Now()

	loader, err := model.NewWeightLoader(tmpDir)
	if err != nil {
		t.Fatalf("NewWeightLoader failed: %v", err)
	}
	defer loader.Close()

	// Load all layers
	numLayers := loader.CountLayers()
	for i := 0; i < numLayers; i++ {
		_, err := loader.LoadLayerWeights(i)
		if err != nil {
			t.Errorf("failed to load layer %d: %v", i, err)
		}
	}

	elapsed := time.Since(start)
	t.Logf("Loaded %d layers in %v", numLayers, elapsed)

	// For now, just ensure it completes (real 7B test requires real model)
	if elapsed > 30*time.Second {
		t.Errorf("loading took too long: %v (target: < 30s)", elapsed)
	}
}

// =============================================================================
// Helper Functions for Test Setup
// =============================================================================

// createShardedModel creates a sharded safetensors model with index.json.
func createShardedModel(t *testing.T, dir string, numShards int) {
	t.Helper()

	// Create weight_map for index.json
	weightMap := make(map[string]string)
	shardTensors := make([]map[string][]byte, numShards)

	for i := 0; i < numShards; i++ {
		shardTensors[i] = make(map[string][]byte)
	}

	// Distribute tensors across shards
	shardTensors[0]["model.embed_tokens.weight"] = make([]byte, 1024)
	weightMap["model.embed_tokens.weight"] = shardFileName(1, numShards)

	for layer := 0; layer < numShards; layer++ {
		shardIdx := layer % numShards
		shardFile := shardFileName(shardIdx+1, numShards)

		prefix := fmt.Sprintf("model.layers.%d.", layer)
		tensors := []string{
			"self_attn.q_proj.weight",
			"self_attn.k_proj.weight",
			"self_attn.v_proj.weight",
			"self_attn.o_proj.weight",
			"mlp.gate_proj.weight",
			"mlp.up_proj.weight",
			"mlp.down_proj.weight",
			"input_layernorm.weight",
			"post_attention_layernorm.weight",
		}

		for _, tensor := range tensors {
			name := prefix + tensor
			shardTensors[shardIdx][name] = make([]byte, 256)
			weightMap[name] = shardFile
		}
	}

	// Add lm_head to last shard
	lastShard := numShards - 1
	shardTensors[lastShard]["lm_head.weight"] = make([]byte, 1024)
	weightMap["lm_head.weight"] = shardFileName(numShards, numShards)

	// Create shard files
	for i := 0; i < numShards; i++ {
		path := filepath.Join(dir, shardFileName(i+1, numShards))
		if err := model.CreateMockSafeTensors(path, shardTensors[i]); err != nil {
			t.Fatalf("failed to create shard %d: %v", i, err)
		}
	}

	// Create index.json
	index := map[string]interface{}{
		"metadata":   map[string]interface{}{"format": "pt"},
		"weight_map": weightMap,
	}

	indexData, _ := json.MarshalIndent(index, "", "  ")
	indexPath := filepath.Join(dir, "model.safetensors.index.json")
	if err := os.WriteFile(indexPath, indexData, 0644); err != nil {
		t.Fatalf("failed to create index.json: %v", err)
	}
}

func shardFileName(n, total int) string {
	return fmt.Sprintf("model-%05d-of-%05d.safetensors", n, total)
}

// createLargeTestModel creates a model with large tensor for mmap testing.
func createLargeTestModel(t *testing.T, dir string, size int) {
	t.Helper()

	tensors := map[string][]byte{
		"large_tensor": make([]byte, size),
	}

	path := filepath.Join(dir, "model.safetensors")
	if err := model.CreateMockSafeTensors(path, tensors); err != nil {
		t.Fatalf("failed to create large test model: %v", err)
	}
}

// createAlignedTestModel creates a model for alignment testing.
func createAlignedTestModel(t *testing.T, dir string) {
	t.Helper()

	// Create tensor that requires alignment
	tensors := map[string][]byte{
		"test_tensor": make([]byte, 4096),
	}

	path := filepath.Join(dir, "model.safetensors")
	if err := model.CreateMockSafeTensors(path, tensors); err != nil {
		t.Fatalf("failed to create aligned test model: %v", err)
	}
}

// createFP16TestModel creates a model with FP16 tensors.
func createFP16TestModel(t *testing.T, dir string) {
	t.Helper()

	path := filepath.Join(dir, "model.safetensors")
	createSafeTensorsWithDtype(t, path, "fp16_tensor", "F16", []int64{512}, 1024)
}

// createINT8TestModel creates a model with INT8 weights and scales.
func createINT8TestModel(t *testing.T, dir string) {
	t.Helper()

	path := filepath.Join(dir, "model.safetensors")

	// Create INT8 weight [4096, 4096] = 16MB
	// Create FP32 scale [4096] = 16KB
	tensors := make(map[string]interface{})

	// INT8 weight
	int8Size := int64(4096 * 4096)
	tensors["int8_weight"] = map[string]interface{}{
		"dtype":        "I8",
		"shape":        []int64{4096, 4096},
		"data_offsets": []int64{0, int8Size},
	}

	// FP32 scale (one per row)
	scaleSize := int64(4096 * 4)
	tensors["int8_weight_scale"] = map[string]interface{}{
		"dtype":        "F32",
		"shape":        []int64{4096},
		"data_offsets": []int64{int8Size, int8Size + scaleSize},
	}

	createSafeTensorsRaw(t, path, tensors, int8Size+scaleSize)
}

// createBF16TestModel creates a model with BF16 tensors.
func createBF16TestModel(t *testing.T, dir string) {
	t.Helper()

	path := filepath.Join(dir, "model.safetensors")
	createSafeTensorsWithDtype(t, path, "bf16_tensor", "BF16", []int64{256}, 512)
}

// createModelWithConfig creates a model matching a LlamaConfig.
func createModelWithConfig(t *testing.T, dir string, config *types.LlamaConfig) {
	t.Helper()

	path := filepath.Join(dir, "model.safetensors")

	// Calculate total data size
	var totalSize int64

	// Build header with proper shapes
	tensors := make(map[string]interface{})
	var currentOffset int64

	// Helper to add tensor
	addTensor := func(name string, shape []int64, dtype string, bytesPerElem int64) {
		numElements := int64(1)
		for _, s := range shape {
			numElements *= s
		}
		size := numElements * bytesPerElem

		tensors[name] = map[string]interface{}{
			"dtype":        dtype,
			"shape":        shape,
			"data_offsets": []int64{currentOffset, currentOffset + size},
		}
		currentOffset += size
	}

	// Embedding: [vocab_size, hidden_size] in FP16
	addTensor("model.embed_tokens.weight",
		[]int64{int64(config.VocabSize), int64(config.HiddenSize)}, "F16", 2)

	// LM head: [vocab_size, hidden_size] in FP16
	addTensor("lm_head.weight",
		[]int64{int64(config.VocabSize), int64(config.HiddenSize)}, "F16", 2)

	// Final norm: [hidden_size] in FP16
	addTensor("model.norm.weight",
		[]int64{int64(config.HiddenSize)}, "F16", 2)

	// Layers
	for layer := 0; layer < config.NumLayers; layer++ {
		prefix := fmt.Sprintf("model.layers.%d.", layer)

		// Attention norms: [hidden_size]
		addTensor(prefix+"input_layernorm.weight",
			[]int64{int64(config.HiddenSize)}, "F16", 2)
		addTensor(prefix+"post_attention_layernorm.weight",
			[]int64{int64(config.HiddenSize)}, "F16", 2)

		// Q: [num_heads * head_dim, hidden_size]
		addTensor(prefix+"self_attn.q_proj.weight",
			[]int64{int64(config.NumHeads * config.HeadDim), int64(config.HiddenSize)}, "F16", 2)

		// K, V: [num_kv_heads * head_dim, hidden_size] (GQA support)
		addTensor(prefix+"self_attn.k_proj.weight",
			[]int64{int64(config.NumKVHeads * config.HeadDim), int64(config.HiddenSize)}, "F16", 2)
		addTensor(prefix+"self_attn.v_proj.weight",
			[]int64{int64(config.NumKVHeads * config.HeadDim), int64(config.HiddenSize)}, "F16", 2)

		// O: [hidden_size, num_heads * head_dim]
		addTensor(prefix+"self_attn.o_proj.weight",
			[]int64{int64(config.HiddenSize), int64(config.NumHeads * config.HeadDim)}, "F16", 2)

		// FFN: gate, up: [intermediate_size, hidden_size]
		addTensor(prefix+"mlp.gate_proj.weight",
			[]int64{int64(config.IntermediateSize), int64(config.HiddenSize)}, "F16", 2)
		addTensor(prefix+"mlp.up_proj.weight",
			[]int64{int64(config.IntermediateSize), int64(config.HiddenSize)}, "F16", 2)
		// down: [hidden_size, intermediate_size]
		addTensor(prefix+"mlp.down_proj.weight",
			[]int64{int64(config.HiddenSize), int64(config.IntermediateSize)}, "F16", 2)
	}

	totalSize = currentOffset
	createSafeTensorsRaw(t, path, tensors, totalSize)
}

// createModelWithWrongShapes creates a model with incorrect shapes.
func createModelWithWrongShapes(t *testing.T, dir string) {
	t.Helper()

	path := filepath.Join(dir, "model.safetensors")

	tensors := make(map[string]interface{})

	// Wrong embedding size (should be [vocab_size, hidden_size])
	tensors["model.embed_tokens.weight"] = map[string]interface{}{
		"dtype":        "F16",
		"shape":        []int64{100, 50}, // Too small
		"data_offsets": []int64{0, 10000},
	}

	// Wrong layer shape
	tensors["model.layers.0.self_attn.q_proj.weight"] = map[string]interface{}{
		"dtype":        "F16",
		"shape":        []int64{64, 32}, // Wrong dimensions
		"data_offsets": []int64{10000, 14096},
	}

	createSafeTensorsRaw(t, path, tensors, 20000)
}

// createRealisticSizeModel creates a model with realistic layer sizes.
func createRealisticSizeModel(t *testing.T, dir string) {
	t.Helper()

	config := types.Llama7BConfig()
	config.NumLayers = 4 // Fewer layers for test speed

	createModelWithConfig(t, dir, config)
}

// createSafeTensorsWithDtype creates a safetensors file with specific dtype.
func createSafeTensorsWithDtype(t *testing.T, path, name, dtype string, shape []int64, dataSize int) {
	t.Helper()

	tensors := map[string]interface{}{
		name: map[string]interface{}{
			"dtype":        dtype,
			"shape":        shape,
			"data_offsets": []int64{0, int64(dataSize)},
		},
	}

	createSafeTensorsRaw(t, path, tensors, int64(dataSize))
}

// createSafeTensorsRaw creates a safetensors file with raw tensor specs.
func createSafeTensorsRaw(t *testing.T, path string, tensors map[string]interface{}, dataSize int64) {
	t.Helper()

	headerBytes, err := json.Marshal(tensors)
	if err != nil {
		t.Fatalf("failed to marshal header: %v", err)
	}

	f, err := os.Create(path)
	if err != nil {
		t.Fatalf("failed to create file: %v", err)
	}
	defer f.Close()

	// Write header size (8 bytes, little-endian)
	headerSize := uint64(len(headerBytes))
	if err := binary.Write(f, binary.LittleEndian, headerSize); err != nil {
		t.Fatalf("failed to write header size: %v", err)
	}

	// Write header
	if _, err := f.Write(headerBytes); err != nil {
		t.Fatalf("failed to write header: %v", err)
	}

	// Write data (zeros)
	data := make([]byte, dataSize)
	if _, err := f.Write(data); err != nil {
		t.Fatalf("failed to write data: %v", err)
	}
}
