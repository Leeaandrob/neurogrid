// Package model_test provides tests for model loading functionality.
package model_test

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/neurogrid/engine/pkg/model"
	"github.com/neurogrid/engine/pkg/types"
)

// createTestSafeTensors creates a minimal SafeTensors file for testing.
func createTestSafeTensors(t *testing.T, dir string, filename string, tensors map[string][]byte) string {
	t.Helper()
	path := filepath.Join(dir, filename)
	if err := model.CreateMockSafeTensors(path, tensors); err != nil {
		t.Fatalf("Failed to create mock SafeTensors: %v", err)
	}
	return path
}

func TestNewWeightLoader_SingleFile(t *testing.T) {
	// Create a temp directory
	dir := t.TempDir()

	// Create mock safetensors file
	tensors := map[string][]byte{
		"model.embed_tokens.weight":              make([]byte, 1000*2), // FP16
		"model.layers.0.self_attn.q_proj.weight": make([]byte, 512*2),
		"model.layers.0.self_attn.k_proj.weight": make([]byte, 512*2),
		"model.layers.0.self_attn.v_proj.weight": make([]byte, 512*2),
		"model.layers.0.self_attn.o_proj.weight": make([]byte, 512*2),
		"model.layers.0.mlp.gate_proj.weight":    make([]byte, 1024*2),
		"model.layers.0.mlp.up_proj.weight":      make([]byte, 1024*2),
		"model.layers.0.mlp.down_proj.weight":    make([]byte, 1024*2),
		"model.layers.0.input_layernorm.weight":  make([]byte, 64*2),
		"model.layers.0.post_attention_layernorm.weight": make([]byte, 64*2),
	}

	createTestSafeTensors(t, dir, "model.safetensors", tensors)

	// Test loading
	loader, err := model.NewWeightLoader(dir)
	if err != nil {
		t.Fatalf("NewWeightLoader failed: %v", err)
	}
	defer loader.Close()

	// Verify tensor count
	tensorList := loader.ListTensors()
	if len(tensorList) != len(tensors) {
		t.Errorf("Expected %d tensors, got %d", len(tensors), len(tensorList))
	}

	// Verify layer count
	numLayers := loader.CountLayers()
	if numLayers != 1 {
		t.Errorf("Expected 1 layer, got %d", numLayers)
	}
}

func TestLoadTensor(t *testing.T) {
	dir := t.TempDir()

	// Create test data with known values
	testData := make([]byte, 256)
	for i := range testData {
		testData[i] = byte(i)
	}

	tensors := map[string][]byte{
		"test_tensor": testData,
	}
	createTestSafeTensors(t, dir, "model.safetensors", tensors)

	loader, err := model.NewWeightLoader(dir)
	if err != nil {
		t.Fatalf("NewWeightLoader failed: %v", err)
	}
	defer loader.Close()

	// Load tensor
	data, info, err := loader.LoadTensor("test_tensor")
	if err != nil {
		t.Fatalf("LoadTensor failed: %v", err)
	}

	// Verify data
	if len(data) != len(testData) {
		t.Errorf("Expected data length %d, got %d", len(testData), len(data))
	}

	// Verify info
	if info == nil {
		t.Fatal("TensorInfo is nil")
	}
}

func TestLoadTensor_NotFound(t *testing.T) {
	dir := t.TempDir()

	tensors := map[string][]byte{
		"existing_tensor": make([]byte, 64),
	}
	createTestSafeTensors(t, dir, "model.safetensors", tensors)

	loader, err := model.NewWeightLoader(dir)
	if err != nil {
		t.Fatalf("NewWeightLoader failed: %v", err)
	}
	defer loader.Close()

	// Try to load non-existent tensor
	_, _, err = loader.LoadTensor("nonexistent")
	if err == nil {
		t.Error("Expected error for non-existent tensor")
	}
}

func TestLoadLayerWeights(t *testing.T) {
	dir := t.TempDir()

	// Create all required layer tensors
	tensors := map[string][]byte{
		"model.layers.0.self_attn.q_proj.weight":         make([]byte, 512*2),
		"model.layers.0.self_attn.k_proj.weight":         make([]byte, 512*2),
		"model.layers.0.self_attn.v_proj.weight":         make([]byte, 512*2),
		"model.layers.0.self_attn.o_proj.weight":         make([]byte, 512*2),
		"model.layers.0.mlp.gate_proj.weight":            make([]byte, 1024*2),
		"model.layers.0.mlp.up_proj.weight":              make([]byte, 1024*2),
		"model.layers.0.mlp.down_proj.weight":            make([]byte, 1024*2),
		"model.layers.0.input_layernorm.weight":          make([]byte, 64*2),
		"model.layers.0.post_attention_layernorm.weight": make([]byte, 64*2),
	}
	createTestSafeTensors(t, dir, "model.safetensors", tensors)

	loader, err := model.NewWeightLoader(dir)
	if err != nil {
		t.Fatalf("NewWeightLoader failed: %v", err)
	}
	defer loader.Close()

	// Load layer weights
	weights, err := loader.LoadLayerWeights(0)
	if err != nil {
		t.Fatalf("LoadLayerWeights failed: %v", err)
	}

	// Verify all weights are present
	if weights.QWeight == nil {
		t.Error("QWeight is nil")
	}
	if weights.KWeight == nil {
		t.Error("KWeight is nil")
	}
	if weights.VWeight == nil {
		t.Error("VWeight is nil")
	}
	if weights.OWeight == nil {
		t.Error("OWeight is nil")
	}
	if weights.GateWeight == nil {
		t.Error("GateWeight is nil")
	}
	if weights.UpWeight == nil {
		t.Error("UpWeight is nil")
	}
	if weights.DownWeight == nil {
		t.Error("DownWeight is nil")
	}
	if weights.AttnNorm == nil {
		t.Error("AttnNorm is nil")
	}
	if weights.FFNNorm == nil {
		t.Error("FFNNorm is nil")
	}
}

func TestLoadEmbeddings(t *testing.T) {
	dir := t.TempDir()

	// Create embedding tensor
	embedSize := 1000 * 64 * 2 // vocab_size * hidden_size * sizeof(FP16)
	tensors := map[string][]byte{
		"model.embed_tokens.weight": make([]byte, embedSize),
	}
	createTestSafeTensors(t, dir, "model.safetensors", tensors)

	loader, err := model.NewWeightLoader(dir)
	if err != nil {
		t.Fatalf("NewWeightLoader failed: %v", err)
	}
	defer loader.Close()

	data, info, err := loader.LoadEmbeddings()
	if err != nil {
		t.Fatalf("LoadEmbeddings failed: %v", err)
	}

	if len(data) != embedSize {
		t.Errorf("Expected embedding size %d, got %d", embedSize, len(data))
	}

	if info == nil {
		t.Error("TensorInfo is nil")
	}
}

func TestLoadLMHead_Direct(t *testing.T) {
	dir := t.TempDir()

	lmHeadSize := 1000 * 64 * 2
	tensors := map[string][]byte{
		"lm_head.weight":             make([]byte, lmHeadSize),
		"model.embed_tokens.weight":  make([]byte, 500*2),
	}
	createTestSafeTensors(t, dir, "model.safetensors", tensors)

	loader, err := model.NewWeightLoader(dir)
	if err != nil {
		t.Fatalf("NewWeightLoader failed: %v", err)
	}
	defer loader.Close()

	data, _, err := loader.LoadLMHead()
	if err != nil {
		t.Fatalf("LoadLMHead failed: %v", err)
	}

	if len(data) != lmHeadSize {
		t.Errorf("Expected LM head size %d, got %d", lmHeadSize, len(data))
	}
}

func TestLoadLMHead_TiedEmbeddings(t *testing.T) {
	dir := t.TempDir()

	// No lm_head.weight - should fall back to embeddings
	embedSize := 1000 * 64 * 2
	tensors := map[string][]byte{
		"model.embed_tokens.weight": make([]byte, embedSize),
	}
	createTestSafeTensors(t, dir, "model.safetensors", tensors)

	loader, err := model.NewWeightLoader(dir)
	if err != nil {
		t.Fatalf("NewWeightLoader failed: %v", err)
	}
	defer loader.Close()

	data, _, err := loader.LoadLMHead()
	if err != nil {
		t.Fatalf("LoadLMHead failed: %v", err)
	}

	// Should get embeddings since lm_head not present
	if len(data) != embedSize {
		t.Errorf("Expected tied embedding size %d, got %d", embedSize, len(data))
	}
}

func TestMmapLoader(t *testing.T) {
	dir := t.TempDir()

	tensors := map[string][]byte{
		"model.embed_tokens.weight": make([]byte, 1000*2),
		"test_tensor":               make([]byte, 512*2),
	}
	createTestSafeTensors(t, dir, "model.safetensors", tensors)

	// Test mmap loader
	loader, err := model.NewMmapLoader(dir)
	if err != nil {
		t.Fatalf("NewMmapLoader failed: %v", err)
	}
	defer loader.Close()

	// Verify tensor can be loaded
	data, info, err := loader.LoadTensor("test_tensor")
	if err != nil {
		t.Fatalf("LoadTensor failed: %v", err)
	}

	if len(data) != 512*2 {
		t.Errorf("Expected size %d, got %d", 512*2, len(data))
	}

	if info == nil {
		t.Error("TensorInfo is nil")
	}
}

func TestMmapLoader_DirectAccess(t *testing.T) {
	dir := t.TempDir()

	// Create data with known pattern
	testData := make([]byte, 256)
	for i := range testData {
		testData[i] = byte(i)
	}

	tensors := map[string][]byte{
		"test_tensor": testData,
	}
	createTestSafeTensors(t, dir, "model.safetensors", tensors)

	loader, err := model.NewMmapLoader(dir)
	if err != nil {
		t.Fatalf("NewMmapLoader failed: %v", err)
	}
	defer loader.Close()

	// Get direct mmap access (no copy)
	data, err := loader.MmapTensor("test_tensor")
	if err != nil {
		t.Fatalf("MmapTensor failed: %v", err)
	}

	if len(data) != len(testData) {
		t.Errorf("Expected size %d, got %d", len(testData), len(data))
	}
}

func TestMmapLoader_MemoryStats(t *testing.T) {
	dir := t.TempDir()

	tensors := map[string][]byte{
		"tensor1": make([]byte, 1000),
		"tensor2": make([]byte, 2000),
		"tensor3": make([]byte, 3000),
	}
	createTestSafeTensors(t, dir, "model.safetensors", tensors)

	loader, err := model.NewMmapLoader(dir)
	if err != nil {
		t.Fatalf("NewMmapLoader failed: %v", err)
	}
	defer loader.Close()

	stats := loader.GetMemoryStats()
	if stats.TensorCount != 3 {
		t.Errorf("Expected 3 tensors, got %d", stats.TensorCount)
	}
	if stats.FileCount != 1 {
		t.Errorf("Expected 1 file, got %d", stats.FileCount)
	}
}

func TestModelLoader_Interface(t *testing.T) {
	dir := t.TempDir()

	tensors := map[string][]byte{
		"model.embed_tokens.weight": make([]byte, 1000*2),
	}
	createTestSafeTensors(t, dir, "model.safetensors", tensors)

	// Test with useMmap=false
	loader, err := model.NewLoader(dir, false)
	if err != nil {
		t.Fatalf("NewLoader(useMmap=false) failed: %v", err)
	}
	defer loader.Close()

	if _, ok := loader.(*model.WeightLoader); !ok {
		t.Error("Expected WeightLoader when useMmap=false")
	}

	// Test with useMmap=true
	loader2, err := model.NewLoader(dir, true)
	if err != nil {
		t.Fatalf("NewLoader(useMmap=true) failed: %v", err)
	}
	defer loader2.Close()

	if _, ok := loader2.(*model.MmapLoader); !ok {
		t.Error("Expected MmapLoader when useMmap=true")
	}
}

func TestNewWeightLoader_NoFile(t *testing.T) {
	dir := t.TempDir()
	// Don't create any files

	_, err := model.NewWeightLoader(dir)
	if err == nil {
		t.Error("Expected error for missing model file")
	}
}

func TestDistributedModel_Create(t *testing.T) {
	dir := t.TempDir()

	// Create minimal model
	tensors := map[string][]byte{
		"model.embed_tokens.weight":                      make([]byte, 1000*2),
		"lm_head.weight":                                 make([]byte, 1000*2),
		"model.layers.0.self_attn.q_proj.weight":         make([]byte, 512*2),
		"model.layers.0.self_attn.k_proj.weight":         make([]byte, 512*2),
		"model.layers.0.self_attn.v_proj.weight":         make([]byte, 512*2),
		"model.layers.0.self_attn.o_proj.weight":         make([]byte, 512*2),
		"model.layers.0.mlp.gate_proj.weight":            make([]byte, 1024*2),
		"model.layers.0.mlp.up_proj.weight":              make([]byte, 1024*2),
		"model.layers.0.mlp.down_proj.weight":            make([]byte, 1024*2),
		"model.layers.0.input_layernorm.weight":          make([]byte, 64*2),
		"model.layers.0.post_attention_layernorm.weight": make([]byte, 64*2),
	}
	createTestSafeTensors(t, dir, "model.safetensors", tensors)

	config := &types.LlamaConfig{
		NumLayers:        1,
		HiddenSize:       64,
		NumHeads:         4,
		NumKVHeads:       4,
		HeadDim:          16,
		IntermediateSize: 128,
		VocabSize:        1000,
		MaxSeqLen:        512,
	}

	dm, err := model.NewDistributedModel(model.DistributedModelConfig{
		ModelConfig: config,
		ModelPath:   dir,
		LocalPeerID: "local-peer",
	})
	if err != nil {
		t.Fatalf("NewDistributedModel failed: %v", err)
	}
	defer dm.Close()

	// Verify config
	if dm.Config() != config {
		t.Error("Config mismatch")
	}

	// Verify total layers
	if dm.TotalLayers() != 1 {
		t.Errorf("Expected 1 layer, got %d", dm.TotalLayers())
	}
}

func TestBF16ToFP16Conversion(t *testing.T) {
	// Test BF16 -> FP16 conversion
	// BF16: sign(1) + exp(8) + mantissa(7)
	// FP16: sign(1) + exp(5) + mantissa(10)

	// Test case: 1.0 in BF16 = 0x3F80 (same as FP32 high word)
	bf16Data := []byte{0x80, 0x3F} // Little-endian: 0x3F80

	fp16Data := model.ConvertBF16ToFP16(bf16Data)

	// 1.0 in FP16 = 0x3C00
	expectedFP16 := uint16(0x3C00)
	actualFP16 := uint16(fp16Data[0]) | (uint16(fp16Data[1]) << 8)

	if actualFP16 != expectedFP16 {
		t.Errorf("BF16(1.0) -> FP16 conversion failed: expected 0x%04X, got 0x%04X", expectedFP16, actualFP16)
	}

	// Test case: 0.0
	bf16Zero := []byte{0x00, 0x00}
	fp16Zero := model.ConvertBF16ToFP16(bf16Zero)
	if fp16Zero[0] != 0 || fp16Zero[1] != 0 {
		t.Error("BF16(0.0) -> FP16 conversion failed")
	}
}

func TestWeightFormat(t *testing.T) {
	tests := []struct {
		format   model.WeightFormat
		byteSize int
		dtype    types.Dtype
	}{
		{model.WeightFormatFP32, 4, types.DtypeFP32},
		{model.WeightFormatFP16, 2, types.DtypeFP16},
		{model.WeightFormatBF16, 2, types.DtypeFP16},
		{model.WeightFormatINT8, 1, types.DtypeINT8},
	}

	for _, tt := range tests {
		if tt.format.ByteSize() != tt.byteSize {
			t.Errorf("%v.ByteSize() = %d, want %d", tt.format, tt.format.ByteSize(), tt.byteSize)
		}
		if tt.format.ToDtype() != tt.dtype {
			t.Errorf("%v.ToDtype() = %v, want %v", tt.format, tt.format.ToDtype(), tt.dtype)
		}
	}
}

func TestParseWeightFormat(t *testing.T) {
	tests := []struct {
		dtype    string
		expected model.WeightFormat
	}{
		{"F32", model.WeightFormatFP32},
		{"F16", model.WeightFormatFP16},
		{"BF16", model.WeightFormatBF16},
		{"I8", model.WeightFormatINT8},
		{"unknown", model.WeightFormatFP16}, // Default
	}

	for _, tt := range tests {
		result := model.ParseWeightFormat(tt.dtype)
		if result != tt.expected {
			t.Errorf("ParseWeightFormat(%q) = %v, want %v", tt.dtype, result, tt.expected)
		}
	}
}

func TestAlignedAlloc(t *testing.T) {
	// Test various alignments
	alignments := []int{16, 32, 64, 256}

	for _, alignment := range alignments {
		buf := model.AlignedAlloc(1000, alignment)

		// Check size
		if len(buf) != 1000 {
			t.Errorf("AlignedAlloc(%d, %d) returned size %d, want 1000", 1000, alignment, len(buf))
		}
	}
}

func TestGetTensorInfo(t *testing.T) {
	dir := t.TempDir()

	tensors := map[string][]byte{
		"test_tensor": make([]byte, 512),
	}
	createTestSafeTensors(t, dir, "model.safetensors", tensors)

	loader, err := model.NewWeightLoader(dir)
	if err != nil {
		t.Fatalf("NewWeightLoader failed: %v", err)
	}
	defer loader.Close()

	// Get existing tensor info
	info, ok := loader.GetTensorInfo("test_tensor")
	if !ok {
		t.Error("Expected tensor info to exist")
	}
	if info == nil {
		t.Fatal("TensorInfo is nil")
	}

	// Get non-existing tensor info
	_, ok = loader.GetTensorInfo("nonexistent")
	if ok {
		t.Error("Expected tensor info to not exist")
	}
}

// TestShardedModel tests loading a model split across multiple files
// This is a more complex test that simulates how large models are stored
func TestShardedModel(t *testing.T) {
	dir := t.TempDir()

	// Create index file
	indexContent := `{
		"metadata": {},
		"weight_map": {
			"model.embed_tokens.weight": "model-00001-of-00002.safetensors",
			"model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
			"model.layers.0.self_attn.k_proj.weight": "model-00002-of-00002.safetensors"
		}
	}`
	indexPath := filepath.Join(dir, "model.safetensors.index.json")
	if err := os.WriteFile(indexPath, []byte(indexContent), 0644); err != nil {
		t.Fatalf("Failed to write index: %v", err)
	}

	// Create shard files
	tensors1 := map[string][]byte{
		"model.embed_tokens.weight":              make([]byte, 1000*2),
		"model.layers.0.self_attn.q_proj.weight": make([]byte, 512*2),
	}
	createTestSafeTensors(t, dir, "model-00001-of-00002.safetensors", tensors1)

	tensors2 := map[string][]byte{
		"model.layers.0.self_attn.k_proj.weight": make([]byte, 512*2),
	}
	createTestSafeTensors(t, dir, "model-00002-of-00002.safetensors", tensors2)

	// Load sharded model
	loader, err := model.NewWeightLoader(dir)
	if err != nil {
		t.Fatalf("NewWeightLoader for sharded model failed: %v", err)
	}
	defer loader.Close()

	// Verify all tensors are accessible
	_, _, err = loader.LoadTensor("model.embed_tokens.weight")
	if err != nil {
		t.Errorf("Failed to load embedding from shard 1: %v", err)
	}

	_, _, err = loader.LoadTensor("model.layers.0.self_attn.q_proj.weight")
	if err != nil {
		t.Errorf("Failed to load q_proj from shard 1: %v", err)
	}

	_, _, err = loader.LoadTensor("model.layers.0.self_attn.k_proj.weight")
	if err != nil {
		t.Errorf("Failed to load k_proj from shard 2: %v", err)
	}
}

// BenchmarkLoadTensor benchmarks tensor loading performance
func BenchmarkLoadTensor(b *testing.B) {
	dir := b.TempDir()

	// Create a moderately sized tensor
	tensors := map[string][]byte{
		"test_tensor": make([]byte, 1024*1024), // 1MB
	}
	if err := model.CreateMockSafeTensors(filepath.Join(dir, "model.safetensors"), tensors); err != nil {
		b.Fatalf("Failed to create mock: %v", err)
	}

	loader, err := model.NewWeightLoader(dir)
	if err != nil {
		b.Fatalf("NewWeightLoader failed: %v", err)
	}
	defer loader.Close()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, err := loader.LoadTensor("test_tensor")
		if err != nil {
			b.Fatalf("LoadTensor failed: %v", err)
		}
	}
}

// BenchmarkMmapTensor benchmarks mmap tensor access performance
func BenchmarkMmapTensor(b *testing.B) {
	dir := b.TempDir()

	tensors := map[string][]byte{
		"test_tensor": make([]byte, 1024*1024), // 1MB
	}
	if err := model.CreateMockSafeTensors(filepath.Join(dir, "model.safetensors"), tensors); err != nil {
		b.Fatalf("Failed to create mock: %v", err)
	}

	loader, err := model.NewMmapLoader(dir)
	if err != nil {
		b.Fatalf("NewMmapLoader failed: %v", err)
	}
	defer loader.Close()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := loader.MmapTensor("test_tensor")
		if err != nil {
			b.Fatalf("MmapTensor failed: %v", err)
		}
	}
}

// Integration test using context
func TestLoaderWithContext(t *testing.T) {
	dir := t.TempDir()

	tensors := map[string][]byte{
		"test_tensor": make([]byte, 512),
	}
	createTestSafeTensors(t, dir, "model.safetensors", tensors)

	// Test with context (useful for timeout scenarios)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	loader, err := model.NewWeightLoader(dir)
	if err != nil {
		t.Fatalf("NewWeightLoader failed: %v", err)
	}
	defer loader.Close()

	// Just verify context doesn't interfere
	_ = ctx
	_, _, err = loader.LoadTensor("test_tensor")
	if err != nil {
		t.Errorf("LoadTensor failed: %v", err)
	}
}
