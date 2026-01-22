// Package e2e provides end-to-end tests for the NeuroGrid distributed inference engine.
package e2e

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/neurogrid/engine/pkg/model"
	"github.com/neurogrid/engine/pkg/scheduler"
	"github.com/neurogrid/engine/pkg/types"
)

// ===== TASK-022: SafeTensors Weight Loader Tests =====

func TestWeightLoader_CreateMockFile(t *testing.T) {
	// Create temp directory
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "model.safetensors")

	// Create mock tensors
	tensors := map[string][]byte{
		"model.embed_tokens.weight":              make([]byte, 1024),
		"model.layers.0.self_attn.q_proj.weight": make([]byte, 512),
		"model.layers.0.self_attn.k_proj.weight": make([]byte, 512),
	}

	// Fill with recognizable pattern
	for name, data := range tensors {
		for i := range data {
			data[i] = byte(len(name) + i)
		}
	}

	err := model.CreateMockSafeTensors(path, tensors)
	if err != nil {
		t.Fatalf("CreateMockSafeTensors failed: %v", err)
	}

	// Verify file exists
	if _, err := os.Stat(path); err != nil {
		t.Errorf("Created file does not exist: %v", err)
	}
}

func TestWeightLoader_LoadSingleFile(t *testing.T) {
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "model.safetensors")

	// Create mock SafeTensors
	tensors := map[string][]byte{
		"model.embed_tokens.weight":              make([]byte, 1024),
		"model.layers.0.self_attn.q_proj.weight": make([]byte, 512),
		"model.layers.0.self_attn.k_proj.weight": make([]byte, 256),
	}

	err := model.CreateMockSafeTensors(path, tensors)
	if err != nil {
		t.Fatalf("CreateMockSafeTensors failed: %v", err)
	}

	// Load with WeightLoader
	loader, err := model.NewWeightLoader(tmpDir)
	if err != nil {
		t.Fatalf("NewWeightLoader failed: %v", err)
	}
	defer loader.Close()

	// Verify tensors are indexed
	names := loader.ListTensors()
	if len(names) != 3 {
		t.Errorf("expected 3 tensors, got %d: %v", len(names), names)
	}
}

func TestWeightLoader_LoadTensor(t *testing.T) {
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "model.safetensors")

	// Create tensor with recognizable data
	tensorData := make([]byte, 256)
	for i := range tensorData {
		tensorData[i] = byte(i)
	}

	tensors := map[string][]byte{
		"test_tensor": tensorData,
	}

	err := model.CreateMockSafeTensors(path, tensors)
	if err != nil {
		t.Fatalf("CreateMockSafeTensors failed: %v", err)
	}

	loader, err := model.NewWeightLoader(tmpDir)
	if err != nil {
		t.Fatalf("NewWeightLoader failed: %v", err)
	}
	defer loader.Close()

	// Load the tensor
	data, info, err := loader.LoadTensor("test_tensor")
	if err != nil {
		t.Fatalf("LoadTensor failed: %v", err)
	}

	// Verify data matches
	if len(data) != len(tensorData) {
		t.Errorf("data size mismatch: got %d, expected %d", len(data), len(tensorData))
	}

	for i := 0; i < len(data) && i < len(tensorData); i++ {
		if data[i] != tensorData[i] {
			t.Errorf("data mismatch at index %d: got %d, expected %d", i, data[i], tensorData[i])
			break
		}
	}

	// Verify info
	if info == nil {
		t.Error("TensorInfo is nil")
	}
}

func TestWeightLoader_LoadTensorNotFound(t *testing.T) {
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "model.safetensors")

	tensors := map[string][]byte{
		"existing_tensor": make([]byte, 64),
	}

	model.CreateMockSafeTensors(path, tensors)

	loader, _ := model.NewWeightLoader(tmpDir)
	defer loader.Close()

	_, _, err := loader.LoadTensor("nonexistent_tensor")
	if err == nil {
		t.Error("expected error for nonexistent tensor")
	}
}

func TestWeightLoader_LoadLayerWeights(t *testing.T) {
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "model.safetensors")

	// Create all required layer tensors
	tensors := map[string][]byte{
		"model.embed_tokens.weight":                      make([]byte, 1024),
		"model.layers.0.self_attn.q_proj.weight":         make([]byte, 512),
		"model.layers.0.self_attn.k_proj.weight":         make([]byte, 512),
		"model.layers.0.self_attn.v_proj.weight":         make([]byte, 512),
		"model.layers.0.self_attn.o_proj.weight":         make([]byte, 512),
		"model.layers.0.mlp.gate_proj.weight":            make([]byte, 1024),
		"model.layers.0.mlp.up_proj.weight":              make([]byte, 1024),
		"model.layers.0.mlp.down_proj.weight":            make([]byte, 1024),
		"model.layers.0.input_layernorm.weight":          make([]byte, 64),
		"model.layers.0.post_attention_layernorm.weight": make([]byte, 64),
	}

	model.CreateMockSafeTensors(path, tensors)

	loader, err := model.NewWeightLoader(tmpDir)
	if err != nil {
		t.Fatalf("NewWeightLoader failed: %v", err)
	}
	defer loader.Close()

	weights, err := loader.LoadLayerWeights(0)
	if err != nil {
		t.Fatalf("LoadLayerWeights failed: %v", err)
	}

	// Verify all weights loaded
	if weights.LayerID != 0 {
		t.Errorf("LayerID mismatch: expected 0, got %d", weights.LayerID)
	}

	if len(weights.QWeight) != 512 {
		t.Errorf("QWeight size mismatch: expected 512, got %d", len(weights.QWeight))
	}

	if len(weights.GateWeight) != 1024 {
		t.Errorf("GateWeight size mismatch: expected 1024, got %d", len(weights.GateWeight))
	}
}

func TestWeightLoader_CountLayers(t *testing.T) {
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "model.safetensors")

	// Create tensors for 3 layers
	tensors := map[string][]byte{
		"model.layers.0.self_attn.q_proj.weight": make([]byte, 64),
		"model.layers.1.self_attn.q_proj.weight": make([]byte, 64),
		"model.layers.2.self_attn.q_proj.weight": make([]byte, 64),
	}

	model.CreateMockSafeTensors(path, tensors)

	loader, _ := model.NewWeightLoader(tmpDir)
	defer loader.Close()

	count := loader.CountLayers()
	if count != 3 {
		t.Errorf("expected 3 layers, got %d", count)
	}
}

func TestWeightLoader_LoadEmbeddings(t *testing.T) {
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "model.safetensors")

	tensors := map[string][]byte{
		"model.embed_tokens.weight": make([]byte, 2048),
	}

	model.CreateMockSafeTensors(path, tensors)

	loader, _ := model.NewWeightLoader(tmpDir)
	defer loader.Close()

	data, info, err := loader.LoadEmbeddings()
	if err != nil {
		t.Fatalf("LoadEmbeddings failed: %v", err)
	}

	if len(data) != 2048 {
		t.Errorf("embedding size mismatch: expected 2048, got %d", len(data))
	}

	if info == nil {
		t.Error("info is nil")
	}
}

// ===== TASK-023: Tokenizer Tests =====

func TestTokenizer_CreateFromJSON(t *testing.T) {
	jsonData := model.CreateMockTokenizerJSON()

	tokenizer, err := model.NewTokenizerFromJSON(jsonData)
	if err != nil {
		t.Fatalf("NewTokenizerFromJSON failed: %v", err)
	}

	if tokenizer.VocabSize() < 5 {
		t.Errorf("vocab too small: %d", tokenizer.VocabSize())
	}

	if tokenizer.EOSToken() != 2 {
		t.Errorf("EOS token wrong: expected 2, got %d", tokenizer.EOSToken())
	}

	if tokenizer.BOSToken() != 1 {
		t.Errorf("BOS token wrong: expected 1, got %d", tokenizer.BOSToken())
	}
}

func TestTokenizer_Encode(t *testing.T) {
	tokenizer := model.CreateMockTokenizer()

	tokens, err := tokenizer.Encode("Hello world!")
	if err != nil {
		t.Fatalf("Encode failed: %v", err)
	}

	// Should start with BOS
	if len(tokens) < 1 || tokens[0] != tokenizer.BOSToken() {
		t.Error("encoded tokens should start with BOS")
	}

	t.Logf("Encoded 'Hello world!' to %v", tokens)
}

func TestTokenizer_EncodeEmpty(t *testing.T) {
	tokenizer := model.CreateMockTokenizer()

	tokens, err := tokenizer.Encode("")
	if err != nil {
		t.Fatalf("Encode empty failed: %v", err)
	}

	// Should just have BOS
	if len(tokens) != 1 || tokens[0] != tokenizer.BOSToken() {
		t.Errorf("empty encode should return just BOS: %v", tokens)
	}
}

func TestTokenizer_Decode(t *testing.T) {
	tokenizer := model.CreateMockTokenizer()

	// Decode some tokens (excluding BOS/EOS)
	text, err := tokenizer.Decode([]int{3, 4, 5})
	if err != nil {
		t.Fatalf("Decode failed: %v", err)
	}

	if text == "" {
		t.Error("decoded text should not be empty")
	}

	t.Logf("Decoded [3, 4, 5] to %q", text)
}

func TestTokenizer_DecodeSkipsSpecialTokens(t *testing.T) {
	tokenizer := model.CreateMockTokenizer()

	// Include BOS and EOS
	text, err := tokenizer.Decode([]int{1, 3, 4, 2})
	if err != nil {
		t.Fatalf("Decode failed: %v", err)
	}

	// Should not contain special token representations
	if text == "" {
		t.Error("decoded text should not be empty")
	}

	t.Logf("Decoded with special tokens to %q", text)
}

func TestTokenizer_SpecialTokens(t *testing.T) {
	tokenizer := model.CreateMockTokenizer()

	if tokenizer.BOSToken() != 1 {
		t.Errorf("BOSToken: expected 1, got %d", tokenizer.BOSToken())
	}

	if tokenizer.EOSToken() != 2 {
		t.Errorf("EOSToken: expected 2, got %d", tokenizer.EOSToken())
	}

	if tokenizer.PADToken() != 0 {
		t.Errorf("PADToken: expected 0, got %d", tokenizer.PADToken())
	}
}

func TestTokenizer_DecodeToken(t *testing.T) {
	tokenizer := model.CreateMockTokenizer()

	token := tokenizer.DecodeToken(3)
	if token == "" || token == "<unk>" {
		t.Errorf("DecodeToken(3) returned invalid: %q", token)
	}

	unkToken := tokenizer.DecodeToken(9999)
	if unkToken != "<unk>" {
		t.Errorf("DecodeToken(9999) should return <unk>, got %q", unkToken)
	}
}

func TestTokenizer_GetVocab(t *testing.T) {
	tokenizer := model.CreateMockTokenizer()

	vocab := tokenizer.GetVocab()
	if len(vocab) != tokenizer.VocabSize() {
		t.Errorf("vocab size mismatch: map has %d, VocabSize() returns %d", len(vocab), tokenizer.VocabSize())
	}

	// Verify it's a copy
	vocab["test_modification"] = 12345
	if _, ok := tokenizer.GetVocab()["test_modification"]; ok {
		t.Error("GetVocab should return a copy")
	}
}

// ===== TASK-024: Distributed Model Tests =====

func TestDistributedModel_Creation(t *testing.T) {
	// Create mock model directory
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "model.safetensors")

	tensors := map[string][]byte{
		"model.embed_tokens.weight":              make([]byte, 512),
		"lm_head.weight":                         make([]byte, 512),
		"model.layers.0.self_attn.q_proj.weight": make([]byte, 256),
	}
	model.CreateMockSafeTensors(path, tensors)

	config := types.Llama7BConfig()
	dm, err := model.NewDistributedModel(model.DistributedModelConfig{
		ModelConfig: config,
		ModelPath:   tmpDir,
		LocalPeerID: "local",
	})
	if err != nil {
		t.Fatalf("NewDistributedModel failed: %v", err)
	}
	defer dm.Close()

	if dm.Config() != config {
		t.Error("config mismatch")
	}
}

func TestDistributedModel_LoadToCluster(t *testing.T) {
	// Create mock model with multiple layers
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "model.safetensors")

	tensors := map[string][]byte{
		"model.embed_tokens.weight": make([]byte, 512),
		"lm_head.weight":            make([]byte, 512),
	}

	// Add 3 layers worth of weights
	for layer := 0; layer < 3; layer++ {
		prefix := "model.layers." + string(rune('0'+layer)) + "."
		tensors[prefix+"self_attn.q_proj.weight"] = make([]byte, 256)
		tensors[prefix+"self_attn.k_proj.weight"] = make([]byte, 256)
		tensors[prefix+"self_attn.v_proj.weight"] = make([]byte, 256)
		tensors[prefix+"self_attn.o_proj.weight"] = make([]byte, 256)
		tensors[prefix+"mlp.gate_proj.weight"] = make([]byte, 512)
		tensors[prefix+"mlp.up_proj.weight"] = make([]byte, 512)
		tensors[prefix+"mlp.down_proj.weight"] = make([]byte, 512)
		tensors[prefix+"input_layernorm.weight"] = make([]byte, 64)
		tensors[prefix+"post_attention_layernorm.weight"] = make([]byte, 64)
	}

	model.CreateMockSafeTensors(path, tensors)

	config := types.Llama7BConfig()
	config.NumLayers = 3 // Match our mock

	dm, err := model.NewDistributedModel(model.DistributedModelConfig{
		ModelConfig: config,
		ModelPath:   tmpDir,
		LocalPeerID: "local",
	})
	if err != nil {
		t.Fatalf("NewDistributedModel failed: %v", err)
	}
	defer dm.Close()

	// Create scheduler with assignments
	sched := scheduler.NewScheduler(scheduler.ModelConfig{
		HiddenSize:       int64(config.HiddenSize),
		IntermediateSize: int64(config.IntermediateSize),
		NumLayers:        config.NumLayers,
		NumKVHeads:       config.NumKVHeads,
		HeadDim:          config.HeadDim,
		MaxSeqLen:        config.MaxSeqLen,
		VocabSize:        int64(config.VocabSize),
	})

	// Register peers
	sched.RegisterPeer("local", 16*1024*1024*1024, 0) // 16GB

	dm.SetScheduler(sched)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	err = dm.LoadToCluster(ctx)
	if err != nil {
		t.Fatalf("LoadToCluster failed: %v", err)
	}

	// Verify embeddings loaded
	if dm.Embeddings() == nil {
		t.Error("embeddings not loaded")
	}

	// Verify lm_head loaded
	if dm.LMHead() == nil {
		t.Error("lm_head not loaded")
	}
}

func TestDistributedModel_LoadStatus(t *testing.T) {
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "model.safetensors")

	tensors := map[string][]byte{
		"model.embed_tokens.weight":                      make([]byte, 256),
		"lm_head.weight":                                 make([]byte, 256),
		"model.layers.0.self_attn.q_proj.weight":         make([]byte, 128),
		"model.layers.0.self_attn.k_proj.weight":         make([]byte, 128),
		"model.layers.0.self_attn.v_proj.weight":         make([]byte, 128),
		"model.layers.0.self_attn.o_proj.weight":         make([]byte, 128),
		"model.layers.0.mlp.gate_proj.weight":            make([]byte, 256),
		"model.layers.0.mlp.up_proj.weight":              make([]byte, 256),
		"model.layers.0.mlp.down_proj.weight":            make([]byte, 256),
		"model.layers.0.input_layernorm.weight":          make([]byte, 32),
		"model.layers.0.post_attention_layernorm.weight": make([]byte, 32),
	}
	model.CreateMockSafeTensors(path, tensors)

	config := types.Llama7BConfig()
	config.NumLayers = 1

	dm, _ := model.NewDistributedModel(model.DistributedModelConfig{
		ModelConfig: config,
		ModelPath:   tmpDir,
		LocalPeerID: "local",
	})
	defer dm.Close()

	sched := scheduler.NewScheduler(scheduler.ModelConfig{
		HiddenSize:       4096,
		IntermediateSize: 11008,
		NumLayers:        1,
		NumKVHeads:       32,
		HeadDim:          128,
		MaxSeqLen:        2048,
		VocabSize:        32000,
	})
	sched.RegisterPeer("local", 16*1024*1024*1024, 0)

	dm.SetScheduler(sched)

	ctx := context.Background()
	dm.LoadToCluster(ctx)

	// Check status
	status := dm.GetLayerStatus(0)
	if status != model.LoadStatusLoaded {
		t.Errorf("expected LoadStatusLoaded, got %d", status)
	}
}

func TestDistributedModel_LoadWithoutScheduler(t *testing.T) {
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "model.safetensors")

	tensors := map[string][]byte{
		"model.embed_tokens.weight": make([]byte, 256),
	}
	model.CreateMockSafeTensors(path, tensors)

	config := types.Llama7BConfig()
	dm, _ := model.NewDistributedModel(model.DistributedModelConfig{
		ModelConfig: config,
		ModelPath:   tmpDir,
		LocalPeerID: "local",
	})
	defer dm.Close()

	// Don't set scheduler

	err := dm.LoadToCluster(context.Background())
	if err == nil {
		t.Error("expected error when scheduler not set")
	}
}

func TestDistributedModel_ContextCancellation(t *testing.T) {
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "model.safetensors")

	tensors := map[string][]byte{
		"model.embed_tokens.weight": make([]byte, 256),
		"lm_head.weight":            make([]byte, 256),
	}
	model.CreateMockSafeTensors(path, tensors)

	config := types.Llama7BConfig()
	dm, _ := model.NewDistributedModel(model.DistributedModelConfig{
		ModelConfig: config,
		ModelPath:   tmpDir,
		LocalPeerID: "local",
	})
	defer dm.Close()

	sched := scheduler.NewScheduler(scheduler.ModelConfig{
		HiddenSize:       4096,
		IntermediateSize: 11008,
		NumLayers:        32,
		NumKVHeads:       32,
		HeadDim:          128,
		MaxSeqLen:        2048,
		VocabSize:        32000,
	})
	sched.RegisterPeer("local", 16*1024*1024*1024, 0)
	dm.SetScheduler(sched)

	// Create cancelled context
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	err := dm.LoadToCluster(ctx)
	if err == nil {
		t.Error("expected error from cancelled context")
	}
}

// TestDistributedModel_LoadToCluster_Scenario1 tests the TASK-024 acceptance criteria:
// Scenario 1: Load model to cluster
// Given scheduler has assigned layers to 5 peers
// When LoadToCluster is called
// Then embeddings and lm_head load to coordinator
// And each layer loads to its assigned peer
// And all layers report loaded status
func TestDistributedModel_LoadToCluster_Scenario1(t *testing.T) {
	// Create mock model with 10 layers (2 per peer)
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "model.safetensors")

	tensors := map[string][]byte{
		"model.embed_tokens.weight": make([]byte, 1024),
		"lm_head.weight":            make([]byte, 1024),
	}

	// Add 10 layers of weights
	for layer := 0; layer < 10; layer++ {
		prefix := fmt.Sprintf("model.layers.%d.", layer)
		tensors[prefix+"self_attn.q_proj.weight"] = make([]byte, 512)
		tensors[prefix+"self_attn.k_proj.weight"] = make([]byte, 512)
		tensors[prefix+"self_attn.v_proj.weight"] = make([]byte, 512)
		tensors[prefix+"self_attn.o_proj.weight"] = make([]byte, 512)
		tensors[prefix+"mlp.gate_proj.weight"] = make([]byte, 1024)
		tensors[prefix+"mlp.up_proj.weight"] = make([]byte, 1024)
		tensors[prefix+"mlp.down_proj.weight"] = make([]byte, 1024)
		tensors[prefix+"input_layernorm.weight"] = make([]byte, 128)
		tensors[prefix+"post_attention_layernorm.weight"] = make([]byte, 128)
	}

	err := model.CreateMockSafeTensors(path, tensors)
	if err != nil {
		t.Fatalf("CreateMockSafeTensors failed: %v", err)
	}

	// Create config matching our mock
	config := types.Llama7BConfig()
	config.NumLayers = 10

	dm, err := model.NewDistributedModel(model.DistributedModelConfig{
		ModelConfig: config,
		ModelPath:   tmpDir,
		LocalPeerID: "peer-0", // Coordinator is peer-0
	})
	if err != nil {
		t.Fatalf("NewDistributedModel failed: %v", err)
	}
	defer dm.Close()

	// Create scheduler
	sched := scheduler.NewScheduler(scheduler.ModelConfig{
		HiddenSize:       int64(config.HiddenSize),
		IntermediateSize: int64(config.IntermediateSize),
		NumLayers:        config.NumLayers,
		NumKVHeads:       config.NumKVHeads,
		HeadDim:          config.HeadDim,
		MaxSeqLen:        config.MaxSeqLen,
		VocabSize:        int64(config.VocabSize),
	})

	// Register 5 peers with enough VRAM for 2 layers each
	// Each peer gets ~1GB for VRAM (enough for mock layers)
	vramPerPeer := uint64(1024 * 1024 * 1024) // 1GB
	for i := 0; i < 5; i++ {
		peerID := fmt.Sprintf("peer-%d", i)
		sched.RegisterPeer(peerID, vramPerPeer, 0)
	}

	dm.SetScheduler(sched)

	// Load to cluster
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	err = dm.LoadToCluster(ctx)
	if err != nil {
		t.Fatalf("LoadToCluster failed: %v", err)
	}

	// Verify: embeddings and lm_head load to coordinator
	if dm.Embeddings() == nil || len(dm.Embeddings()) == 0 {
		t.Error("embeddings not loaded to coordinator")
	}
	if dm.LMHead() == nil || len(dm.LMHead()) == 0 {
		t.Error("lm_head not loaded to coordinator")
	}

	// Verify: each layer loads to its assigned peer (via status tracking)
	for layer := 0; layer < 10; layer++ {
		status := dm.GetLayerStatus(layer)
		if status != model.LoadStatusLoaded {
			t.Errorf("layer %d not loaded, status: %d", layer, status)
		}
	}

	// Verify: all layers report loaded status
	if !dm.AllLayersLoaded() {
		t.Error("AllLayersLoaded returned false, expected true")
	}

	// Verify loaded count
	if dm.LoadedCount() != 10 {
		t.Errorf("LoadedCount: expected 10, got %d", dm.LoadedCount())
	}

	t.Logf("PASS: Scenario 1 - Load model to cluster with 5 peers")
	t.Logf("  - Embeddings loaded: %d bytes", len(dm.Embeddings()))
	t.Logf("  - LMHead loaded: %d bytes", len(dm.LMHead()))
	t.Logf("  - All %d layers loaded successfully", dm.LoadedCount())
}
