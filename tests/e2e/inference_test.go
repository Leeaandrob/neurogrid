// Package e2e provides end-to-end tests for the NeuroGrid distributed inference engine.
package e2e

import (
	"context"
	"testing"
	"time"

	"github.com/neurogrid/engine/pkg/inference"
	"github.com/neurogrid/engine/pkg/scheduler"
	"github.com/neurogrid/engine/pkg/types"
)

// ===== TASK-021: Token Sampler Tests =====

func TestSampler_GreedyDecoding(t *testing.T) {
	sampler := inference.NewSampler(42)

	// Logits with clear maximum at index 42
	logits := make([]float32, 100)
	logits[42] = 10.0

	// Temperature <= 0 should use greedy decoding
	result := sampler.Sample(logits, 0.0, 1.0)
	if result != 42 {
		t.Errorf("greedy decode: expected 42, got %d", result)
	}

	// Negative temperature should also use greedy
	result = sampler.Sample(logits, -1.0, 1.0)
	if result != 42 {
		t.Errorf("negative temp: expected 42, got %d", result)
	}
}

func TestSampler_TemperatureScaling(t *testing.T) {
	sampler := inference.NewSampler(12345)

	// Create logits with a clear winner
	logits := make([]float32, 10)
	logits[5] = 5.0 // Clear winner

	// Low temperature should strongly favor the highest logit
	lowTempResults := make(map[int]int)
	for i := 0; i < 100; i++ {
		result := sampler.Sample(logits, 0.1, 1.0)
		lowTempResults[result]++
	}

	// Token 5 should be selected most of the time with low temp
	if lowTempResults[5] < 80 {
		t.Errorf("low temp: expected token 5 to be selected >80%%, got %d%%", lowTempResults[5])
	}
}

func TestSampler_TopPSampling(t *testing.T) {
	sampler := inference.NewSampler(42)

	// Create logits where top 2 tokens have 90% probability
	logits := make([]float32, 10)
	logits[0] = 5.0 // ~45%
	logits[1] = 5.0 // ~45%
	// Others ~10% combined

	// Sample with top-p = 0.9, should mostly get 0 or 1
	results := make(map[int]int)
	for i := 0; i < 100; i++ {
		result := sampler.Sample(logits, 1.0, 0.9)
		results[result]++
	}

	topTwo := results[0] + results[1]
	if topTwo < 80 {
		t.Errorf("top-p 0.9: expected tokens 0,1 to be selected >80%%, got %d%%", topTwo)
	}
}

func TestSampler_Reproducibility(t *testing.T) {
	logits := make([]float32, 100)
	for i := range logits {
		logits[i] = float32(i % 10)
	}

	// Same seed should produce same sequence
	sampler1 := inference.NewSampler(42)
	sampler2 := inference.NewSampler(42)

	for i := 0; i < 10; i++ {
		r1 := sampler1.Sample(logits, 0.7, 0.9)
		r2 := sampler2.Sample(logits, 0.7, 0.9)
		if r1 != r2 {
			t.Errorf("iteration %d: seed 42 produced different results: %d vs %d", i, r1, r2)
		}
	}
}

func TestSampler_EmptyLogits(t *testing.T) {
	sampler := inference.NewSampler(42)

	// Empty logits should return 0
	result := sampler.Sample([]float32{}, 1.0, 1.0)
	if result != 0 {
		t.Errorf("empty logits: expected 0, got %d", result)
	}
}

func TestSampler_SampleGreedy(t *testing.T) {
	sampler := inference.NewSampler(42)

	logits := []float32{1.0, 5.0, 3.0, 2.0}
	result := sampler.SampleGreedy(logits)
	if result != 1 {
		t.Errorf("SampleGreedy: expected 1, got %d", result)
	}
}

// ===== TASK-020: KV Cache Tests =====

func TestDistributedKVCache_LocalUpdate(t *testing.T) {
	config := inference.KVCacheConfig{
		LayerID:    0,
		NumKVHeads: 4,
		HeadDim:    64,
		MaxSeqLen:  128,
	}

	cache := inference.NewDistributedKVCache(config, "local", 0, true)

	// Create K and V data
	dataSize := config.NumKVHeads * config.HeadDim * 2 // FP16
	k := make([]byte, dataSize)
	v := make([]byte, dataSize)
	for i := range k {
		k[i] = byte(i % 256)
		v[i] = byte((i + 1) % 256)
	}

	ctx := context.Background()

	// Update at position 0
	err := cache.Update(ctx, k, v, 0)
	if err != nil {
		t.Fatalf("Update failed: %v", err)
	}

	if cache.CurrentLength() != 1 {
		t.Errorf("CurrentLength: expected 1, got %d", cache.CurrentLength())
	}

	// Update at position 5
	err = cache.Update(ctx, k, v, 5)
	if err != nil {
		t.Fatalf("Update at position 5 failed: %v", err)
	}

	if cache.CurrentLength() != 6 {
		t.Errorf("CurrentLength: expected 6, got %d", cache.CurrentLength())
	}
}

func TestDistributedKVCache_Get(t *testing.T) {
	config := inference.KVCacheConfig{
		LayerID:    0,
		NumKVHeads: 2,
		HeadDim:    32,
		MaxSeqLen:  64,
	}

	cache := inference.NewDistributedKVCache(config, "local", 0, true)

	dataSize := config.NumKVHeads * config.HeadDim * 2
	k := make([]byte, dataSize)
	v := make([]byte, dataSize)

	ctx := context.Background()

	// Fill first 3 positions
	for i := 0; i < 3; i++ {
		cache.Update(ctx, k, v, i)
	}

	// Get up to length 2
	keys, values, err := cache.Get(ctx, 2)
	if err != nil {
		t.Fatalf("Get failed: %v", err)
	}

	expectedSize := 2 * dataSize
	if len(keys) != expectedSize || len(values) != expectedSize {
		t.Errorf("Get returned wrong size: keys=%d, values=%d, expected %d", len(keys), len(values), expectedSize)
	}
}

func TestDistributedKVCache_Clear(t *testing.T) {
	config := inference.KVCacheConfig{
		LayerID:    0,
		NumKVHeads: 2,
		HeadDim:    32,
		MaxSeqLen:  64,
	}

	cache := inference.NewDistributedKVCache(config, "local", 0, true)

	dataSize := config.NumKVHeads * config.HeadDim * 2
	k := make([]byte, dataSize)
	v := make([]byte, dataSize)

	ctx := context.Background()
	cache.Update(ctx, k, v, 0)
	cache.Update(ctx, k, v, 1)

	if cache.CurrentLength() != 2 {
		t.Errorf("before clear: expected length 2, got %d", cache.CurrentLength())
	}

	cache.Clear()

	if cache.CurrentLength() != 0 {
		t.Errorf("after clear: expected length 0, got %d", cache.CurrentLength())
	}
}

func TestDistributedKVCache_RemoteNoAllocation(t *testing.T) {
	config := inference.KVCacheConfig{
		LayerID:    5,
		NumKVHeads: 4,
		HeadDim:    64,
		MaxSeqLen:  128,
	}

	// Remote cache should not allocate local memory
	cache := inference.NewDistributedKVCache(config, "remote-peer", 0, false)

	if cache.IsLocal() {
		t.Error("remote cache should not be local")
	}

	// Update should be no-op for remote cache
	ctx := context.Background()
	err := cache.Update(ctx, nil, nil, 0)
	if err != nil {
		t.Errorf("remote Update should not error: %v", err)
	}

	// Get should error for remote cache
	_, _, err = cache.Get(ctx, 1)
	if err == nil {
		t.Error("remote Get should error")
	}
}

func TestKVCacheManager_RegisterAndGet(t *testing.T) {
	manager := inference.NewKVCacheManager()

	config := inference.KVCacheConfig{
		LayerID:    3,
		NumKVHeads: 4,
		HeadDim:    64,
		MaxSeqLen:  128,
	}

	cache := inference.NewDistributedKVCache(config, "local", 0, true)
	manager.RegisterCache(cache)

	retrieved, ok := manager.GetCache(3)
	if !ok {
		t.Fatal("GetCache failed for layer 3")
	}

	if retrieved.LayerID() != 3 {
		t.Errorf("wrong layer ID: expected 3, got %d", retrieved.LayerID())
	}

	_, ok = manager.GetCache(5)
	if ok {
		t.Error("GetCache should return false for unregistered layer")
	}
}

func TestKVCacheManager_ClearAll(t *testing.T) {
	manager := inference.NewKVCacheManager()

	for i := 0; i < 3; i++ {
		config := inference.KVCacheConfig{
			LayerID:    i,
			NumKVHeads: 2,
			HeadDim:    32,
			MaxSeqLen:  64,
		}
		cache := inference.NewDistributedKVCache(config, "local", 0, true)

		// Add some data
		dataSize := config.NumKVHeads * config.HeadDim * 2
		cache.Update(context.Background(), make([]byte, dataSize), make([]byte, dataSize), 0)

		manager.RegisterCache(cache)
	}

	// Verify all have data
	for i := 0; i < 3; i++ {
		cache, _ := manager.GetCache(i)
		if cache.CurrentLength() != 1 {
			t.Errorf("layer %d: expected length 1 before clear", i)
		}
	}

	manager.ClearAll()

	// Verify all cleared
	for i := 0; i < 3; i++ {
		cache, _ := manager.GetCache(i)
		if cache.CurrentLength() != 0 {
			t.Errorf("layer %d: expected length 0 after clear", i)
		}
	}
}

// ===== TASK-019: Inference Engine Tests =====

// MockTokenizer implements Tokenizer for testing.
type MockTokenizer struct {
	vocabSize int
	eosToken  int
}

func NewMockTokenizer() *MockTokenizer {
	return &MockTokenizer{
		vocabSize: 1000,
		eosToken:  2,
	}
}

func (t *MockTokenizer) Encode(text string) ([]int, error) {
	// Simple mock: one token per character
	tokens := make([]int, len(text))
	for i, c := range text {
		tokens[i] = int(c) % t.vocabSize
	}
	return tokens, nil
}

func (t *MockTokenizer) Decode(tokens []int) (string, error) {
	// Simple mock: convert tokens back
	chars := make([]rune, len(tokens))
	for i, tok := range tokens {
		chars[i] = rune(tok + 32) // Offset to printable ASCII
	}
	return string(chars), nil
}

func (t *MockTokenizer) EOSToken() int  { return t.eosToken }
func (t *MockTokenizer) BOSToken() int  { return 1 }
func (t *MockTokenizer) VocabSize() int { return t.vocabSize }

func TestEngine_Creation(t *testing.T) {
	config := types.Llama7BConfig()
	engine := inference.NewEngine(inference.EngineConfig{
		ModelConfig: config,
		LocalPeerID: "local",
	})

	if engine == nil {
		t.Fatal("NewEngine returned nil")
	}

	if engine.Config() != config {
		t.Error("Config mismatch")
	}
}

func TestEngine_SetComponents(t *testing.T) {
	config := types.Llama7BConfig()
	engine := inference.NewEngine(inference.EngineConfig{
		ModelConfig: config,
		LocalPeerID: "local",
	})

	// Set tokenizer
	tokenizer := NewMockTokenizer()
	engine.SetTokenizer(tokenizer)

	// Set sampler
	engine.SetSampler(12345)

	// These shouldn't panic
	engine.SetScheduler(nil)
	engine.SetRouter(nil)
}

func TestEngine_InitializeKVCaches(t *testing.T) {
	config := types.Llama7BConfig()
	engine := inference.NewEngine(inference.EngineConfig{
		ModelConfig: config,
		LocalPeerID: "local",
	})

	// Create assignments
	assignments := []scheduler.LayerAssignment{
		{LayerID: -1, PeerID: "local", MemoryNeeded: 1000}, // Embedding
		{LayerID: 0, PeerID: "local", MemoryNeeded: 1000},  // Layer 0
		{LayerID: 1, PeerID: "local", MemoryNeeded: 1000},  // Layer 1
		{LayerID: 2, PeerID: "remote", MemoryNeeded: 1000}, // Layer 2 remote
		{LayerID: 32, PeerID: "local", MemoryNeeded: 1000}, // Output
	}

	engine.SetAssignments(assignments)
	err := engine.InitializeKVCaches()
	if err != nil {
		t.Fatalf("InitializeKVCaches failed: %v", err)
	}

	// Check local caches
	caches := engine.KVCaches()
	if caches.CacheCount() != 3 { // layers 0, 1, 2 (not embedding -1 or output 32)
		t.Errorf("expected 3 caches, got %d", caches.CacheCount())
	}

	// Verify layer 0 is local
	cache0, ok := caches.GetCache(0)
	if !ok {
		t.Fatal("cache for layer 0 not found")
	}
	if !cache0.IsLocal() {
		t.Error("layer 0 should be local")
	}

	// Verify layer 2 is remote
	cache2, ok := caches.GetCache(2)
	if !ok {
		t.Fatal("cache for layer 2 not found")
	}
	if cache2.IsLocal() {
		t.Error("layer 2 should be remote")
	}
}

func TestEngine_GenerateWithMockTokenizer(t *testing.T) {
	config := types.Llama7BConfig()
	config.NumLayers = 2 // Reduce for faster test

	engine := inference.NewEngine(inference.EngineConfig{
		ModelConfig: config,
		LocalPeerID: "local",
	})

	// Set up tokenizer
	tokenizer := NewMockTokenizer()
	engine.SetTokenizer(tokenizer)

	// Set up assignments (all local)
	assignments := []scheduler.LayerAssignment{
		{LayerID: -1, PeerID: "local"},
		{LayerID: 0, PeerID: "local"},
		{LayerID: 1, PeerID: "local"},
		{LayerID: 2, PeerID: "local"},
	}
	engine.SetAssignments(assignments)
	engine.InitializeKVCaches()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	req := &inference.GenerateRequest{
		Prompt:      "Hello",
		MaxTokens:   5,
		Temperature: 0.0, // Greedy for determinism
		TopP:        1.0,
	}

	resp, err := engine.Generate(ctx, req)
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	if resp.TokenCount == 0 {
		t.Error("expected non-zero token count")
	}

	if resp.Text == "" {
		t.Error("expected non-empty response text")
	}

	// With mock LM head returning peak at 42, all tokens should be 42
	// Decode converts 42 -> 'J' (42 + 32 = 74 = 'J')
	t.Logf("Generated: %q (tokens: %d, stop: %s)", resp.Text, resp.TokenCount, resp.StopReason)
}

func TestEngine_GenerateRespectsMaxTokens(t *testing.T) {
	config := types.Llama7BConfig()
	config.NumLayers = 1

	engine := inference.NewEngine(inference.EngineConfig{
		ModelConfig: config,
		LocalPeerID: "local",
	})

	engine.SetTokenizer(NewMockTokenizer())
	engine.SetAssignments([]scheduler.LayerAssignment{
		{LayerID: -1, PeerID: "local"},
		{LayerID: 0, PeerID: "local"},
		{LayerID: 1, PeerID: "local"},
	})
	engine.InitializeKVCaches()

	ctx := context.Background()
	req := &inference.GenerateRequest{
		Prompt:      "Hi",
		MaxTokens:   3,
		Temperature: 0.0,
		TopP:        1.0,
	}

	resp, err := engine.Generate(ctx, req)
	if err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	if resp.TokenCount > 3 {
		t.Errorf("expected at most 3 tokens, got %d", resp.TokenCount)
	}

	if resp.StopReason != "max_tokens" {
		t.Errorf("expected stop reason 'max_tokens', got '%s'", resp.StopReason)
	}
}

func TestEngine_GenerateRespectsContext(t *testing.T) {
	config := types.Llama7BConfig()
	config.NumLayers = 1

	engine := inference.NewEngine(inference.EngineConfig{
		ModelConfig: config,
		LocalPeerID: "local",
	})

	engine.SetTokenizer(NewMockTokenizer())
	engine.SetAssignments([]scheduler.LayerAssignment{
		{LayerID: -1, PeerID: "local"},
		{LayerID: 0, PeerID: "local"},
		{LayerID: 1, PeerID: "local"},
	})
	engine.InitializeKVCaches()

	// Create already-cancelled context
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	req := &inference.GenerateRequest{
		Prompt:      "Hello",
		MaxTokens:   100,
		Temperature: 0.7,
		TopP:        0.9,
	}

	_, err := engine.Generate(ctx, req)
	if err == nil {
		t.Error("expected error from cancelled context")
	}
}

func TestEngine_GenerateWithoutTokenizer(t *testing.T) {
	config := types.Llama7BConfig()
	engine := inference.NewEngine(inference.EngineConfig{
		ModelConfig: config,
		LocalPeerID: "local",
	})

	// Don't set tokenizer

	req := &inference.GenerateRequest{
		Prompt:    "Hello",
		MaxTokens: 5,
	}

	_, err := engine.Generate(context.Background(), req)
	if err == nil {
		t.Error("expected error when tokenizer not set")
	}
}

func TestEngine_DistributedLayerForward(t *testing.T) {
	config := types.Llama7BConfig()
	config.NumLayers = 4

	engine := inference.NewEngine(inference.EngineConfig{
		ModelConfig: config,
		LocalPeerID: "local",
	})

	engine.SetTokenizer(NewMockTokenizer())

	// Create mixed local/remote assignments
	assignments := []scheduler.LayerAssignment{
		{LayerID: -1, PeerID: "local"},
		{LayerID: 0, PeerID: "local"},
		{LayerID: 1, PeerID: "local"},
		{LayerID: 2, PeerID: "remote-peer-1"}, // Remote
		{LayerID: 3, PeerID: "remote-peer-1"}, // Remote
		{LayerID: 4, PeerID: "local"},
	}
	engine.SetAssignments(assignments)
	engine.InitializeKVCaches()

	// Verify assignments
	if len(engine.Assignments()) != 6 {
		t.Errorf("expected 6 assignments, got %d", len(engine.Assignments()))
	}

	// Check local caches
	localCaches := engine.KVCaches().LocalCaches()
	if len(localCaches) != 2 { // layers 0 and 1
		t.Errorf("expected 2 local caches, got %d", len(localCaches))
	}
}
