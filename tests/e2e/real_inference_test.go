//go:build cuda
// +build cuda

package e2e

import (
	"context"
	"math"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/neurogrid/engine/gpu/bindings"
	"github.com/neurogrid/engine/pkg/inference"
	"github.com/neurogrid/engine/pkg/model"
	"github.com/neurogrid/engine/pkg/scheduler"
	"github.com/neurogrid/engine/pkg/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Ensure imports are used
var _ = model.NewWeightLoader

// TestCUDALayerExecutor_Interface verifies CUDALayerExecutor implements LayerExecutor
// AC1: CUDALayerExecutor Implementation
func TestCUDALayerExecutor_Interface(t *testing.T) {
	if !bindings.HasCUDA() {
		t.Skip("CUDA not available")
	}

	config := getTestConfig()

	// This should compile - CUDALayerExecutor must implement LayerExecutor
	cudaExec, err := inference.NewCUDALayerExecutor(config, 0)
	require.NoError(t, err, "Failed to create CUDALayerExecutor")
	var executor inference.LayerExecutor = cudaExec
	require.NotNil(t, executor, "CUDALayerExecutor should not be nil")

	// Cleanup
	if closer, ok := executor.(interface{ Close() error }); ok {
		closer.Close()
	}
}

// TestCUDALayerExecutor_LoadLayer verifies layer weight loading to GPU
// AC1: CUDALayerExecutor Implementation
func TestCUDALayerExecutor_LoadLayer(t *testing.T) {
	if !bindings.HasCUDA() {
		t.Skip("CUDA not available")
	}

	config := getTestConfig()
	executor, execErr := inference.NewCUDALayerExecutor(config, 0)
	require.NoError(t, execErr, "Failed to create executor")
	defer executor.Close()

	// Load golden weights (skip if not available)
	weights, err := tryLoadGoldenLayerWeights(0)
	if err != nil {
		t.Skipf("Golden data not available: %v", err)
	}

	// Should successfully load layer to GPU
	err = executor.LoadLayer(0, weights)
	require.NoError(t, err, "LoadLayer should succeed")
}

// TestCUDALayerExecutor_Forward verifies forward pass execution
// AC1: CUDALayerExecutor Implementation
func TestCUDALayerExecutor_Forward(t *testing.T) {
	if !bindings.HasCUDA() {
		t.Skip("CUDA not available")
	}

	config := getTestConfig()
	executor, execErr := inference.NewCUDALayerExecutor(config, 0)
	require.NoError(t, execErr, "Failed to create executor")
	defer executor.Close()

	// Load layer (skip if not available)
	weights, err := tryLoadGoldenLayerWeights(0)
	if err != nil {
		t.Skipf("Golden data not available: %v", err)
	}
	require.NoError(t, executor.LoadLayer(0, weights))

	// Load golden input (skip if not available)
	input, err := tryLoadBinaryFile("tests/golden/layer_0_input.bin")
	if err != nil {
		t.Skipf("Golden input not available: %v", err)
	}

	// Execute forward pass
	ctx := context.Background()
	output, k, v, err := executor.Forward(ctx, 0, input, 0)

	require.NoError(t, err, "Forward should succeed")
	require.NotNil(t, output, "Output should not be nil")
	require.NotNil(t, k, "K should not be nil")
	require.NotNil(t, v, "V should not be nil")

	// Output should match golden data
	expected, err := tryLoadBinaryFile("tests/golden/layer_0_output.bin")
	if err != nil {
		t.Skipf("Golden output not available: %v", err)
	}
	assertTensorClose(t, output, expected, 5e-3)
}

// TestCreateLayerWeightsFromHost verifies weight transfer from Go to GPU
// AC2: GPU Weight Bridge
func TestCreateLayerWeightsFromHost(t *testing.T) {
	if !bindings.HasCUDA() {
		t.Skip("CUDA not available")
	}

	config := getTestConfig()

	// Try to load golden weights (skip if not available)
	qProj, err := tryLoadBinaryFile("tests/golden/layer_0_weights/q_proj.bin")
	if err != nil {
		t.Skipf("Golden data not available: %v", err)
	}
	kProj := loadBinaryFile(t, "tests/golden/layer_0_weights/k_proj.bin")
	vProj := loadBinaryFile(t, "tests/golden/layer_0_weights/v_proj.bin")
	oProj := loadBinaryFile(t, "tests/golden/layer_0_weights/o_proj.bin")
	gateProj := loadBinaryFile(t, "tests/golden/layer_0_weights/gate_proj.bin")
	upProj := loadBinaryFile(t, "tests/golden/layer_0_weights/up_proj.bin")
	downProj := loadBinaryFile(t, "tests/golden/layer_0_weights/down_proj.bin")
	attnNorm := loadBinaryFile(t, "tests/golden/layer_0_weights/input_layernorm.bin")
	ffnNorm := loadBinaryFile(t, "tests/golden/layer_0_weights/post_attn_layernorm.bin")

	// Create GPU weights
	weights, err := bindings.CreateLayerWeightsFromHost(
		qProj, kProj, vProj, oProj,
		gateProj, upProj, downProj,
		attnNorm, ffnNorm,
		config,
	)

	require.NoError(t, err, "CreateLayerWeightsFromHost should succeed")
	require.NotNil(t, weights, "Weights should not be nil")

	// Cleanup
	bindings.FreeLayerWeights(weights)
}

// TestCreateLayerWeightsFromHost_Mock verifies weight bridge with mock data
// AC2: GPU Weight Bridge
func TestCreateLayerWeightsFromHost_Mock(t *testing.T) {
	if !bindings.HasCUDA() {
		t.Skip("CUDA not available")
	}

	// Initialize multi-GPU context
	if err := bindings.InitMultiGPU([]int{0}); err != nil {
		t.Fatalf("Failed to initialize multi-GPU: %v", err)
	}
	defer bindings.ShutdownMultiGPU()

	config := &types.LlamaConfig{
		HiddenSize:       128,
		IntermediateSize: 256,
		NumHeads:         4,
		NumKVHeads:       4,
		HeadDim:          32,
		NumLayers:        2,
		VocabSize:        1000,
		RMSNormEps:       1e-6,
	}

	// Create mock weight data with valid patterns
	hiddenSize := config.HiddenSize
	intermediateSize := config.IntermediateSize

	// Projection matrices: [out, in] * sizeof(int8) + scales
	projSize := hiddenSize * hiddenSize
	ffnProjSize := intermediateSize * hiddenSize
	normSize := hiddenSize * 2 // FP16 norm weights

	makeWeights := func(size int) []byte {
		data := make([]byte, size)
		for i := range data {
			data[i] = byte(i % 256)
		}
		return data
	}

	qProj := makeWeights(projSize)
	kProj := makeWeights(projSize)
	vProj := makeWeights(projSize)
	oProj := makeWeights(projSize)
	gateProj := makeWeights(ffnProjSize)
	upProj := makeWeights(ffnProjSize)
	downProj := makeWeights(ffnProjSize)
	attnNorm := makeWeights(normSize)
	ffnNorm := makeWeights(normSize)

	// Create GPU weights
	weights, err := bindings.CreateLayerWeightsFromHost(
		qProj, kProj, vProj, oProj,
		gateProj, upProj, downProj,
		attnNorm, ffnNorm,
		config,
	)

	require.NoError(t, err, "CreateLayerWeightsFromHost should succeed")
	require.NotNil(t, weights, "Weights should not be nil")

	// Cleanup
	bindings.FreeLayerWeights(weights)
}

// TestGPUEmbeddings_Lookup verifies token embedding lookup on GPU
// AC3: GPU Embedding Lookup
func TestGPUEmbeddings_Lookup(t *testing.T) {
	if !bindings.HasCUDA() {
		t.Skip("CUDA not available")
	}

	// Initialize multi-GPU context
	if err := bindings.InitMultiGPU([]int{0}); err != nil {
		t.Fatalf("Failed to initialize multi-GPU: %v", err)
	}
	defer bindings.ShutdownMultiGPU()

	// Create mock embeddings (small for testing)
	vocabSize := 100
	hiddenSize := 128
	embData := make([]byte, vocabSize*hiddenSize*2) // FP16

	// Fill with pattern
	for i := range embData {
		embData[i] = byte(i % 256)
	}

	embeddings, err := inference.NewGPUEmbeddings(embData, vocabSize, hiddenSize)
	require.NoError(t, err, "NewGPUEmbeddings should succeed")
	defer embeddings.Close()

	// Lookup token
	hidden, err := embeddings.LookupToHost(42)
	require.NoError(t, err, "LookupToHost should succeed")
	require.Equal(t, hiddenSize*2, len(hidden), "Hidden size should match")

	// Verify content matches expected offset
	offset := 42 * hiddenSize * 2
	for i := 0; i < min(10, len(hidden)); i++ {
		assert.Equal(t, byte((offset+i)%256), hidden[i], "Embedding content mismatch at %d", i)
	}
}

// TestGPUEmbeddings_OutOfRange verifies bounds checking
// AC3: GPU Embedding Lookup
func TestGPUEmbeddings_OutOfRange(t *testing.T) {
	if !bindings.HasCUDA() {
		t.Skip("CUDA not available")
	}

	// Initialize multi-GPU context
	if err := bindings.InitMultiGPU([]int{0}); err != nil {
		t.Fatalf("Failed to initialize multi-GPU: %v", err)
	}
	defer bindings.ShutdownMultiGPU()

	vocabSize := 100
	hiddenSize := 128
	embData := make([]byte, vocabSize*hiddenSize*2)

	embeddings, err := inference.NewGPUEmbeddings(embData, vocabSize, hiddenSize)
	require.NoError(t, err)
	defer embeddings.Close()

	// Should fail for out-of-range token
	_, err = embeddings.LookupToHost(vocabSize + 10)
	require.Error(t, err, "Should error on out-of-range token")
}

// TestGPULMHead_Forward verifies LM head matmul
// AC4: GPU LM Head
func TestGPULMHead_Forward(t *testing.T) {
	if !bindings.HasCUDA() {
		t.Skip("CUDA not available")
	}

	// Initialize multi-GPU context
	if err := bindings.InitMultiGPU([]int{0}); err != nil {
		t.Fatalf("Failed to initialize multi-GPU: %v", err)
	}
	defer bindings.ShutdownMultiGPU()

	hiddenSize := 128
	vocabSize := 1000

	// Create mock LM head weights with valid FP16 values
	// Using FP16 representation of small numbers (0.5 = 0x3800)
	lmData := make([]byte, hiddenSize*vocabSize*2) // FP16
	for i := 0; i < len(lmData); i += 2 {
		// Small positive FP16 value
		lmData[i] = 0x00
		lmData[i+1] = 0x38 // ~0.5 in FP16
	}

	lmHead, err := inference.NewGPULMHead(lmData, hiddenSize, vocabSize)
	require.NoError(t, err, "NewGPULMHead should succeed")
	defer lmHead.Close()

	// Create hidden state with valid FP16 values
	hidden := make([]byte, hiddenSize*2) // FP16
	for i := 0; i < len(hidden); i += 2 {
		// Small positive FP16 value
		hidden[i] = 0x00
		hidden[i+1] = 0x3C // 1.0 in FP16
	}

	// Forward should return logits
	logits, err := lmHead.Forward(hidden)
	require.NoError(t, err, "Forward should succeed")
	require.Equal(t, vocabSize, len(logits), "Logits should have vocab_size elements")

	// Logits should be valid floats (not NaN/Inf)
	for i, l := range logits {
		assert.False(t, isNaN(l), "Logit %d should not be NaN", i)
		assert.False(t, isInf(l), "Logit %d should not be Inf", i)
	}
}

// TestEngine_InitializeGPU verifies GPU initialization
// AC5: End-to-End Inference Test
func TestEngine_InitializeGPU(t *testing.T) {
	if !bindings.HasCUDA() {
		t.Skip("CUDA not available")
	}

	// Load TinyLlama model
	loader, err := model.NewWeightLoader(getModelPath())
	if err != nil {
		t.Skip("TinyLlama model not available: ", err)
	}
	defer loader.Close()

	config := getTinyLlamaEngineConfig()
	engine := inference.NewEngine(config)

	// Should initialize GPU successfully
	gpu, err := engine.InitializeGPU(loader, 0)
	require.NoError(t, err, "InitializeGPU should succeed")
	defer gpu.Close()

	// Engine should be ready for inference
	assert.True(t, gpu.Initialized, "GPU should be initialized")
}

// TestRealInference_CoherentOutput verifies coherent text generation
// AC5: End-to-End Inference Test
func TestRealInference_CoherentOutput(t *testing.T) {
	if !bindings.HasCUDA() {
		t.Skip("CUDA not available")
	}

	engine := setupEngineWithTinyLlama(t)
	defer engine.Close()

	ctx := context.Background()
	resp, err := engine.Generate(ctx, &inference.GenerateRequest{
		Prompt:      "The capital of France is",
		MaxTokens:   10,
		Temperature: 0.0, // Greedy for determinism
	})

	require.NoError(t, err, "Generate should succeed")
	require.NotEmpty(t, resp.Text, "Response should not be empty")

	// Output should contain "Paris" or related words
	output := strings.ToLower(resp.Text)
	isCoherent := strings.Contains(output, "paris") ||
		strings.Contains(output, "city") ||
		strings.Contains(output, "french") ||
		strings.Contains(output, "capital")

	assert.True(t, isCoherent,
		"Expected coherent output about Paris, got: %s", resp.Text)
}

// TestRealInference_NotGarbage verifies output is real text
// AC5: End-to-End Inference Test
func TestRealInference_NotGarbage(t *testing.T) {
	if !bindings.HasCUDA() {
		t.Skip("CUDA not available")
	}

	engine := setupEngineWithTinyLlama(t)
	defer engine.Close()

	ctx := context.Background()
	resp, err := engine.Generate(ctx, &inference.GenerateRequest{
		Prompt:    "Hello, how are you?",
		MaxTokens: 20,
	})

	require.NoError(t, err)

	// Should not contain hex escape sequences (garbage output indicator)
	assert.NotContains(t, resp.Text, "<0x",
		"Output should not contain hex escapes: %s", resp.Text)

	// Should contain mostly English words (at least 50%)
	words := strings.Fields(resp.Text)
	if len(words) > 0 {
		englishWords := countEnglishWords(words)
		ratio := float64(englishWords) / float64(len(words))
		assert.GreaterOrEqual(t, ratio, 0.5,
			"Expected mostly English words (%.1f%%), got: %s",
			ratio*100, resp.Text)
	}
}

// TestRealInference_TTFT verifies time-to-first-token
// AC6: Chat Completions API Functional
func TestRealInference_TTFT(t *testing.T) {
	if !bindings.HasCUDA() {
		t.Skip("CUDA not available")
	}

	engine := setupEngineWithTinyLlama(t)
	defer engine.Close()

	ctx := context.Background()
	start := time.Now()

	_, err := engine.Generate(ctx, &inference.GenerateRequest{
		Prompt:    "Hello",
		MaxTokens: 1, // Just first token
	})

	ttft := time.Since(start)

	require.NoError(t, err)
	assert.Less(t, ttft, 500*time.Millisecond,
		"TTFT should be < 500ms, got %v", ttft)
}

// ============================================================================
// Helper Functions
// ============================================================================

func getTestConfig() *types.LlamaConfig {
	return &types.LlamaConfig{
		HiddenSize:       4096,
		IntermediateSize: 11008,
		NumHeads:         32,
		NumKVHeads:       32,
		HeadDim:          128,
		NumLayers:        32,
		VocabSize:        32000,
		RMSNormEps:       1e-6,
	}
}

func getTinyLlamaConfig() *types.LlamaConfig {
	return &types.LlamaConfig{
		HiddenSize:       2048,
		IntermediateSize: 5632,
		NumHeads:         32,
		NumKVHeads:       4,
		HeadDim:          64,
		NumLayers:        22,
		VocabSize:        32000,
		RMSNormEps:       1e-5,
	}
}

func getTinyLlamaEngineConfig() inference.EngineConfig {
	return inference.EngineConfig{
		ModelConfig: getTinyLlamaConfig(),
		LocalPeerID: "local",
	}
}

// getModelPath returns the model path, detecting whether we're running from
// project root or tests/e2e directory.
func getModelPath() string {
	if _, err := os.Stat("models/tinyllama"); err == nil {
		return "models/tinyllama"
	}
	return "../../models/tinyllama"
}

func loadGoldenLayerWeights(t *testing.T, layerID int) *inference.TransformerLayerWeights {
	t.Helper()
	dir := "tests/golden/layer_0_weights/"

	return &inference.TransformerLayerWeights{
		QProj:    loadBinaryFile(t, dir+"q_proj.bin"),
		KProj:    loadBinaryFile(t, dir+"k_proj.bin"),
		VProj:    loadBinaryFile(t, dir+"v_proj.bin"),
		OProj:    loadBinaryFile(t, dir+"o_proj.bin"),
		GateProj: loadBinaryFile(t, dir+"gate_proj.bin"),
		UpProj:   loadBinaryFile(t, dir+"up_proj.bin"),
		DownProj: loadBinaryFile(t, dir+"down_proj.bin"),
		AttnNorm: loadBinaryFile(t, dir+"input_layernorm.bin"),
		FFNNorm:  loadBinaryFile(t, dir+"post_attn_layernorm.bin"),
	}
}

func loadBinaryFile(t *testing.T, path string) []byte {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("Failed to load %s: %v", path, err)
	}
	return data
}

func loadGoldenTensor(t *testing.T, path string) []byte {
	t.Helper()
	return loadBinaryFile(t, path)
}

// tryLoadBinaryFile attempts to load a binary file, returning error instead of failing test
func tryLoadBinaryFile(path string) ([]byte, error) {
	return os.ReadFile(path)
}

// tryLoadGoldenLayerWeights attempts to load golden weights, returning error instead of failing test
func tryLoadGoldenLayerWeights(layerID int) (*inference.TransformerLayerWeights, error) {
	dir := "tests/golden/layer_0_weights/"

	qProj, err := tryLoadBinaryFile(dir + "q_proj.bin")
	if err != nil {
		return nil, err
	}
	kProj, err := tryLoadBinaryFile(dir + "k_proj.bin")
	if err != nil {
		return nil, err
	}
	vProj, err := tryLoadBinaryFile(dir + "v_proj.bin")
	if err != nil {
		return nil, err
	}
	oProj, err := tryLoadBinaryFile(dir + "o_proj.bin")
	if err != nil {
		return nil, err
	}
	gateProj, err := tryLoadBinaryFile(dir + "gate_proj.bin")
	if err != nil {
		return nil, err
	}
	upProj, err := tryLoadBinaryFile(dir + "up_proj.bin")
	if err != nil {
		return nil, err
	}
	downProj, err := tryLoadBinaryFile(dir + "down_proj.bin")
	if err != nil {
		return nil, err
	}
	attnNorm, err := tryLoadBinaryFile(dir + "input_layernorm.bin")
	if err != nil {
		return nil, err
	}
	ffnNorm, err := tryLoadBinaryFile(dir + "post_attn_layernorm.bin")
	if err != nil {
		return nil, err
	}

	return &inference.TransformerLayerWeights{
		QProj:    qProj,
		KProj:    kProj,
		VProj:    vProj,
		OProj:    oProj,
		GateProj: gateProj,
		UpProj:   upProj,
		DownProj: downProj,
		AttnNorm: attnNorm,
		FFNNorm:  ffnNorm,
	}, nil
}

func assertTensorClose(t *testing.T, actual, expected []byte, tolerance float64) {
	t.Helper()
	require.Equal(t, len(expected), len(actual), "Tensor sizes must match")

	maxDiff := 0.0
	for i := 0; i < len(actual); i += 2 {
		a := float16ToFloat32(actual[i], actual[i+1])
		e := float16ToFloat32(expected[i], expected[i+1])
		diff := abs(a - e)
		if diff > maxDiff {
			maxDiff = diff
		}
	}

	assert.Less(t, maxDiff, tolerance,
		"Max diff %.6f exceeds tolerance %.6f", maxDiff, tolerance)
}

// testEngine wraps Engine and GPUComponents for cleanup
type testEngine struct {
	*inference.Engine
	gpu    *inference.GPUComponents
	loader *model.WeightLoader
}

func (te *testEngine) Close() error {
	if te.gpu != nil {
		te.gpu.Close()
	}
	if te.loader != nil {
		te.loader.Close()
	}
	return nil
}

func setupEngineWithTinyLlama(t *testing.T) *testEngine {
	t.Helper()

	modelPath := getModelPath()
	t.Logf("Using model path: %s", modelPath)

	loader, err := model.NewWeightLoader(modelPath)
	if err != nil {
		t.Skip("TinyLlama model not available: ", err)
	}

	config := getTinyLlamaEngineConfig()
	engine := inference.NewEngine(config)

	// Configure all layers as local for single-GPU execution
	numLayers := config.ModelConfig.NumLayers
	assignments := make([]scheduler.LayerAssignment, numLayers)
	for i := 0; i < numLayers; i++ {
		assignments[i] = scheduler.LayerAssignment{
			LayerID: i,
			PeerID:  config.LocalPeerID,
		}
	}
	engine.SetAssignments(assignments)

	tokenizer, err := model.NewSentencePieceTokenizer(modelPath)
	if err != nil {
		loader.Close()
		t.Skip("Tokenizer not available: ", err)
	}
	engine.SetTokenizer(tokenizer)

	gpu, err := engine.InitializeGPU(loader, 0)
	if err != nil {
		loader.Close()
		t.Fatalf("Failed to initialize GPU: %v", err)
	}

	return &testEngine{
		Engine: engine,
		gpu:    gpu,
		loader: loader,
	}
}

func float16ToFloat32(lo, hi byte) float32 {
	bits := uint16(lo) | uint16(hi)<<8
	// Simplified FP16 to FP32 conversion
	sign := uint32((bits >> 15) & 1)
	exp := uint32((bits >> 10) & 0x1F)
	frac := uint32(bits & 0x3FF)

	if exp == 0 {
		return 0
	}
	if exp == 31 {
		if frac != 0 {
			return float32(math.NaN())
		}
		if sign == 1 {
			return float32(math.Inf(-1))
		}
		return float32(math.Inf(1))
	}

	exp32 := exp - 15 + 127
	frac32 := frac << 13
	bits32 := (sign << 31) | (exp32 << 23) | frac32
	return math.Float32frombits(bits32)
}

func abs(x float32) float64 {
	if x < 0 {
		return float64(-x)
	}
	return float64(x)
}

func isNaN(f float32) bool {
	return f != f
}

func isInf(f float32) bool {
	return math.IsInf(float64(f), 0)
}

func countEnglishWords(words []string) int {
	// Simple heuristic: words with only ASCII letters
	count := 0
	for _, w := range words {
		w = strings.ToLower(w)
		isWord := true
		for _, c := range w {
			if c < 'a' || c > 'z' {
				isWord = false
				break
			}
		}
		if isWord && len(w) > 1 {
			count++
		}
	}
	return count
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
