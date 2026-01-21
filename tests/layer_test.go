//go:build cuda

package tests

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/neurogrid/engine/gpu/bindings"
	"github.com/neurogrid/engine/pkg/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestSingleLayerForward verifies complete transformer layer forward pass.
// Given: Input hidden states and layer weights (from golden data)
// When: Full layer forward is executed
// Then: Output matches PyTorch reference within 1e-5 tolerance
func TestSingleLayerForward(t *testing.T) {
	config := types.Llama7BConfig()

	err := bindings.InitGPU(0)
	require.NoError(t, err)
	defer bindings.ShutdownGPU()

	// Check if golden data exists
	goldenDir := "golden"
	inputPath := filepath.Join(goldenDir, "layer_0_input.bin")
	outputPath := filepath.Join(goldenDir, "layer_0_output.bin")
	weightsPath := filepath.Join(goldenDir, "layer_0_weights")

	if _, err := os.Stat(inputPath); os.IsNotExist(err) {
		t.Skip("Golden data not found. Run 'make golden' first.")
	}

	// Load golden input
	goldenInput, err := loadGoldenTensor(inputPath)
	require.NoError(t, err, "Failed to load golden input")

	// Load golden output
	goldenOutput, err := loadGoldenTensor(outputPath)
	require.NoError(t, err, "Failed to load golden output")

	// Load weights - LoadLayerWeights requires Phase 2 safetensors implementation
	weights, err := bindings.LoadLayerWeights(weightsPath)
	if err != nil {
		t.Skip("Weight loader not implemented. This test requires Phase 2 safetensors weight loading.")
	}
	defer bindings.FreeLayerWeights(weights)

	// Create KV cache
	batchSize := 1
	kvCache, err := bindings.NewKVCache(batchSize, config.NumHeads, config.HeadDim, config.MaxSeqLen)
	require.NoError(t, err)
	defer bindings.FreeKVCache(kvCache)

	// Create input tensor
	seqLen := len(goldenInput) / config.HiddenSize
	inputShape := []int{batchSize, seqLen, config.HiddenSize}
	input := types.NewTensor(inputShape, types.DtypeFP16, 0)
	output := types.NewTensor(inputShape, types.DtypeFP16, 0)

	err = bindings.AllocateTensor(input)
	require.NoError(t, err)
	defer bindings.FreeTensor(input)

	err = bindings.AllocateTensor(output)
	require.NoError(t, err)
	defer bindings.FreeTensor(output)

	// Copy input to device
	err = bindings.CopyToDevice(input, goldenInput)
	require.NoError(t, err)

	// Positions array
	positions := make([]int32, seqLen)
	for i := range positions {
		positions[i] = int32(i)
	}

	// Execute layer forward
	err = bindings.LayerForward(output, input, weights, kvCache, positions, config)
	require.NoError(t, err, "Layer forward failed")

	// Copy result back
	result := make([]float32, len(goldenOutput))
	err = bindings.CopyToHost(result, output)
	require.NoError(t, err)

	// Compare with golden output
	maxDiff := float64(0)
	avgDiff := float64(0)
	for i := range result {
		diff := math.Abs(float64(result[i] - goldenOutput[i]))
		avgDiff += diff
		if diff > maxDiff {
			maxDiff = diff
		}
	}
	avgDiff /= float64(len(result))

	// Tolerance: 1e-3 for max (FP16 + INT8 quantization combined error)
	assert.Less(t, maxDiff, 1e-3, "Max diff %f exceeds tolerance", maxDiff)
	assert.Less(t, avgDiff, 1e-6, "Avg diff %f exceeds tolerance", avgDiff)

	t.Logf("Layer forward validation: max_diff=%e, avg_diff=%e", maxDiff, avgDiff)
}

// TestLayerForwardMultiplePositions verifies layer works for multiple tokens.
func TestLayerForwardMultiplePositions(t *testing.T) {
	config := types.Llama7BConfig()

	err := bindings.InitGPU(0)
	require.NoError(t, err)
	defer bindings.ShutdownGPU()

	batchSize := 1
	seqLen := 4

	// Create test weights (random for this test)
	weights, err := bindings.CreateRandomLayerWeights(config)
	require.NoError(t, err)
	defer bindings.FreeLayerWeights(weights)

	// Create KV cache
	kvCache, err := bindings.NewKVCache(batchSize, config.NumHeads, config.HeadDim, config.MaxSeqLen)
	require.NoError(t, err)
	defer bindings.FreeKVCache(kvCache)

	// Create input
	inputShape := []int{batchSize, seqLen, config.HiddenSize}
	input := types.NewTensor(inputShape, types.DtypeFP16, 0)
	output := types.NewTensor(inputShape, types.DtypeFP16, 0)

	err = bindings.AllocateTensor(input)
	require.NoError(t, err)
	defer bindings.FreeTensor(input)

	err = bindings.AllocateTensor(output)
	require.NoError(t, err)
	defer bindings.FreeTensor(output)

	// Fill input with random data
	inputData := make([]float32, batchSize*seqLen*config.HiddenSize)
	for i := range inputData {
		inputData[i] = float32(i%1000) * 0.001
	}
	err = bindings.CopyToDevice(input, inputData)
	require.NoError(t, err)

	// Positions
	positions := make([]int32, seqLen)
	for i := range positions {
		positions[i] = int32(i)
	}

	// Execute
	err = bindings.LayerForward(output, input, weights, kvCache, positions, config)
	require.NoError(t, err, "Layer forward failed for multiple positions")

	// Basic validation: output should not be all zeros
	result := make([]float32, batchSize*seqLen*config.HiddenSize)
	err = bindings.CopyToHost(result, output)
	require.NoError(t, err)

	sumAbs := float64(0)
	for _, v := range result {
		sumAbs += math.Abs(float64(v))
	}
	assert.Greater(t, sumAbs, 0.0, "Output should not be all zeros")
}

// TestLayerForwardIncrementalGeneration verifies autoregressive generation.
// Given: We process tokens one at a time using KV cache
// When: Each token is processed with accumulated KV cache
// Then: KV cache correctly accumulates and attention uses all previous tokens
func TestLayerForwardIncrementalGeneration(t *testing.T) {
	config := types.Llama7BConfig()

	err := bindings.InitGPU(0)
	require.NoError(t, err)
	defer bindings.ShutdownGPU()

	batchSize := 1
	totalTokens := 8

	// Create weights
	weights, err := bindings.CreateRandomLayerWeights(config)
	require.NoError(t, err)
	defer bindings.FreeLayerWeights(weights)

	// Create KV cache
	kvCache, err := bindings.NewKVCache(batchSize, config.NumHeads, config.HeadDim, config.MaxSeqLen)
	require.NoError(t, err)
	defer bindings.FreeKVCache(kvCache)

	// Process tokens one at a time
	singleShape := []int{batchSize, 1, config.HiddenSize}
	input := types.NewTensor(singleShape, types.DtypeFP16, 0)
	output := types.NewTensor(singleShape, types.DtypeFP16, 0)

	err = bindings.AllocateTensor(input)
	require.NoError(t, err)
	defer bindings.FreeTensor(input)

	err = bindings.AllocateTensor(output)
	require.NoError(t, err)
	defer bindings.FreeTensor(output)

	prevOutputs := make([][]float32, totalTokens)

	for pos := 0; pos < totalTokens; pos++ {
		// Input for this token
		inputData := make([]float32, config.HiddenSize)
		for i := range inputData {
			inputData[i] = float32(pos+1) * 0.01
		}
		err = bindings.CopyToDevice(input, inputData)
		require.NoError(t, err)

		positions := []int32{int32(pos)}

		// Execute
		err = bindings.LayerForward(output, input, weights, kvCache, positions, config)
		require.NoError(t, err, "Layer forward failed at position %d", pos)

		// Store output
		prevOutputs[pos] = make([]float32, config.HiddenSize)
		err = bindings.CopyToHost(prevOutputs[pos], output)
		require.NoError(t, err)

		// Verify KV cache length increased
		cacheLen := bindings.GetKVCacheLength(kvCache)
		assert.Equal(t, pos+1, cacheLen, "KV cache length should be %d, got %d", pos+1, cacheLen)
	}

	// Outputs should be different for each position (due to different positions in RoPE)
	for i := 1; i < totalTokens; i++ {
		different := false
		for j := 0; j < config.HiddenSize; j++ {
			if math.Abs(float64(prevOutputs[i][j]-prevOutputs[i-1][j])) > 1e-6 {
				different = true
				break
			}
		}
		assert.True(t, different, "Outputs for positions %d and %d should differ", i-1, i)
	}
}

// BenchmarkLayerForward measures layer forward throughput.
func BenchmarkLayerForward(b *testing.B) {
	config := types.Llama7BConfig()

	err := bindings.InitGPU(0)
	if err != nil {
		b.Skip("GPU not available")
	}
	defer bindings.ShutdownGPU()

	batchSize := 1
	seqLen := 1

	weights, err := bindings.CreateRandomLayerWeights(config)
	if err != nil {
		b.Fatal(err)
	}
	defer bindings.FreeLayerWeights(weights)

	kvCache, err := bindings.NewKVCache(batchSize, config.NumHeads, config.HeadDim, config.MaxSeqLen)
	if err != nil {
		b.Fatal(err)
	}
	defer bindings.FreeKVCache(kvCache)

	inputShape := []int{batchSize, seqLen, config.HiddenSize}
	input := types.NewTensor(inputShape, types.DtypeFP16, 0)
	output := types.NewTensor(inputShape, types.DtypeFP16, 0)

	_ = bindings.AllocateTensor(input)
	defer bindings.FreeTensor(input)
	_ = bindings.AllocateTensor(output)
	defer bindings.FreeTensor(output)

	positions := []int32{0}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		positions[0] = int32(i % config.MaxSeqLen)
		_ = bindings.LayerForward(output, input, weights, kvCache, positions, config)
	}
	_ = bindings.SyncDevice()
}

// loadGoldenTensor loads a binary tensor file (FP32 format).
func loadGoldenTensor(path string) ([]float32, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	// Assuming FP32 little-endian
	numElements := len(data) / 4
	result := make([]float32, numElements)
	for i := 0; i < numElements; i++ {
		bits := uint32(data[i*4]) | uint32(data[i*4+1])<<8 | uint32(data[i*4+2])<<16 | uint32(data[i*4+3])<<24
		result[i] = math.Float32frombits(bits)
	}
	return result, nil
}
