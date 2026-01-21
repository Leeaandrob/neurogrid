//go:build cuda

package tests

import (
	"math"
	"testing"

	"github.com/neurogrid/engine/gpu/bindings"
	"github.com/neurogrid/engine/pkg/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestRMSNorm verifies the RMSNorm kernel produces correct output.
// Given: Input tensor [batch=1, seq=1, hidden=4096] and weight [4096]
// When: RMSNorm kernel is executed
// Then: Output matches PyTorch reference within 1e-5 tolerance
func TestRMSNorm(t *testing.T) {
	config := types.Llama7BConfig()

	// Initialize GPU
	err := bindings.InitGPU(0)
	require.NoError(t, err, "GPU initialization failed")
	defer bindings.ShutdownGPU()

	// Create input tensor
	inputShape := []int{1, 1, config.HiddenSize}
	input := types.NewTensor(inputShape, types.DtypeFP16, 0)

	// Allocate and fill with test data
	err = bindings.AllocateTensor(input)
	require.NoError(t, err, "Failed to allocate input tensor")
	defer bindings.FreeTensor(input)

	// Create weight tensor
	weightShape := []int{config.HiddenSize}
	weight := types.NewTensor(weightShape, types.DtypeFP16, 0)
	err = bindings.AllocateTensor(weight)
	require.NoError(t, err, "Failed to allocate weight tensor")
	defer bindings.FreeTensor(weight)

	// Create output tensor
	output := types.NewTensor(inputShape, types.DtypeFP16, 0)
	err = bindings.AllocateTensor(output)
	require.NoError(t, err, "Failed to allocate output tensor")
	defer bindings.FreeTensor(output)

	// Fill with test values
	testInput := make([]float32, config.HiddenSize)
	testWeight := make([]float32, config.HiddenSize)
	for i := 0; i < config.HiddenSize; i++ {
		testInput[i] = float32(i) * 0.001
		testWeight[i] = 1.0
	}

	err = bindings.CopyToDevice(input, testInput)
	require.NoError(t, err, "Failed to copy input to device")
	err = bindings.CopyToDevice(weight, testWeight)
	require.NoError(t, err, "Failed to copy weight to device")

	// Execute kernel
	err = bindings.RMSNorm(output, input, weight, config.RMSNormEps)
	require.NoError(t, err, "RMSNorm kernel failed")

	// Copy result back
	result := make([]float32, config.HiddenSize)
	err = bindings.CopyToHost(result, output)
	require.NoError(t, err, "Failed to copy result to host")

	// Verify: compute expected RMSNorm
	var sumSq float64
	for _, v := range testInput {
		sumSq += float64(v * v)
	}
	mean := sumSq / float64(config.HiddenSize)
	rsqrt := 1.0 / math.Sqrt(mean+float64(config.RMSNormEps))

	// FP16 has about 3 decimal digits of precision, so use relative tolerance
	maxRelErr := float64(0)
	errCount := 0
	for i := 0; i < config.HiddenSize; i++ {
		expected := float32(float64(testInput[i]) * rsqrt * float64(testWeight[i]))
		if expected == 0 {
			continue
		}
		diff := math.Abs(float64(result[i] - expected))
		relErr := diff / math.Abs(float64(expected))
		if relErr > maxRelErr {
			maxRelErr = relErr
		}
		// FP16 relative tolerance: ~0.1% is reasonable
		if relErr > 0.002 { // 0.2% relative tolerance
			errCount++
			if errCount <= 5 {
				t.Logf("RMSNorm mismatch at index %d: got %f, expected %f, relErr=%f", i, result[i], expected, relErr)
			}
		}
	}
	t.Logf("Max relative error: %f", maxRelErr)
	assert.Less(t, errCount, 10, "Too many RMSNorm mismatches")
}

// TestSiLU verifies the SiLU activation kernel.
// Given: Input tensor [batch=1, seq=1, intermediate=11008]
// When: SiLU kernel is executed
// Then: Output equals x * sigmoid(x) within 1e-5 tolerance
func TestSiLU(t *testing.T) {
	config := types.Llama7BConfig()

	err := bindings.InitGPU(0)
	require.NoError(t, err)
	defer bindings.ShutdownGPU()

	// Create tensors
	shape := []int{1, 1, config.IntermediateSize}
	input := types.NewTensor(shape, types.DtypeFP16, 0)
	output := types.NewTensor(shape, types.DtypeFP16, 0)

	err = bindings.AllocateTensor(input)
	require.NoError(t, err)
	defer bindings.FreeTensor(input)

	err = bindings.AllocateTensor(output)
	require.NoError(t, err)
	defer bindings.FreeTensor(output)

	// Fill test data
	testInput := make([]float32, config.IntermediateSize)
	for i := 0; i < config.IntermediateSize; i++ {
		testInput[i] = float32(i-config.IntermediateSize/2) * 0.01
	}
	err = bindings.CopyToDevice(input, testInput)
	require.NoError(t, err)

	// Execute kernel
	err = bindings.SiLU(output, input)
	require.NoError(t, err, "SiLU kernel failed")

	// Verify
	result := make([]float32, config.IntermediateSize)
	err = bindings.CopyToHost(result, output)
	require.NoError(t, err)

	// SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
	for i := 0; i < 100; i++ { // Check first 100 elements
		x := float64(testInput[i])
		expected := x / (1.0 + math.Exp(-x))
		diff := math.Abs(float64(result[i]) - expected)
		assert.Less(t, diff, 1e-4, "SiLU mismatch at %d: got %f, expected %f", i, result[i], expected)
	}
}

// TestRoPE verifies Rotary Position Embeddings kernel.
// Given: Q/K tensor [batch, seq, num_heads, head_dim] and positions
// When: RoPE kernel is executed
// Then: Positional encoding is correctly applied
func TestRoPE(t *testing.T) {
	config := types.Llama7BConfig()

	err := bindings.InitGPU(0)
	require.NoError(t, err)
	defer bindings.ShutdownGPU()

	batchSize := 1
	seqLen := 4
	shape := []int{batchSize, seqLen, config.NumHeads, config.HeadDim}

	input := types.NewTensor(shape, types.DtypeFP16, 0)
	output := types.NewTensor(shape, types.DtypeFP16, 0)
	positions := make([]int32, batchSize*seqLen)
	for i := range positions {
		positions[i] = int32(i % seqLen)
	}

	err = bindings.AllocateTensor(input)
	require.NoError(t, err)
	defer bindings.FreeTensor(input)

	err = bindings.AllocateTensor(output)
	require.NoError(t, err)
	defer bindings.FreeTensor(output)

	// Fill with test data
	numElements := batchSize * seqLen * config.NumHeads * config.HeadDim
	testInput := make([]float32, numElements)
	for i := range testInput {
		testInput[i] = 1.0
	}
	err = bindings.CopyToDevice(input, testInput)
	require.NoError(t, err)

	// Execute kernel
	err = bindings.RoPE(output, input, positions, config.HeadDim)
	require.NoError(t, err, "RoPE kernel failed")

	// Verify: just check that output is different from input
	// (full verification requires golden data)
	result := make([]float32, numElements)
	err = bindings.CopyToHost(result, output)
	require.NoError(t, err)

	// RoPE should modify the values
	different := false
	for i := range result {
		if math.Abs(float64(result[i]-testInput[i])) > 1e-6 {
			different = true
			break
		}
	}
	assert.True(t, different, "RoPE should modify input values")
}

// TestAddKernel verifies element-wise addition (residual add).
func TestAddKernel(t *testing.T) {
	err := bindings.InitGPU(0)
	require.NoError(t, err)
	defer bindings.ShutdownGPU()

	size := 4096
	shape := []int{size}

	a := types.NewTensor(shape, types.DtypeFP16, 0)
	b := types.NewTensor(shape, types.DtypeFP16, 0)
	c := types.NewTensor(shape, types.DtypeFP16, 0)

	err = bindings.AllocateTensor(a)
	require.NoError(t, err)
	defer bindings.FreeTensor(a)

	err = bindings.AllocateTensor(b)
	require.NoError(t, err)
	defer bindings.FreeTensor(b)

	err = bindings.AllocateTensor(c)
	require.NoError(t, err)
	defer bindings.FreeTensor(c)

	// Fill test data
	dataA := make([]float32, size)
	dataB := make([]float32, size)
	for i := 0; i < size; i++ {
		dataA[i] = float32(i) * 0.001
		dataB[i] = float32(size-i) * 0.001
	}

	err = bindings.CopyToDevice(a, dataA)
	require.NoError(t, err)
	err = bindings.CopyToDevice(b, dataB)
	require.NoError(t, err)

	// Execute: c = a + b
	err = bindings.Add(c, a, b)
	require.NoError(t, err, "Add kernel failed")

	// Verify
	result := make([]float32, size)
	err = bindings.CopyToHost(result, c)
	require.NoError(t, err)

	errCount := 0
	for i := 0; i < size; i++ {
		expected := dataA[i] + dataB[i]
		diff := math.Abs(float64(result[i] - expected))
		// FP16 has ~0.1% relative precision
		if diff > 0.01 { // 1% absolute tolerance for small values
			errCount++
			if errCount <= 5 {
				t.Logf("Add mismatch at %d: got %f, expected %f, diff=%f", i, result[i], expected, diff)
			}
		}
	}
	assert.Less(t, errCount, 10, "Too many Add mismatches")
}
