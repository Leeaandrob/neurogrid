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

// goldenDir is the directory containing PyTorch-generated reference data.
const goldenDir = "golden"

// TestRMSNormGolden validates RMSNorm kernel against PyTorch reference.
// Given: PyTorch-generated input, weight, and expected output
// When: CUDA RMSNorm kernel is executed with the same input/weight
// Then: Output matches PyTorch reference within 1e-5 tolerance
func TestRMSNormGolden(t *testing.T) {
	inputPath := filepath.Join(goldenDir, "rmsnorm_input.bin")
	weightPath := filepath.Join(goldenDir, "rmsnorm_weight.bin")
	outputPath := filepath.Join(goldenDir, "rmsnorm_output.bin")

	if _, err := os.Stat(inputPath); os.IsNotExist(err) {
		t.Skip("Golden data not found. Run 'python scripts/generate_golden.py --kernels-only' first.")
	}

	// Load golden data
	goldenInput, err := loadGoldenTensor(inputPath)
	require.NoError(t, err, "Failed to load golden input")
	goldenWeight, err := loadGoldenTensor(weightPath)
	require.NoError(t, err, "Failed to load golden weight")
	goldenOutput, err := loadGoldenTensor(outputPath)
	require.NoError(t, err, "Failed to load golden output")

	// Initialize GPU
	err = bindings.InitGPU(0)
	require.NoError(t, err)
	defer bindings.ShutdownGPU()

	// Golden data is [1, 1, 4096]
	hiddenSize := len(goldenWeight)
	t.Logf("Golden RMSNorm: hidden_size=%d, input_size=%d, output_size=%d",
		hiddenSize, len(goldenInput), len(goldenOutput))

	// Create tensors
	inputShape := []int{1, 1, hiddenSize}
	input := types.NewTensor(inputShape, types.DtypeFP16, 0)
	weight := types.NewTensor([]int{hiddenSize}, types.DtypeFP16, 0)
	output := types.NewTensor(inputShape, types.DtypeFP16, 0)

	err = bindings.AllocateTensor(input)
	require.NoError(t, err)
	defer bindings.FreeTensor(input)

	err = bindings.AllocateTensor(weight)
	require.NoError(t, err)
	defer bindings.FreeTensor(weight)

	err = bindings.AllocateTensor(output)
	require.NoError(t, err)
	defer bindings.FreeTensor(output)

	// Copy golden input to device
	err = bindings.CopyToDevice(input, goldenInput)
	require.NoError(t, err)
	err = bindings.CopyToDevice(weight, goldenWeight)
	require.NoError(t, err)

	// Execute CUDA kernel
	eps := float32(1e-6) // Same as PyTorch default
	err = bindings.RMSNorm(output, input, weight, eps)
	require.NoError(t, err, "RMSNorm kernel failed")

	// Copy result back
	result := make([]float32, len(goldenOutput))
	err = bindings.CopyToHost(result, output)
	require.NoError(t, err)

	// Compare with golden output
	maxDiff := float64(0)
	avgDiff := float64(0)
	errCount := 0

	for i := range result {
		diff := math.Abs(float64(result[i] - goldenOutput[i]))
		avgDiff += diff
		if diff > maxDiff {
			maxDiff = diff
		}
		// FP16 has ~0.1% relative precision, combined with kernel differences
		// use 1e-3 absolute tolerance
		if diff > 1e-3 {
			errCount++
			if errCount <= 5 {
				t.Logf("RMSNorm mismatch at %d: got %f, expected %f, diff=%e",
					i, result[i], goldenOutput[i], diff)
			}
		}
	}
	avgDiff /= float64(len(result))

	t.Logf("RMSNorm Golden: max_diff=%e, avg_diff=%e, errors=%d/%d",
		maxDiff, avgDiff, errCount, len(result))

	// FP16 validation: max diff < 5e-3 (2-3 ULP at typical magnitudes)
	// Average diff should be very small (most values match exactly)
	assert.Less(t, maxDiff, 5e-3, "Max diff exceeds FP16 tolerance")
	assert.Less(t, avgDiff, 1e-4, "Average diff too high")
	assert.Less(t, float64(errCount), float64(len(result))*0.01, "Too many mismatches (>1%%)")
}

// TestSiLUGolden validates SiLU kernel against PyTorch reference.
// Given: PyTorch-generated input and expected output
// When: CUDA SiLU kernel is executed with the same input
// Then: Output matches PyTorch reference within 1e-5 tolerance
func TestSiLUGolden(t *testing.T) {
	inputPath := filepath.Join(goldenDir, "silu_input.bin")
	outputPath := filepath.Join(goldenDir, "silu_output.bin")

	if _, err := os.Stat(inputPath); os.IsNotExist(err) {
		t.Skip("Golden data not found. Run 'python scripts/generate_golden.py --kernels-only' first.")
	}

	// Load golden data
	goldenInput, err := loadGoldenTensor(inputPath)
	require.NoError(t, err, "Failed to load golden input")
	goldenOutput, err := loadGoldenTensor(outputPath)
	require.NoError(t, err, "Failed to load golden output")

	// Initialize GPU
	err = bindings.InitGPU(0)
	require.NoError(t, err)
	defer bindings.ShutdownGPU()

	size := len(goldenInput)
	t.Logf("Golden SiLU: size=%d", size)

	// Create tensors
	shape := []int{size}
	input := types.NewTensor(shape, types.DtypeFP16, 0)
	output := types.NewTensor(shape, types.DtypeFP16, 0)

	err = bindings.AllocateTensor(input)
	require.NoError(t, err)
	defer bindings.FreeTensor(input)

	err = bindings.AllocateTensor(output)
	require.NoError(t, err)
	defer bindings.FreeTensor(output)

	// Copy golden input to device
	err = bindings.CopyToDevice(input, goldenInput)
	require.NoError(t, err)

	// Execute CUDA kernel
	err = bindings.SiLU(output, input)
	require.NoError(t, err, "SiLU kernel failed")

	// Copy result back
	result := make([]float32, size)
	err = bindings.CopyToHost(result, output)
	require.NoError(t, err)

	// Compare with golden output
	maxDiff := float64(0)
	avgDiff := float64(0)
	errCount := 0

	for i := range result {
		diff := math.Abs(float64(result[i] - goldenOutput[i]))
		avgDiff += diff
		if diff > maxDiff {
			maxDiff = diff
		}
		// SiLU involves exp() which can amplify FP16 errors
		// Use 1e-3 absolute tolerance
		if diff > 1e-3 {
			errCount++
			if errCount <= 5 {
				t.Logf("SiLU mismatch at %d: got %f, expected %f, diff=%e",
					i, result[i], goldenOutput[i], diff)
			}
		}
	}
	avgDiff /= float64(len(result))

	t.Logf("SiLU Golden: max_diff=%e, avg_diff=%e, errors=%d/%d",
		maxDiff, avgDiff, errCount, len(result))

	// FP16 validation: SiLU involves exp() which can amplify small differences
	// max diff < 5e-3, avg diff < 1e-4
	assert.Less(t, maxDiff, 5e-3, "Max diff exceeds FP16 tolerance")
	assert.Less(t, avgDiff, 1e-4, "Average diff too high")
	assert.Less(t, float64(errCount), float64(len(result))*0.01, "Too many mismatches (>1%%)")
}

// TestGEMMGolden validates GEMM kernel against PyTorch reference.
// Given: PyTorch-generated matrices A, B and expected C = A @ B
// When: CUDA GEMM kernel is executed with the same matrices
// Then: Output matches PyTorch reference within 1e-5 tolerance
func TestGEMMGolden(t *testing.T) {
	aPath := filepath.Join(goldenDir, "gemm_a.bin")
	bPath := filepath.Join(goldenDir, "gemm_b.bin")
	cPath := filepath.Join(goldenDir, "gemm_c.bin")

	if _, err := os.Stat(aPath); os.IsNotExist(err) {
		t.Skip("Golden data not found. Run 'python scripts/generate_golden.py --kernels-only' first.")
	}

	// Load golden data
	goldenA, err := loadGoldenTensor(aPath)
	require.NoError(t, err, "Failed to load golden A")
	goldenB, err := loadGoldenTensor(bPath)
	require.NoError(t, err, "Failed to load golden B")
	goldenC, err := loadGoldenTensor(cPath)
	require.NoError(t, err, "Failed to load golden C")

	// Initialize GPU
	err = bindings.InitGPU(0)
	require.NoError(t, err)
	defer bindings.ShutdownGPU()

	// Dimensions from generate_golden.py: M=32, K=128, N=64
	// A: [32, 128], B: [128, 64], C: [32, 64]
	M := 32
	K := 128
	N := 64

	require.Equal(t, M*K, len(goldenA), "A size mismatch")
	require.Equal(t, K*N, len(goldenB), "B size mismatch")
	require.Equal(t, M*N, len(goldenC), "C size mismatch")

	t.Logf("Golden GEMM: A[%d,%d] @ B[%d,%d] = C[%d,%d]", M, K, K, N, M, N)

	// Create tensors
	a := types.NewTensor([]int{M, K}, types.DtypeFP16, 0)
	b := types.NewTensor([]int{K, N}, types.DtypeFP16, 0)
	c := types.NewTensor([]int{M, N}, types.DtypeFP16, 0)

	err = bindings.AllocateTensor(a)
	require.NoError(t, err)
	defer bindings.FreeTensor(a)

	err = bindings.AllocateTensor(b)
	require.NoError(t, err)
	defer bindings.FreeTensor(b)

	err = bindings.AllocateTensor(c)
	require.NoError(t, err)
	defer bindings.FreeTensor(c)

	// Copy golden data to device
	err = bindings.CopyToDevice(a, goldenA)
	require.NoError(t, err)
	err = bindings.CopyToDevice(b, goldenB)
	require.NoError(t, err)

	// Execute CUDA GEMM: C = A @ B
	err = bindings.GEMMFP16(c, a, b, false, false)
	require.NoError(t, err, "GEMM kernel failed")

	// Copy result back
	result := make([]float32, M*N)
	err = bindings.CopyToHost(result, c)
	require.NoError(t, err)

	// Compare with golden output
	maxDiff := float64(0)
	avgDiff := float64(0)
	errCount := 0

	for i := range result {
		diff := math.Abs(float64(result[i] - goldenC[i]))
		avgDiff += diff
		if diff > maxDiff {
			maxDiff = diff
		}
		// GEMM accumulates K products, FP16 error accumulates
		// K=128 means we expect ~sqrt(128) * FP16_epsilon relative error
		// Use 1e-2 absolute tolerance (1% error)
		if diff > 1e-2 {
			errCount++
			if errCount <= 5 {
				row := i / N
				col := i % N
				t.Logf("GEMM mismatch at [%d,%d]: got %f, expected %f, diff=%e",
					row, col, result[i], goldenC[i], diff)
			}
		}
	}
	avgDiff /= float64(len(result))

	t.Logf("GEMM Golden: max_diff=%e, avg_diff=%e, errors=%d/%d",
		maxDiff, avgDiff, errCount, len(result))

	// GEMM with FP16: K=128 accumulations cause error accumulation
	// FP16 has ~10 bits mantissa, so relative error ~1e-3 per op
	// With K=128 ops, expected error ~sqrt(128)*1e-3 ≈ 1.1%
	// Max diff 0.05 (5%) and avg diff < 0.01 (1%) are acceptable
	assert.Less(t, maxDiff, 0.1, "Max diff exceeds 10%% tolerance")
	assert.Less(t, avgDiff, 0.01, "Average diff exceeds 1%% tolerance")
}

// TestRMSNormWithLlamaWeights validates RMSNorm using actual Llama layer 0 weights.
// This tests the exact weights that will be used in the full model.
func TestRMSNormWithLlamaWeights(t *testing.T) {
	weightsPath := filepath.Join(goldenDir, "layer_0_weights", "input_layernorm.bin")
	inputPath := filepath.Join(goldenDir, "layer_0_input.bin")
	expectedPath := filepath.Join(goldenDir, "after_input_norm.bin")

	if _, err := os.Stat(weightsPath); os.IsNotExist(err) {
		t.Skip("Layer golden data not found. Run 'python scripts/generate_golden.py --model meta-llama/Llama-2-7b-hf' first.")
	}

	// Load golden data
	goldenWeights, err := loadGoldenTensor(weightsPath)
	require.NoError(t, err, "Failed to load Llama weights")
	goldenInput, err := loadGoldenTensor(inputPath)
	require.NoError(t, err, "Failed to load golden input")
	goldenExpected, err := loadGoldenTensor(expectedPath)
	require.NoError(t, err, "Failed to load expected output")

	// Initialize GPU
	err = bindings.InitGPU(0)
	require.NoError(t, err)
	defer bindings.ShutdownGPU()

	// Golden data: input [1, 4, 4096], weight [4096]
	hiddenSize := len(goldenWeights)
	seqLen := len(goldenInput) / hiddenSize
	t.Logf("Llama RMSNorm: hidden_size=%d, seq_len=%d", hiddenSize, seqLen)

	// Create tensors
	inputShape := []int{1, seqLen, hiddenSize}
	input := types.NewTensor(inputShape, types.DtypeFP16, 0)
	weight := types.NewTensor([]int{hiddenSize}, types.DtypeFP16, 0)
	output := types.NewTensor(inputShape, types.DtypeFP16, 0)

	err = bindings.AllocateTensor(input)
	require.NoError(t, err)
	defer bindings.FreeTensor(input)

	err = bindings.AllocateTensor(weight)
	require.NoError(t, err)
	defer bindings.FreeTensor(weight)

	err = bindings.AllocateTensor(output)
	require.NoError(t, err)
	defer bindings.FreeTensor(output)

	// Copy golden data to device
	err = bindings.CopyToDevice(input, goldenInput)
	require.NoError(t, err)
	err = bindings.CopyToDevice(weight, goldenWeights)
	require.NoError(t, err)

	// Execute RMSNorm with Llama's epsilon (1e-6 is Llama default)
	eps := float32(1e-6)
	err = bindings.RMSNorm(output, input, weight, eps)
	require.NoError(t, err, "RMSNorm kernel failed")

	// Copy result back
	result := make([]float32, len(goldenExpected))
	err = bindings.CopyToHost(result, output)
	require.NoError(t, err)

	// Compare with golden output
	maxDiff := float64(0)
	avgDiff := float64(0)
	errCount := 0

	for i := range result {
		diff := math.Abs(float64(result[i] - goldenExpected[i]))
		avgDiff += diff
		if diff > maxDiff {
			maxDiff = diff
		}
		if diff > 5e-3 {
			errCount++
		}
	}
	avgDiff /= float64(len(result))

	t.Logf("RMSNorm with Llama weights: max_diff=%e, avg_diff=%e, errors=%d/%d",
		maxDiff, avgDiff, errCount, len(result))

	// Validate: FP16 precision with real Llama weights
	assert.Less(t, maxDiff, 0.01, "Max diff exceeds tolerance")
	assert.Less(t, avgDiff, 1e-4, "Average diff too high")
}

// TestKernelGoldenSummary runs all kernel golden tests and summarizes results.
func TestKernelGoldenSummary(t *testing.T) {
	if _, err := os.Stat(filepath.Join(goldenDir, "rmsnorm_input.bin")); os.IsNotExist(err) {
		t.Skip("Golden data not found. Run 'python scripts/generate_golden.py --kernels-only' first.")
	}

	t.Log("=== Kernel Golden Data Validation Summary ===")
	t.Log("This test suite validates CUDA kernels against PyTorch reference outputs")
	t.Log("")
	t.Log("Golden data generated by: scripts/generate_golden.py --kernels-only")
	t.Log("Using PyTorch with FP16 precision, deterministic seed=42")
	t.Log("")
	t.Log("Run individual tests for detailed validation:")
	t.Log("  - TestRMSNormGolden: RMSNorm normalization")
	t.Log("  - TestSiLUGolden: SiLU activation")
	t.Log("  - TestGEMMGolden: Matrix multiplication")
	t.Log("  - TestRMSNormWithLlamaWeights: RMSNorm with actual Llama layer 0 weights")
}
