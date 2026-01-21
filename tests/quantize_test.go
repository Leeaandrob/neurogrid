//go:build cuda

package tests

import (
	"math"
	"math/rand"
	"testing"

	"github.com/neurogrid/engine/gpu/bindings"
	"github.com/neurogrid/engine/pkg/types"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestQuantizePerTensor verifies FP16 to INT8 quantization.
// Given: FP16 tensor with values in range [-1, 1]
// When: Quantize kernel is executed
// Then: INT8 values correctly represent the original within quantization error
func TestQuantizePerTensor(t *testing.T) {
	err := bindings.InitGPU(0)
	require.NoError(t, err)
	defer bindings.ShutdownGPU()

	size := 4096
	input := types.NewTensor([]int{size}, types.DtypeFP16, 0)
	output := types.NewTensor([]int{size}, types.DtypeINT8, 0)
	scale := types.NewTensor([]int{1}, types.DtypeFP32, 0)

	err = bindings.AllocateTensor(input)
	require.NoError(t, err)
	defer bindings.FreeTensor(input)

	err = bindings.AllocateTensor(output)
	require.NoError(t, err)
	defer bindings.FreeTensor(output)

	err = bindings.AllocateTensor(scale)
	require.NoError(t, err)
	defer bindings.FreeTensor(scale)

	// Fill with random data
	rng := rand.New(rand.NewSource(42))
	data := make([]float32, size)
	maxAbs := float32(0)
	for i := range data {
		data[i] = rng.Float32()*2 - 1
		if abs := float32(math.Abs(float64(data[i]))); abs > maxAbs {
			maxAbs = abs
		}
	}

	err = bindings.CopyToDevice(input, data)
	require.NoError(t, err)

	// Quantize
	err = bindings.QuantizePerTensor(output, scale, input)
	require.NoError(t, err, "Quantize kernel failed")

	// Get results
	resultINT8 := make([]int8, size)
	resultScale := make([]float32, 1)
	err = bindings.CopyToHostINT8(resultINT8, output)
	require.NoError(t, err)
	err = bindings.CopyToHost(resultScale, scale)
	require.NoError(t, err)

	// Verify scale is reasonable
	expectedScale := maxAbs / 127.0
	assert.InDelta(t, expectedScale, resultScale[0], 0.01, "Scale mismatch")

	// Verify roundtrip error
	maxError := float32(0)
	for i := range data {
		dequant := float32(resultINT8[i]) * resultScale[0]
		err := float32(math.Abs(float64(data[i] - dequant)))
		if err > maxError {
			maxError = err
		}
	}

	// INT8 quantization error: max_error ≈ scale/2 = max_abs/(127*2) ≈ 0.4% of max_abs
	// With max_abs ≈ 1, expect max_error ≈ 0.004
	t.Logf("Quantize: maxError=%f, scale=%f", maxError, resultScale[0])
	assert.Less(t, float64(maxError), 0.01, "Quantization error too large: %f", maxError)
}

// TestDequantizePerTensor verifies INT8 to FP16 dequantization.
func TestDequantizePerTensor(t *testing.T) {
	err := bindings.InitGPU(0)
	require.NoError(t, err)
	defer bindings.ShutdownGPU()

	size := 4096
	input := types.NewTensor([]int{size}, types.DtypeINT8, 0)
	output := types.NewTensor([]int{size}, types.DtypeFP16, 0)
	scale := types.NewTensor([]int{1}, types.DtypeFP32, 0)

	err = bindings.AllocateTensor(input)
	require.NoError(t, err)
	defer bindings.FreeTensor(input)

	err = bindings.AllocateTensor(output)
	require.NoError(t, err)
	defer bindings.FreeTensor(output)

	err = bindings.AllocateTensor(scale)
	require.NoError(t, err)
	defer bindings.FreeTensor(scale)

	// Fill with INT8 data
	rng := rand.New(rand.NewSource(42))
	dataINT8 := make([]int8, size)
	for i := range dataINT8 {
		dataINT8[i] = int8(rng.Intn(256) - 128)
	}
	scaleVal := []float32{0.01}

	err = bindings.CopyToDeviceINT8(input, dataINT8)
	require.NoError(t, err)
	err = bindings.CopyToDevice(scale, scaleVal)
	require.NoError(t, err)

	// Dequantize
	err = bindings.DequantizePerTensor(output, input, scale)
	require.NoError(t, err, "Dequantize kernel failed")

	// Verify
	result := make([]float32, size)
	err = bindings.CopyToHost(result, output)
	require.NoError(t, err)

	maxDiff := 0.0
	errCount := 0
	for i := range result {
		expected := float32(dataINT8[i]) * scaleVal[0]
		diff := math.Abs(float64(result[i] - expected))
		if diff > maxDiff {
			maxDiff = diff
		}
		// FP16 has ~0.1% relative precision, so use absolute tolerance based on value range
		// INT8 values in [-128, 127] with scale 0.01 gives range [-1.28, 1.27]
		// FP16 epsilon at this scale is ~1e-3, so 1e-3 absolute tolerance
		if diff > 1e-3 {
			errCount++
		}
	}
	t.Logf("Dequant: maxDiff=%f, errCount=%d/%d", maxDiff, errCount, len(result))
	assert.Less(t, errCount, 10, "Too many dequant mismatches")
}

// TestQuantizeWeights verifies static weight quantization.
// Given: FP16 weight matrix [K, N]
// When: Static quantization is performed (per-column or per-tensor)
// Then: INT8 weights and scales are correctly computed
func TestQuantizeWeights(t *testing.T) {
	err := bindings.InitGPU(0)
	require.NoError(t, err)
	defer bindings.ShutdownGPU()

	K, N := 128, 64
	weights := types.NewTensor([]int{K, N}, types.DtypeFP16, 0)
	quantized := types.NewTensor([]int{K, N}, types.DtypeINT8, 0)
	scales := types.NewTensor([]int{N}, types.DtypeFP32, 0) // Per-column scales

	err = bindings.AllocateTensor(weights)
	require.NoError(t, err)
	defer bindings.FreeTensor(weights)

	err = bindings.AllocateTensor(quantized)
	require.NoError(t, err)
	defer bindings.FreeTensor(quantized)

	err = bindings.AllocateTensor(scales)
	require.NoError(t, err)
	defer bindings.FreeTensor(scales)

	// Fill weights
	rng := rand.New(rand.NewSource(42))
	data := make([]float32, K*N)
	for i := range data {
		data[i] = rng.Float32()*2 - 1
	}
	err = bindings.CopyToDevice(weights, data)
	require.NoError(t, err)

	// Quantize weights (per-column)
	err = bindings.QuantizeWeights(quantized, scales, weights)
	require.NoError(t, err, "QuantizeWeights failed")

	// Verify
	resultINT8 := make([]int8, K*N)
	resultScales := make([]float32, N)
	err = bindings.CopyToHostINT8(resultINT8, quantized)
	require.NoError(t, err)
	err = bindings.CopyToHost(resultScales, scales)
	require.NoError(t, err)

	// Check that scales are non-zero
	for j := 0; j < N; j++ {
		assert.NotZero(t, resultScales[j], "Scale for column %d is zero", j)
	}

	// Check reconstruction error
	maxError := float32(0)
	for i := 0; i < K; i++ {
		for j := 0; j < N; j++ {
			idx := i*N + j
			reconstructed := float32(resultINT8[idx]) * resultScales[j]
			err := float32(math.Abs(float64(data[idx] - reconstructed)))
			if err > maxError {
				maxError = err
			}
		}
	}
	t.Logf("Weight quantization: maxError=%f", maxError)
	// INT8 quantization inherently has 1/256 = ~0.4% quantization step error
	// Combined with FP16 precision, expect up to ~1% error
	assert.Less(t, float64(maxError), 0.01, "Weight quantization error too large: %f", maxError)
}

// TestQuantizeRoundtrip verifies full quantize-dequantize roundtrip.
func TestQuantizeRoundtrip(t *testing.T) {
	err := bindings.InitGPU(0)
	require.NoError(t, err)
	defer bindings.ShutdownGPU()

	size := 4096
	original := types.NewTensor([]int{size}, types.DtypeFP16, 0)
	quantized := types.NewTensor([]int{size}, types.DtypeINT8, 0)
	scale := types.NewTensor([]int{1}, types.DtypeFP32, 0)
	restored := types.NewTensor([]int{size}, types.DtypeFP16, 0)

	err = bindings.AllocateTensor(original)
	require.NoError(t, err)
	defer bindings.FreeTensor(original)

	err = bindings.AllocateTensor(quantized)
	require.NoError(t, err)
	defer bindings.FreeTensor(quantized)

	err = bindings.AllocateTensor(scale)
	require.NoError(t, err)
	defer bindings.FreeTensor(scale)

	err = bindings.AllocateTensor(restored)
	require.NoError(t, err)
	defer bindings.FreeTensor(restored)

	// Fill original
	rng := rand.New(rand.NewSource(42))
	data := make([]float32, size)
	for i := range data {
		data[i] = rng.Float32()*2 - 1
	}
	err = bindings.CopyToDevice(original, data)
	require.NoError(t, err)

	// Quantize
	err = bindings.QuantizePerTensor(quantized, scale, original)
	require.NoError(t, err)

	// Dequantize
	err = bindings.DequantizePerTensor(restored, quantized, scale)
	require.NoError(t, err)

	// Verify roundtrip
	result := make([]float32, size)
	err = bindings.CopyToHost(result, restored)
	require.NoError(t, err)

	var sumError float64
	for i := range data {
		sumError += math.Abs(float64(data[i] - result[i]))
	}
	avgError := sumError / float64(size)

	// Average error should be small
	assert.Less(t, avgError, 0.01, "Average roundtrip error too large: %f", avgError)
}
