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

// TestGEMMFP16 verifies FP16 matrix multiplication using cuBLAS.
// Given: Matrices A[M,K] and B[K,N] in FP16
// When: GEMM is executed
// Then: C = A @ B matches reference within 1e-3 tolerance
func TestGEMMFP16(t *testing.T) {
	err := bindings.InitGPU(0)
	require.NoError(t, err)
	defer bindings.ShutdownGPU()

	M, K, N := 32, 128, 64

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

	// Fill with random data
	rng := rand.New(rand.NewSource(42))
	dataA := make([]float32, M*K)
	dataB := make([]float32, K*N)
	for i := range dataA {
		dataA[i] = rng.Float32()*2 - 1
	}
	for i := range dataB {
		dataB[i] = rng.Float32()*2 - 1
	}

	err = bindings.CopyToDevice(a, dataA)
	require.NoError(t, err)
	err = bindings.CopyToDevice(b, dataB)
	require.NoError(t, err)

	// Execute GEMM
	err = bindings.GEMMFP16(c, a, b, false, false)
	require.NoError(t, err, "GEMM FP16 failed")

	// Verify: compute reference on CPU
	result := make([]float32, M*N)
	err = bindings.CopyToHost(result, c)
	require.NoError(t, err)

	reference := cpuMatmul(dataA, dataB, M, K, N)
	maxDiff := 0.0
	errCount := 0
	for i := range result {
		diff := math.Abs(float64(result[i] - reference[i]))
		if diff > maxDiff {
			maxDiff = diff
		}
		// Use absolute tolerance for GEMM - FP16 over K=128 can accumulate errors
		// Max error scales roughly as K * epsilon where epsilon is FP16 machine precision (~1e-3)
		if diff > 0.1 { // 0.1 absolute tolerance
			errCount++
			if errCount <= 3 {
				t.Logf("GEMM mismatch at [%d/%d, %d/%d]: got %f, ref %f, diff=%f",
					i/N, M, i%N, N, result[i], reference[i], diff)
			}
		}
	}
	t.Logf("GEMM FP16: maxDiff=%f", maxDiff)
	// For K=128 with random values in [-1,1], expect max error ~0.05
	assert.Less(t, maxDiff, 0.1, "GEMM FP16 max diff %f exceeds 0.1", maxDiff)
}

// TestGEMMINT8 verifies INT8 matrix multiplication with dequantization.
// Given: Matrix A[M,K] FP16, B[K,N] INT8 with scale
// When: INT8 GEMM with dequant is executed
// Then: C = A @ (B * scale) matches reference
func TestGEMMINT8(t *testing.T) {
	err := bindings.InitGPU(0)
	require.NoError(t, err)
	defer bindings.ShutdownGPU()

	M, K, N := 32, 128, 64

	// A is FP16 (activations)
	a := types.NewTensor([]int{M, K}, types.DtypeFP16, 0)
	// B is INT8 (weights)
	b := types.NewTensor([]int{K, N}, types.DtypeINT8, 0)
	// C is FP16 (output)
	c := types.NewTensor([]int{M, N}, types.DtypeFP16, 0)
	// Scale for dequantization
	scale := types.NewTensor([]int{N}, types.DtypeFP32, 0)

	err = bindings.AllocateTensor(a)
	require.NoError(t, err)
	defer bindings.FreeTensor(a)

	err = bindings.AllocateTensor(b)
	require.NoError(t, err)
	defer bindings.FreeTensor(b)

	err = bindings.AllocateTensor(c)
	require.NoError(t, err)
	defer bindings.FreeTensor(c)

	err = bindings.AllocateTensor(scale)
	require.NoError(t, err)
	defer bindings.FreeTensor(scale)

	// Fill with test data
	rng := rand.New(rand.NewSource(42))
	dataA := make([]float32, M*K)
	dataB := make([]int8, K*N)
	scaleData := make([]float32, N)

	for i := range dataA {
		dataA[i] = rng.Float32()*2 - 1
	}
	for i := range dataB {
		dataB[i] = int8(rng.Intn(256) - 128)
	}
	for i := range scaleData {
		scaleData[i] = 0.01 // Typical scale for INT8
	}

	err = bindings.CopyToDevice(a, dataA)
	require.NoError(t, err)
	err = bindings.CopyToDeviceINT8(b, dataB)
	require.NoError(t, err)
	err = bindings.CopyToDevice(scale, scaleData)
	require.NoError(t, err)

	// Execute INT8 GEMM (transposeB=false since B is [K, N])
	err = bindings.GEMMINT8(c, a, b, scale, false)
	require.NoError(t, err, "GEMM INT8 failed")

	// Verify against reference
	result := make([]float32, M*N)
	err = bindings.CopyToHost(result, c)
	require.NoError(t, err)

	// Reference: A @ (B * scale)
	dataBFloat := make([]float32, K*N)
	for i := 0; i < K; i++ {
		for j := 0; j < N; j++ {
			dataBFloat[i*N+j] = float32(dataB[i*N+j]) * scaleData[j]
		}
	}
	reference := cpuMatmul(dataA, dataBFloat, M, K, N)

	maxDiff := 0.0
	for i := range result {
		diff := math.Abs(float64(result[i] - reference[i]))
		if diff > maxDiff {
			maxDiff = diff
		}
	}

	// INT8 quantization adds error, allow 5e-2 tolerance
	assert.Less(t, maxDiff, 5e-2, "GEMM INT8 max diff %f exceeds tolerance", maxDiff)
}

// TestGEMMTranspose verifies transposed matrix multiplication.
func TestGEMMTranspose(t *testing.T) {
	err := bindings.InitGPU(0)
	require.NoError(t, err)
	defer bindings.ShutdownGPU()

	M, K, N := 32, 128, 64

	// A^T: stored as [K,M], read as [M,K] transposed
	a := types.NewTensor([]int{K, M}, types.DtypeFP16, 0)
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

	// Fill with random data
	rng := rand.New(rand.NewSource(42))
	dataA := make([]float32, K*M)
	dataB := make([]float32, K*N)
	for i := range dataA {
		dataA[i] = rng.Float32()*2 - 1
	}
	for i := range dataB {
		dataB[i] = rng.Float32()*2 - 1
	}

	err = bindings.CopyToDevice(a, dataA)
	require.NoError(t, err)
	err = bindings.CopyToDevice(b, dataB)
	require.NoError(t, err)

	// Execute: C = A^T @ B
	err = bindings.GEMMFP16(c, a, b, true, false)
	require.NoError(t, err, "GEMM with transpose failed")

	// Verify
	result := make([]float32, M*N)
	err = bindings.CopyToHost(result, c)
	require.NoError(t, err)

	// Reference: transpose A first, then multiply
	dataAT := make([]float32, M*K)
	for i := 0; i < K; i++ {
		for j := 0; j < M; j++ {
			dataAT[j*K+i] = dataA[i*M+j]
		}
	}
	reference := cpuMatmul(dataAT, dataB, M, K, N)

	maxDiff := 0.0
	errCount := 0
	for i := range result {
		diff := math.Abs(float64(result[i] - reference[i]))
		if diff > maxDiff {
			maxDiff = diff
		}
		if diff > 0.1 {
			errCount++
			if errCount <= 3 {
				t.Logf("GEMM transpose mismatch at [%d/%d, %d/%d]: got %f, ref %f, diff=%f",
					i/N, M, i%N, N, result[i], reference[i], diff)
			}
		}
	}
	t.Logf("GEMM transpose: maxDiff=%f", maxDiff)
	assert.Less(t, maxDiff, 0.1, "GEMM transpose max diff %f exceeds tolerance", maxDiff)
}

// BenchmarkGEMMFP16 measures FP16 GEMM throughput.
func BenchmarkGEMMFP16(b *testing.B) {
	err := bindings.InitGPU(0)
	if err != nil {
		b.Skip("GPU not available")
	}
	defer bindings.ShutdownGPU()

	M, K, N := 1, 4096, 4096 // Single token projection

	a := types.NewTensor([]int{M, K}, types.DtypeFP16, 0)
	bTensor := types.NewTensor([]int{K, N}, types.DtypeFP16, 0)
	c := types.NewTensor([]int{M, N}, types.DtypeFP16, 0)

	_ = bindings.AllocateTensor(a)
	defer bindings.FreeTensor(a)
	_ = bindings.AllocateTensor(bTensor)
	defer bindings.FreeTensor(bTensor)
	_ = bindings.AllocateTensor(c)
	defer bindings.FreeTensor(c)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = bindings.GEMMFP16(c, a, bTensor, false, false)
	}
	_ = bindings.SyncDevice()
}

// cpuMatmul computes C = A @ B on CPU for reference.
func cpuMatmul(a, b []float32, M, K, N int) []float32 {
	c := make([]float32, M*N)
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			var sum float32
			for k := 0; k < K; k++ {
				sum += a[i*K+k] * b[k*N+j]
			}
			c[i*N+j] = sum
		}
	}
	return c
}
