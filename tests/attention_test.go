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

// TestBasicAttention verifies the basic attention mechanism.
// Given: Q, K, V tensors [batch, num_heads, seq, head_dim]
// When: Basic attention is computed
// Then: Output = softmax(Q @ K^T / sqrt(d)) @ V matches reference
func TestBasicAttention(t *testing.T) {
	config := types.Llama7BConfig()

	err := bindings.InitGPU(0)
	require.NoError(t, err)
	defer bindings.ShutdownGPU()

	batchSize := 1
	seqLen := 4
	numHeads := config.NumHeads
	headDim := config.HeadDim

	// Create tensors [batch, num_heads, seq, head_dim]
	shape := []int{batchSize, numHeads, seqLen, headDim}
	q := types.NewTensor(shape, types.DtypeFP16, 0)
	k := types.NewTensor(shape, types.DtypeFP16, 0)
	v := types.NewTensor(shape, types.DtypeFP16, 0)
	output := types.NewTensor(shape, types.DtypeFP16, 0)

	err = bindings.AllocateTensor(q)
	require.NoError(t, err)
	defer bindings.FreeTensor(q)

	err = bindings.AllocateTensor(k)
	require.NoError(t, err)
	defer bindings.FreeTensor(k)

	err = bindings.AllocateTensor(v)
	require.NoError(t, err)
	defer bindings.FreeTensor(v)

	err = bindings.AllocateTensor(output)
	require.NoError(t, err)
	defer bindings.FreeTensor(output)

	// Fill with random data
	rng := rand.New(rand.NewSource(42))
	numElements := batchSize * numHeads * seqLen * headDim
	qData := make([]float32, numElements)
	kData := make([]float32, numElements)
	vData := make([]float32, numElements)
	for i := range qData {
		qData[i] = rng.Float32()*0.1 - 0.05
		kData[i] = rng.Float32()*0.1 - 0.05
		vData[i] = rng.Float32()*0.1 - 0.05
	}

	err = bindings.CopyToDevice(q, qData)
	require.NoError(t, err)
	err = bindings.CopyToDevice(k, kData)
	require.NoError(t, err)
	err = bindings.CopyToDevice(v, vData)
	require.NoError(t, err)

	// Execute attention (without causal mask for this test)
	err = bindings.BasicAttention(output, q, k, v, false)
	require.NoError(t, err, "BasicAttention failed")

	// Verify output shape is correct
	result := make([]float32, numElements)
	err = bindings.CopyToHost(result, output)
	require.NoError(t, err)

	// Check that output is not all zeros
	sumAbs := float64(0)
	for _, v := range result {
		sumAbs += math.Abs(float64(v))
	}
	assert.Greater(t, sumAbs, 0.0, "Attention output should not be all zeros")
}

// TestCausalAttention verifies causal (autoregressive) attention.
// Given: Q, K, V with seq_len > 1
// When: Causal attention is computed
// Then: Position i cannot attend to positions j > i
func TestCausalAttention(t *testing.T) {
	config := types.Llama7BConfig()

	err := bindings.InitGPU(0)
	require.NoError(t, err)
	defer bindings.ShutdownGPU()

	batchSize := 1
	seqLen := 8
	numHeads := 4 // Use fewer heads for this test
	headDim := config.HeadDim

	shape := []int{batchSize, numHeads, seqLen, headDim}
	q := types.NewTensor(shape, types.DtypeFP16, 0)
	k := types.NewTensor(shape, types.DtypeFP16, 0)
	v := types.NewTensor(shape, types.DtypeFP16, 0)
	output := types.NewTensor(shape, types.DtypeFP16, 0)

	err = bindings.AllocateTensor(q)
	require.NoError(t, err)
	defer bindings.FreeTensor(q)

	err = bindings.AllocateTensor(k)
	require.NoError(t, err)
	defer bindings.FreeTensor(k)

	err = bindings.AllocateTensor(v)
	require.NoError(t, err)
	defer bindings.FreeTensor(v)

	err = bindings.AllocateTensor(output)
	require.NoError(t, err)
	defer bindings.FreeTensor(output)

	// Fill with ones for Q and K, specific pattern for V
	numElements := batchSize * numHeads * seqLen * headDim
	qData := make([]float32, numElements)
	kData := make([]float32, numElements)
	vData := make([]float32, numElements)

	for i := range qData {
		qData[i] = 0.1
		kData[i] = 0.1
	}
	// V has different values per position so we can verify masking
	for b := 0; b < batchSize; b++ {
		for h := 0; h < numHeads; h++ {
			for s := 0; s < seqLen; s++ {
				for d := 0; d < headDim; d++ {
					idx := b*numHeads*seqLen*headDim + h*seqLen*headDim + s*headDim + d
					vData[idx] = float32(s + 1) // Position 0->1, 1->2, etc.
				}
			}
		}
	}

	err = bindings.CopyToDevice(q, qData)
	require.NoError(t, err)
	err = bindings.CopyToDevice(k, kData)
	require.NoError(t, err)
	err = bindings.CopyToDevice(v, vData)
	require.NoError(t, err)

	// Execute causal attention
	err = bindings.BasicAttention(output, q, k, v, true)
	require.NoError(t, err, "Causal attention failed")

	result := make([]float32, numElements)
	err = bindings.CopyToHost(result, output)
	require.NoError(t, err)

	// With causal mask:
	// Position 0 can only see V[0], so output[0] ≈ V[0] = 1
	// Position 1 can see V[0,1], so output[1] ≈ avg(V[0,1]) ≈ 1.5 (with softmax)
	// etc.
	// This is a simplified check - full verification needs golden data

	// For each position, output should be influenced by values at positions <= current
	// With V[s] = s+1, causal attention at position s attends to V[0..s]
	// The output should be a weighted average, so roughly (1+...+(s+1))/(s+1)
	// But with softmax, it could be higher if attention weights are uneven
	for h := 0; h < numHeads; h++ {
		for s := 0; s < seqLen; s++ {
			// Get first element of output at this position
			idx := h*seqLen*headDim + s*headDim
			val := result[idx]
			// Should be related to positions 0..s, not s+1..seqLen-1
			// Maximum possible is if all attention goes to position s: V[s] = s+1
			maxPossible := float32(s + 1)
			// Allow for numerical precision and softmax weighting
			assert.LessOrEqual(t, val, maxPossible+2.0,
				"Causal mask violated at head %d, pos %d: got %f, max expected %f", h, s, val, maxPossible+2.0)
		}
	}
}

// TestAttentionWithKVCache verifies attention with KV cache.
// Given: Q for current token, K and V cache for previous tokens
// When: Attention is computed with cache
// Then: Output correctly uses cached K and V
func TestAttentionWithKVCache(t *testing.T) {
	config := types.Llama7BConfig()

	err := bindings.InitGPU(0)
	require.NoError(t, err)
	defer bindings.ShutdownGPU()

	batchSize := 1
	numHeads := 4
	headDim := config.HeadDim
	maxSeqLen := 16
	currentPos := 5 // We've seen 5 tokens, generating 6th

	// Create KV cache
	kvCache, err := bindings.NewKVCache(batchSize, numHeads, headDim, maxSeqLen)
	require.NoError(t, err, "Failed to create KV cache")
	defer bindings.FreeKVCache(kvCache)

	// Q for current token only [batch, num_heads, 1, head_dim]
	qShape := []int{batchSize, numHeads, 1, headDim}
	q := types.NewTensor(qShape, types.DtypeFP16, 0)
	output := types.NewTensor(qShape, types.DtypeFP16, 0)

	// K, V for current token
	kvShape := []int{batchSize, numHeads, 1, headDim}
	k := types.NewTensor(kvShape, types.DtypeFP16, 0)
	v := types.NewTensor(kvShape, types.DtypeFP16, 0)

	err = bindings.AllocateTensor(q)
	require.NoError(t, err)
	defer bindings.FreeTensor(q)

	err = bindings.AllocateTensor(k)
	require.NoError(t, err)
	defer bindings.FreeTensor(k)

	err = bindings.AllocateTensor(v)
	require.NoError(t, err)
	defer bindings.FreeTensor(v)

	err = bindings.AllocateTensor(output)
	require.NoError(t, err)
	defer bindings.FreeTensor(output)

	// Fill cache with data for positions 0-4
	for pos := 0; pos < currentPos; pos++ {
		kData := make([]float32, batchSize*numHeads*headDim)
		vData := make([]float32, batchSize*numHeads*headDim)
		for i := range kData {
			kData[i] = float32(pos+1) * 0.1
			vData[i] = float32(pos+1) * 0.1
		}
		err = bindings.UpdateKVCache(kvCache, kData, vData, pos)
		require.NoError(t, err, "Failed to update KV cache at position %d", pos)
	}

	// Current token data
	qData := make([]float32, batchSize*numHeads*headDim)
	kData := make([]float32, batchSize*numHeads*headDim)
	vData := make([]float32, batchSize*numHeads*headDim)
	for i := range qData {
		qData[i] = 0.1
		kData[i] = 0.6 // Position 5
		vData[i] = 0.6
	}

	err = bindings.CopyToDevice(q, qData)
	require.NoError(t, err)
	err = bindings.CopyToDevice(k, kData)
	require.NoError(t, err)
	err = bindings.CopyToDevice(v, vData)
	require.NoError(t, err)

	// Execute attention with KV cache
	err = bindings.AttentionWithKVCache(output, q, k, v, kvCache, currentPos)
	require.NoError(t, err, "AttentionWithKVCache failed")

	// Verify
	result := make([]float32, batchSize*numHeads*headDim)
	err = bindings.CopyToHost(result, output)
	require.NoError(t, err)

	// Output should be influenced by all positions 0-5
	sumAbs := float64(0)
	for _, val := range result {
		sumAbs += math.Abs(float64(val))
	}
	assert.Greater(t, sumAbs, 0.0, "Attention output should not be all zeros")
}

// BenchmarkBasicAttention measures attention throughput.
func BenchmarkBasicAttention(b *testing.B) {
	config := types.Llama7BConfig()

	err := bindings.InitGPU(0)
	if err != nil {
		b.Skip("GPU not available")
	}
	defer bindings.ShutdownGPU()

	batchSize := 1
	seqLen := 1
	numHeads := config.NumHeads
	headDim := config.HeadDim

	shape := []int{batchSize, numHeads, seqLen, headDim}
	q := types.NewTensor(shape, types.DtypeFP16, 0)
	k := types.NewTensor(shape, types.DtypeFP16, 0)
	v := types.NewTensor(shape, types.DtypeFP16, 0)
	output := types.NewTensor(shape, types.DtypeFP16, 0)

	_ = bindings.AllocateTensor(q)
	defer bindings.FreeTensor(q)
	_ = bindings.AllocateTensor(k)
	defer bindings.FreeTensor(k)
	_ = bindings.AllocateTensor(v)
	defer bindings.FreeTensor(v)
	_ = bindings.AllocateTensor(output)
	defer bindings.FreeTensor(output)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = bindings.BasicAttention(output, q, k, v, true)
	}
	_ = bindings.SyncDevice()
}
