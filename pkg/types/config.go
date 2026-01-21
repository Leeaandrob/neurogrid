// Package types provides core data types for the NeuroGrid engine.
package types

// LlamaConfig holds the configuration for a Llama model.
type LlamaConfig struct {
	HiddenSize       int     // Hidden dimension (4096 for 7B)
	IntermediateSize int     // FFN intermediate dimension (11008 for 7B)
	NumLayers        int     // Number of transformer layers (32 for 7B)
	NumHeads         int     // Number of attention heads (32 for 7B)
	NumKVHeads       int     // Number of KV heads (32 for 7B, different for 70B GQA)
	HeadDim          int     // Dimension per head (128)
	VocabSize        int     // Vocabulary size (32000)
	MaxSeqLen        int     // Maximum sequence length (4096)
	RMSNormEps       float32 // RMSNorm epsilon (1e-6)
}

// Llama7BConfig returns the standard configuration for Llama 2 7B.
func Llama7BConfig() *LlamaConfig {
	return &LlamaConfig{
		HiddenSize:       4096,
		IntermediateSize: 11008,
		NumLayers:        32,
		NumHeads:         32,
		NumKVHeads:       32,
		HeadDim:          128,
		VocabSize:        32000,
		MaxSeqLen:        4096,
		RMSNormEps:       1e-6,
	}
}

// Llama13BConfig returns the standard configuration for Llama 2 13B.
func Llama13BConfig() *LlamaConfig {
	return &LlamaConfig{
		HiddenSize:       5120,
		IntermediateSize: 13824,
		NumLayers:        40,
		NumHeads:         40,
		NumKVHeads:       40,
		HeadDim:          128,
		VocabSize:        32000,
		MaxSeqLen:        4096,
		RMSNormEps:       1e-6,
	}
}

// Llama70BConfig returns the standard configuration for Llama 2 70B.
func Llama70BConfig() *LlamaConfig {
	return &LlamaConfig{
		HiddenSize:       8192,
		IntermediateSize: 28672,
		NumLayers:        80,
		NumHeads:         64,
		NumKVHeads:       8, // GQA: 8 KV heads for 64 query heads
		HeadDim:          128,
		VocabSize:        32000,
		MaxSeqLen:        4096,
		RMSNormEps:       1e-6,
	}
}

// KVDim returns the dimension of keys/values based on KV heads.
func (c *LlamaConfig) KVDim() int {
	return c.NumKVHeads * c.HeadDim
}

// QueryDim returns the dimension of queries (same as hidden).
func (c *LlamaConfig) QueryDim() int {
	return c.NumHeads * c.HeadDim
}

// HeadRatio returns how many query heads per KV head (for GQA).
func (c *LlamaConfig) HeadRatio() int {
	return c.NumHeads / c.NumKVHeads
}
