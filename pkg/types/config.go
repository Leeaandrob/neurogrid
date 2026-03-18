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
	RopeTheta        float32 // RoPE base frequency (10000.0 for Llama 2, 1000000.0 for Mistral Nemo)

	// LFM2 hybrid architecture support
	// Zero-value defaults preserve existing Llama behavior.
	LayerTypes     []string // "conv" or "full_attention" per layer. nil = all attention (backward compat)
	ConvKernelSize int      // conv_L_cache from config.json (default 0 = no conv)
	ConvDim        int      // conv_dim (typically == HiddenSize)
	ConvBias       bool     // conv_bias (default false)
	TieEmbeddings  bool     // tie_word_embeddings (lm_head == embed_tokens)
	Dtype          string   // "fp16", "bf16", "int8" (default "" = fp16)
	ModelType      string   // "llama", "lfm2", etc.
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
		RopeTheta:        10000.0,
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
		RopeTheta:        10000.0,
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
		RopeTheta:        10000.0,
	}
}

// Llama3_70BConfig returns the configuration for Llama 3.3 70B Instruct.
func Llama3_70BConfig() *LlamaConfig {
	return &LlamaConfig{
		HiddenSize:       8192,
		IntermediateSize: 28672,
		NumLayers:        80,
		NumHeads:         64,
		NumKVHeads:       8, // GQA: 8 KV heads for 64 query heads
		HeadDim:          128,
		VocabSize:        128256, // Llama 3 has larger vocab
		MaxSeqLen:        8192,   // 8K default, supports up to 128K
		RMSNormEps:       1e-5,   // Llama 3 uses 1e-5
		RopeTheta:        500000.0, // Llama 3 uses 500000
	}
}

// TinyLlamaConfig returns the configuration for TinyLlama 1.1B.
func TinyLlamaConfig() *LlamaConfig {
	return &LlamaConfig{
		HiddenSize:       2048,
		IntermediateSize: 5632,
		NumLayers:        22,
		NumHeads:         32,
		NumKVHeads:       4, // GQA: 4 KV heads
		HeadDim:          64,
		VocabSize:        32000,
		MaxSeqLen:        2048,
		RMSNormEps:       1e-6,
		RopeTheta:        10000.0,
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

// IsConvLayer returns true if the given layer is a convolution layer.
// Returns false when LayerTypes is nil (backward compat: all attention).
func (c *LlamaConfig) IsConvLayer(layerID int) bool {
	if c.LayerTypes == nil {
		return false
	}
	if layerID < 0 || layerID >= len(c.LayerTypes) {
		return false
	}
	return c.LayerTypes[layerID] == "conv"
}

// IsAttentionLayer returns true if the given layer is an attention layer.
func (c *LlamaConfig) IsAttentionLayer(layerID int) bool {
	return !c.IsConvLayer(layerID)
}

// NumConvLayers returns the number of convolution layers.
func (c *LlamaConfig) NumConvLayers() int {
	count := 0
	for _, lt := range c.LayerTypes {
		if lt == "conv" {
			count++
		}
	}
	return count
}

// NumAttentionLayers returns the number of attention layers.
func (c *LlamaConfig) NumAttentionLayers() int {
	return c.NumLayers - c.NumConvLayers()
}

// LFM2_1_2BThinkingConfig returns the configuration for LFM2.5-1.2B-Thinking.
// Layer types are interleaved: conv,conv,attn,conv,conv,attn,conv,conv,attn,conv,attn,conv,attn,conv,attn,conv
func LFM2_1_2BThinkingConfig() *LlamaConfig {
	layerTypes := []string{
		"conv", "conv", "full_attention",
		"conv", "conv", "full_attention",
		"conv", "conv", "full_attention",
		"conv", "full_attention",
		"conv", "full_attention",
		"conv", "full_attention",
		"conv",
	}

	return &LlamaConfig{
		HiddenSize:       2048,
		IntermediateSize: 8192, // SwiGLU: 2/3 of block_ff_dim (12288)
		NumLayers:        16,
		NumHeads:         32,
		NumKVHeads:       8,
		HeadDim:          64,
		VocabSize:        65536,
		MaxSeqLen:        8192,
		RMSNormEps:       1e-5,
		RopeTheta:        1000000.0,
		LayerTypes:       layerTypes,
		ConvKernelSize:   3,
		ConvDim:          2048,
		ConvBias:         false,
		TieEmbeddings:    true,
		Dtype:            "bf16",
		ModelType:        "lfm2",
	}
}
