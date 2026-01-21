// Package model provides model loading and weight management for LLM inference.
package model

import (
	"github.com/neurogrid/engine/pkg/types"
)

// ModelLoader is the common interface for all weight loaders.
// Both WeightLoader and MmapLoader implement this interface.
type ModelLoader interface {
	// LoadTensor loads a tensor by name, returning the raw bytes and metadata.
	LoadTensor(name string) ([]byte, *TensorInfo, error)

	// GetTensorInfo returns metadata about a tensor without loading its data.
	GetTensorInfo(name string) (*TensorInfo, bool)

	// ListTensors returns all tensor names in the model.
	ListTensors() []string

	// CountLayers returns the number of transformer layers in the model.
	CountLayers() int

	// Close releases all resources held by the loader.
	Close() error
}

// LoadedModel represents a complete loaded Llama model.
type LoadedModel struct {
	Config     *types.LlamaConfig
	Embedding  *types.Tensor // Token embeddings [vocab_size, hidden_size]
	Layers     []*LoadedLayerWeights
	OutputNorm *types.Tensor // Final RMSNorm
	LMHead     *types.Tensor // Output projection [vocab_size, hidden_size]
}

// LoadedLayerWeights holds all weight tensors for a single transformer layer
// with GPU tensor references.
type LoadedLayerWeights struct {
	LayerID int

	// Attention weights
	AttnNorm *types.Tensor // RMSNorm [hidden_size]
	QProj    *types.Tensor // Query projection [num_heads * head_dim, hidden_size]
	KProj    *types.Tensor // Key projection [num_kv_heads * head_dim, hidden_size]
	VProj    *types.Tensor // Value projection [num_kv_heads * head_dim, hidden_size]
	OProj    *types.Tensor // Output projection [hidden_size, num_heads * head_dim]

	// INT8 quantization scales (optional, nil if FP16)
	QScale *types.Tensor
	KScale *types.Tensor
	VScale *types.Tensor
	OScale *types.Tensor

	// FFN weights
	FFNNorm  *types.Tensor // RMSNorm [hidden_size]
	GateProj *types.Tensor // Gate projection [intermediate_size, hidden_size]
	UpProj   *types.Tensor // Up projection [intermediate_size, hidden_size]
	DownProj *types.Tensor // Down projection [hidden_size, intermediate_size]

	// INT8 quantization scales for FFN (optional)
	GateScale *types.Tensor
	UpScale   *types.Tensor
	DownScale *types.Tensor
}

// WeightNameMapping maps HuggingFace tensor names to internal names.
// This helps with loading models from different sources.
var WeightNameMapping = map[string]string{
	// Embedding
	"model.embed_tokens.weight": "embedding",
	"lm_head.weight":            "lm_head",
	"model.norm.weight":         "output_norm",

	// Layer weights (templated - use with layer prefix)
	"input_layernorm.weight":          "attn_norm",
	"post_attention_layernorm.weight": "ffn_norm",
	"self_attn.q_proj.weight":         "q_proj",
	"self_attn.k_proj.weight":         "k_proj",
	"self_attn.v_proj.weight":         "v_proj",
	"self_attn.o_proj.weight":         "o_proj",
	"mlp.gate_proj.weight":            "gate_proj",
	"mlp.up_proj.weight":              "up_proj",
	"mlp.down_proj.weight":            "down_proj",

	// INT8 scales (if present)
	"self_attn.q_proj.weight_scale": "q_scale",
	"self_attn.k_proj.weight_scale": "k_scale",
	"self_attn.v_proj.weight_scale": "v_scale",
	"self_attn.o_proj.weight_scale": "o_scale",
	"mlp.gate_proj.weight_scale":    "gate_scale",
	"mlp.up_proj.weight_scale":      "up_scale",
	"mlp.down_proj.weight_scale":    "down_scale",
}

// WeightFormat represents the format of weight tensors.
type WeightFormat string

const (
	WeightFormatFP32 WeightFormat = "F32"
	WeightFormatFP16 WeightFormat = "F16"
	WeightFormatBF16 WeightFormat = "BF16"
	WeightFormatINT8 WeightFormat = "I8"
)

// ByteSize returns the byte size per element for a weight format.
func (f WeightFormat) ByteSize() int {
	switch f {
	case WeightFormatFP32:
		return 4
	case WeightFormatFP16, WeightFormatBF16:
		return 2
	case WeightFormatINT8:
		return 1
	default:
		return 0
	}
}

// ToDtype converts WeightFormat to types.Dtype.
func (f WeightFormat) ToDtype() types.Dtype {
	switch f {
	case WeightFormatFP32:
		return types.DtypeFP32
	case WeightFormatFP16, WeightFormatBF16:
		return types.DtypeFP16
	case WeightFormatINT8:
		return types.DtypeINT8
	default:
		return types.DtypeFP32
	}
}

// ParseWeightFormat parses a dtype string into a WeightFormat.
func ParseWeightFormat(dtype string) WeightFormat {
	switch dtype {
	case "F32":
		return WeightFormatFP32
	case "F16":
		return WeightFormatFP16
	case "BF16":
		return WeightFormatBF16
	case "I8":
		return WeightFormatINT8
	default:
		return WeightFormatFP16 // Default to FP16
	}
}

// NewLoader creates a new model loader based on the loading strategy.
// useMmap enables memory-mapped loading for large models.
func NewLoader(basePath string, useMmap bool) (ModelLoader, error) {
	if useMmap {
		return NewMmapLoader(basePath)
	}
	return NewWeightLoader(basePath)
}

// Ensure both loaders implement ModelLoader
var _ ModelLoader = (*WeightLoader)(nil)
var _ ModelLoader = (*MmapLoader)(nil)
