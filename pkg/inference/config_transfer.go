// Package inference provides config transfer functionality for distributed inference.
// This file implements config serialization for P2P transfer to stateless workers.
package inference

import (
	"encoding/json"
	"errors"

	"github.com/neurogrid/engine/pkg/types"
)

// Error messages for config transfer operations.
var (
	ErrNilConfig       = errors.New("config is nil")
	ErrEmptyModelName  = errors.New("model name is empty")
	ErrEmptyData       = errors.New("empty data")
	ErrNullConfig      = errors.New("null config")
)

// TransferableConfig is the JSON-serializable model configuration
// sent from coordinator to workers before weight transfer.
//
// All fields use snake_case JSON tags to match Python/config file conventions.
// The struct maps 1:1 with types.LlamaConfig for lossless round-trip serialization.
type TransferableConfig struct {
	ModelName        string  `json:"model_name"`
	HiddenSize       int     `json:"hidden_size"`
	IntermediateSize int     `json:"intermediate_size"`
	NumLayers        int     `json:"num_layers"`
	NumHeads         int     `json:"num_heads"`
	NumKVHeads       int     `json:"num_kv_heads"`
	HeadDim          int     `json:"head_dim"`
	VocabSize        int     `json:"vocab_size"`
	MaxSeqLen        int     `json:"max_seq_len"`
	RMSNormEps       float32 `json:"rms_norm_eps"`
	RopeTheta        float32 `json:"rope_theta"` // RoPE base frequency (10000.0 for Llama 2, 1000000.0 for Mistral Nemo)

	// LFM2 hybrid architecture fields
	LayerTypes     []string `json:"layer_types,omitempty"`
	ConvKernelSize int      `json:"conv_kernel_size,omitempty"`
	ConvDim        int      `json:"conv_dim,omitempty"`
	ConvBias       bool     `json:"conv_bias,omitempty"`
	TieEmbeddings  bool     `json:"tie_embeddings,omitempty"`
	Dtype          string   `json:"dtype,omitempty"`
	ModelType      string   `json:"model_type,omitempty"`
}

// ToLlamaConfig converts TransferableConfig to types.LlamaConfig.
// This is used by workers after receiving config over P2P.
func (tc *TransferableConfig) ToLlamaConfig() *types.LlamaConfig {
	ropeTheta := tc.RopeTheta
	if ropeTheta == 0 {
		ropeTheta = 10000.0 // Default for Llama 2 compatibility
	}
	return &types.LlamaConfig{
		HiddenSize:       tc.HiddenSize,
		IntermediateSize: tc.IntermediateSize,
		NumLayers:        tc.NumLayers,
		NumHeads:         tc.NumHeads,
		NumKVHeads:       tc.NumKVHeads,
		HeadDim:          tc.HeadDim,
		VocabSize:        tc.VocabSize,
		MaxSeqLen:        tc.MaxSeqLen,
		RMSNormEps:       tc.RMSNormEps,
		RopeTheta:        ropeTheta,
		LayerTypes:       tc.LayerTypes,
		ConvKernelSize:   tc.ConvKernelSize,
		ConvDim:          tc.ConvDim,
		ConvBias:         tc.ConvBias,
		TieEmbeddings:    tc.TieEmbeddings,
		Dtype:            tc.Dtype,
		ModelType:        tc.ModelType,
	}
}

// FromLlamaConfig creates a TransferableConfig from types.LlamaConfig.
// This is used by coordinator before sending config over P2P.
func FromLlamaConfig(config *types.LlamaConfig, modelName string) *TransferableConfig {
	return &TransferableConfig{
		ModelName:        modelName,
		HiddenSize:       config.HiddenSize,
		IntermediateSize: config.IntermediateSize,
		NumLayers:        config.NumLayers,
		NumHeads:         config.NumHeads,
		NumKVHeads:       config.NumKVHeads,
		HeadDim:          config.HeadDim,
		VocabSize:        config.VocabSize,
		MaxSeqLen:        config.MaxSeqLen,
		RMSNormEps:       config.RMSNormEps,
		RopeTheta:        config.RopeTheta,
		LayerTypes:       config.LayerTypes,
		ConvKernelSize:   config.ConvKernelSize,
		ConvDim:          config.ConvDim,
		ConvBias:         config.ConvBias,
		TieEmbeddings:    config.TieEmbeddings,
		Dtype:            config.Dtype,
		ModelType:        config.ModelType,
	}
}

// SerializeConfig serializes a LlamaConfig to JSON for P2P transfer.
// Config is sent as a single message (< 1KB), not chunked.
//
// Returns ErrNilConfig if config is nil, ErrEmptyModelName if modelName is empty.
func SerializeConfig(config *types.LlamaConfig, modelName string) ([]byte, error) {
	if config == nil {
		return nil, ErrNilConfig
	}
	if modelName == "" {
		return nil, ErrEmptyModelName
	}

	tc := FromLlamaConfig(config, modelName)
	return json.Marshal(tc)
}

// DeserializeConfig deserializes JSON to a LlamaConfig and model name.
//
// Returns ErrEmptyData if data is empty, ErrNullConfig if data is JSON null.
// Returns json.Unmarshal errors for invalid JSON.
func DeserializeConfig(data []byte) (*types.LlamaConfig, string, error) {
	if len(data) == 0 {
		return nil, "", ErrEmptyData
	}

	// Handle JSON null explicitly for better error messages
	if string(data) == "null" {
		return nil, "", ErrNullConfig
	}

	var tc TransferableConfig
	if err := json.Unmarshal(data, &tc); err != nil {
		return nil, "", err
	}

	return tc.ToLlamaConfig(), tc.ModelName, nil
}
