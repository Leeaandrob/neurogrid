// Package model provides model loading and configuration detection.
package model

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// RoPE style constants (must match CUDA kernels.h)
const (
	RoPEStyleSplitHalf   = 0 // Llama 2, TinyLlama, Mistral: pairs (0,64), (1,65)...
	RoPEStyleInterleaved = 1 // Llama 3, GPT-NeoX: pairs (0,1), (2,3)...
)

// ModelConfig holds model-specific configuration detected from config files.
type ModelConfig struct {
	// Model architecture
	ModelType       string `json:"model_type"`
	HiddenSize      int    `json:"hidden_size"`
	IntermediateSize int   `json:"intermediate_size"`
	NumHiddenLayers int    `json:"num_hidden_layers"`
	NumAttentionHeads int  `json:"num_attention_heads"`
	NumKeyValueHeads int   `json:"num_key_value_heads"`
	VocabSize       int    `json:"vocab_size"`
	MaxPositionEmbeddings int `json:"max_position_embeddings"`
	RMSNormEps      float64 `json:"rms_norm_eps"`
	RopeTheta       float64 `json:"rope_theta"`

	// LFM2 hybrid architecture fields
	LayerTypes        []string    `json:"layer_types"`
	ConvLCache        int         `json:"conv_L_cache"`
	ConvDim           int         `json:"conv_dim"`
	ConvBias          bool        `json:"conv_bias"`
	TieEmbedding      bool        `json:"tie_embedding"`
	TieWordEmbeddings bool        `json:"tie_word_embeddings"`
	Dtype             string      `json:"dtype"`
	TorchDtype        string      `json:"torch_dtype"`
	NormEps           float64     `json:"norm_eps"`
	EOSTokenID        interface{} `json:"eos_token_id"`
	BOSTokenID        interface{} `json:"bos_token_id"`

	// Detected settings
	RoPEStyle       int             // RoPEStyleSplitHalf or RoPEStyleInterleaved
	ChatTemplate    ChatTemplate    // Detected chat template
	ChatTemplateRaw string          // Raw Jinja2 template from tokenizer_config.json
}

// TokenizerConfig represents the tokenizer_config.json structure.
type TokenizerConfig struct {
	ChatTemplate string `json:"chat_template"`
	EOSToken     interface{} `json:"eos_token"` // Can be string or object
	BOSToken     interface{} `json:"bos_token"` // Can be string or object
	ModelType    string `json:"model_type"`
}

// LoadModelConfig loads and detects model configuration from the model directory.
func LoadModelConfig(modelPath string) (*ModelConfig, error) {
	config := &ModelConfig{}

	// Load config.json
	configPath := filepath.Join(modelPath, "config.json")
	configData, err := os.ReadFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read config.json: %w", err)
	}

	if err := json.Unmarshal(configData, config); err != nil {
		return nil, fmt.Errorf("failed to parse config.json: %w", err)
	}

	// Load tokenizer_config.json for chat template
	tokenizerConfigPath := filepath.Join(modelPath, "tokenizer_config.json")
	if tokenizerData, err := os.ReadFile(tokenizerConfigPath); err == nil {
		var tokConfig TokenizerConfig
		if err := json.Unmarshal(tokenizerData, &tokConfig); err == nil {
			config.ChatTemplateRaw = tokConfig.ChatTemplate
		}
	}

	// Detect RoPE style based on model type
	config.RoPEStyle = DetectRoPEStyle(config.ModelType)

	// Detect chat template
	config.ChatTemplate = DetectChatTemplate(config.ModelType, config.ChatTemplateRaw)

	return config, nil
}

// DetectRoPEStyle detects the RoPE pairing style based on model type.
func DetectRoPEStyle(modelType string) int {
	lower := strings.ToLower(modelType)

	// LFM2 uses interleaved RoPE with theta=1000000
	if strings.Contains(lower, "lfm") {
		return RoPEStyleInterleaved
	}

	// Llama 3 and GPT-NeoX use interleaved pairing
	if strings.Contains(lower, "llama3") ||
	   strings.Contains(lower, "llama-3") ||
	   strings.Contains(lower, "gpt-neox") ||
	   strings.Contains(lower, "falcon") ||
	   strings.Contains(lower, "mpt") {
		return RoPEStyleInterleaved
	}

	// Default to split-half (Llama 2, TinyLlama, Mistral, etc.)
	return RoPEStyleSplitHalf
}

// DetectChatTemplate detects the appropriate chat template for a model.
func DetectChatTemplate(modelType string, rawTemplate string) ChatTemplate {
	lower := strings.ToLower(modelType)

	// Check for specific model types first
	if strings.Contains(lower, "lfm") {
		return NewChatMLTemplate()
	}

	if strings.Contains(lower, "tinyllama") {
		return NewTinyLlamaChatTemplate()
	}

	if strings.Contains(lower, "llama-3") || strings.Contains(lower, "llama3") {
		return NewLlama3ChatTemplate()
	}

	if strings.Contains(lower, "mistral") || strings.Contains(lower, "mixtral") {
		return NewMistralChatTemplate()
	}

	// Try to detect from raw template
	if rawTemplate != "" {
		if strings.Contains(rawTemplate, "<|im_start|>") {
			return NewChatMLTemplate()
		}
		if strings.Contains(rawTemplate, "<|user|>") && strings.Contains(rawTemplate, "<|assistant|>") {
			// ChatML style (TinyLlama, Zephyr, etc.)
			return NewTinyLlamaChatTemplate()
		}
		if strings.Contains(rawTemplate, "<|start_header_id|>") {
			// Llama 3 style
			return NewLlama3ChatTemplate()
		}
		if strings.Contains(rawTemplate, "[INST]") {
			// Llama 2 / Mistral style
			if strings.Contains(rawTemplate, "<<SYS>>") {
				return NewLlama2ChatTemplate()
			}
			return NewMistralChatTemplate()
		}
	}

	// Default to Llama 2 format
	return NewLlama2ChatTemplate()
}

// MistralChatTemplate formats messages according to Mistral chat format.
type MistralChatTemplate struct{}

// NewMistralChatTemplate creates a new Mistral chat template.
func NewMistralChatTemplate() *MistralChatTemplate {
	return &MistralChatTemplate{}
}

// Format formats messages according to Mistral chat format.
// Mistral uses [INST] [/INST] without <<SYS>> tags.
//
// Format:
//
//	<s>[INST] {system}\n\n{user} [/INST] {assistant}</s>[INST] {user2} [/INST]
func (t *MistralChatTemplate) Format(messages []Message) string {
	var result strings.Builder
	result.WriteString("<s>")

	var systemMsg string
	for _, msg := range messages {
		if msg.Role == "system" {
			systemMsg = msg.Content
			break
		}
	}

	firstUser := true
	for _, msg := range messages {
		switch msg.Role {
		case "system":
			// System message is prepended to first user message
			continue
		case "user":
			if !firstUser {
				result.WriteString("[INST] ")
			} else {
				result.WriteString("[INST] ")
				if systemMsg != "" {
					result.WriteString(systemMsg)
					result.WriteString("\n\n")
				}
				firstUser = false
			}
			result.WriteString(msg.Content)
			result.WriteString(" [/INST] ")
		case "assistant":
			result.WriteString(msg.Content)
			result.WriteString("</s>")
		}
	}

	return result.String()
}

// GetRoPEStyleName returns a human-readable name for the RoPE style.
func GetRoPEStyleName(style int) string {
	switch style {
	case RoPEStyleSplitHalf:
		return "split-half"
	case RoPEStyleInterleaved:
		return "interleaved"
	default:
		return "unknown"
	}
}

// ModelFamily represents known model families and their configurations.
type ModelFamily struct {
	Name      string
	RoPEStyle int
	ChatTemplate func() ChatTemplate
}

// KnownModelFamilies contains configurations for known model families.
var KnownModelFamilies = map[string]ModelFamily{
	"llama2": {
		Name:      "Llama 2",
		RoPEStyle: RoPEStyleSplitHalf,
		ChatTemplate: func() ChatTemplate { return NewLlama2ChatTemplate() },
	},
	"llama3": {
		Name:      "Llama 3",
		RoPEStyle: RoPEStyleInterleaved,
		ChatTemplate: func() ChatTemplate { return NewLlama3ChatTemplate() },
	},
	"tinyllama": {
		Name:      "TinyLlama",
		RoPEStyle: RoPEStyleSplitHalf,
		ChatTemplate: func() ChatTemplate { return NewTinyLlamaChatTemplate() },
	},
	"mistral": {
		Name:      "Mistral",
		RoPEStyle: RoPEStyleSplitHalf,
		ChatTemplate: func() ChatTemplate { return NewMistralChatTemplate() },
	},
	"mixtral": {
		Name:      "Mixtral",
		RoPEStyle: RoPEStyleSplitHalf,
		ChatTemplate: func() ChatTemplate { return NewMistralChatTemplate() },
	},
	"lfm2": {
		Name:      "LFM2",
		RoPEStyle: RoPEStyleInterleaved,
		ChatTemplate: func() ChatTemplate { return NewChatMLTemplate() },
	},
	"lfm": {
		Name:      "LFM",
		RoPEStyle: RoPEStyleInterleaved,
		ChatTemplate: func() ChatTemplate { return NewChatMLTemplate() },
	},
}

// DetectModelFamily attempts to detect the model family from the model name.
func DetectModelFamily(modelName string) *ModelFamily {
	lower := strings.ToLower(modelName)

	// Check each known family
	for key, family := range KnownModelFamilies {
		if strings.Contains(lower, key) {
			return &family
		}
	}

	// Special case: llama-3 vs llama (default to llama2 for just "llama")
	if strings.Contains(lower, "llama-3") || strings.Contains(lower, "llama3") {
		family := KnownModelFamilies["llama3"]
		return &family
	}
	if strings.Contains(lower, "llama") {
		family := KnownModelFamilies["llama2"]
		return &family
	}

	return nil
}
