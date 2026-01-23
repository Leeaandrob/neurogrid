// Package model provides model loading and weight management for LLM inference.
package model

import "strings"

// Message represents a chat message for chat template formatting.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ChatTemplate defines the interface for chat template formatters.
// Implementations format chat messages according to model-specific templates.
type ChatTemplate interface {
	// Format converts a slice of messages into a formatted prompt string.
	Format(messages []Message) string
}

// Llama2ChatTemplate formats messages according to Llama 2 chat format.
// The format uses [INST] and [/INST] tags with <<SYS>> for system messages.
type Llama2ChatTemplate struct{}

// NewLlama2ChatTemplate creates a new Llama 2 chat template.
func NewLlama2ChatTemplate() *Llama2ChatTemplate {
	return &Llama2ChatTemplate{}
}

// Format formats messages according to Llama 2 chat format.
//
// Format examples:
//
// Single turn with system:
//
//	[INST] <<SYS>>
//	{system}
//	<</SYS>>
//
//	{user} [/INST] {assistant} </s>
//
// Multi-turn:
//
//	[INST] {user1} [/INST] {assistant1} </s><s>[INST] {user2} [/INST]
func (t *Llama2ChatTemplate) Format(messages []Message) string {
	var result strings.Builder

	hasSystem := false
	for i, msg := range messages {
		switch msg.Role {
		case "system":
			result.WriteString("[INST] <<SYS>>\n")
			result.WriteString(msg.Content)
			result.WriteString("\n<</SYS>>\n\n")
			hasSystem = true
		case "user":
			if i > 0 && !hasSystem {
				// Multi-turn: add </s><s>[INST] before user message
				if messages[i-1].Role == "assistant" {
					result.WriteString("<s>")
				}
				result.WriteString("[INST] ")
			} else if !hasSystem {
				result.WriteString("[INST] ")
			}
			result.WriteString(msg.Content)
			result.WriteString(" [/INST] ")
			hasSystem = false
		case "assistant":
			result.WriteString(msg.Content)
			result.WriteString(" </s>")
		}
	}

	return result.String()
}

// Llama3ChatTemplate formats messages according to Llama 3 chat format.
// The format uses special header tokens like <|start_header_id|>.
type Llama3ChatTemplate struct{}

// NewLlama3ChatTemplate creates a new Llama 3 chat template.
func NewLlama3ChatTemplate() *Llama3ChatTemplate {
	return &Llama3ChatTemplate{}
}

// Format formats messages according to Llama 3 chat format.
//
// Format:
//
//	<|begin_of_text|><|start_header_id|>system<|end_header_id|>
//
//	{system}<|eot_id|><|start_header_id|>user<|end_header_id|>
//
//	{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
//
//	{assistant}
func (t *Llama3ChatTemplate) Format(messages []Message) string {
	var result strings.Builder
	result.WriteString("<|begin_of_text|>")

	for _, msg := range messages {
		result.WriteString("<|start_header_id|>")
		result.WriteString(msg.Role)
		result.WriteString("<|end_header_id|>\n\n")
		result.WriteString(msg.Content)
		result.WriteString("<|eot_id|>")
	}

	// Add assistant header for generation
	result.WriteString("<|start_header_id|>assistant<|end_header_id|>\n\n")

	return result.String()
}

// TinyLlamaChatTemplate formats messages according to TinyLlama chat format (ChatML style).
// TinyLlama uses <|system|>, <|user|>, and <|assistant|> tags.
type TinyLlamaChatTemplate struct{}

// NewTinyLlamaChatTemplate creates a new TinyLlama chat template.
func NewTinyLlamaChatTemplate() *TinyLlamaChatTemplate {
	return &TinyLlamaChatTemplate{}
}

// Format formats messages according to TinyLlama ChatML format.
//
// Format:
//
//	<|system|>
//	{system}</s>
//	<|user|>
//	{user}</s>
//	<|assistant|>
//	{assistant}</s>
func (t *TinyLlamaChatTemplate) Format(messages []Message) string {
	var result strings.Builder

	for _, msg := range messages {
		switch msg.Role {
		case "system":
			result.WriteString("<|system|>\n")
			result.WriteString(msg.Content)
			result.WriteString("</s>\n")
		case "user":
			result.WriteString("<|user|>\n")
			result.WriteString(msg.Content)
			result.WriteString("</s>\n")
		case "assistant":
			result.WriteString("<|assistant|>\n")
			result.WriteString(msg.Content)
			result.WriteString("</s>\n")
		}
	}

	// Add assistant header for generation
	result.WriteString("<|assistant|>\n")

	return result.String()
}

// ChatTemplateFactory returns the appropriate chat template for a model.
// It selects the template based on the model name or config.
// For more advanced detection, use LoadModelConfig() and DetectChatTemplate().
func ChatTemplateFactory(modelName string) ChatTemplate {
	lower := strings.ToLower(modelName)

	// Check for TinyLlama models (ChatML style)
	if strings.Contains(lower, "tinyllama") {
		return NewTinyLlamaChatTemplate()
	}

	// Check for Llama 3 models (new header style)
	if strings.Contains(lower, "llama-3") || strings.Contains(lower, "llama3") {
		return NewLlama3ChatTemplate()
	}

	// Check for Mistral/Mixtral models
	if strings.Contains(lower, "mistral") || strings.Contains(lower, "mixtral") {
		return NewMistralChatTemplate()
	}

	// Default to Llama 2 format
	return NewLlama2ChatTemplate()
}
