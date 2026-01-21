// Package api provides HTTP API handlers for the NeuroGrid inference engine.
package api

import (
	"crypto/rand"
	"encoding/hex"
	"time"
)

// Message represents a chat message.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ChatCompletionRequest represents an OpenAI-compatible chat completion request.
type ChatCompletionRequest struct {
	Model            string    `json:"model"`
	Messages         []Message `json:"messages"`
	MaxTokens        int       `json:"max_tokens,omitempty"`
	Temperature      float32   `json:"temperature,omitempty"`
	TopP             float32   `json:"top_p,omitempty"`
	Stream           bool      `json:"stream,omitempty"`
	Stop             []string  `json:"stop,omitempty"`
	PresencePenalty  float32   `json:"presence_penalty,omitempty"`
	FrequencyPenalty float32   `json:"frequency_penalty,omitempty"`
	User             string    `json:"user,omitempty"`
}

// SetDefaults applies default values to the request.
func (r *ChatCompletionRequest) SetDefaults() {
	if r.MaxTokens == 0 {
		r.MaxTokens = 256
	}
	if r.Temperature == 0 {
		r.Temperature = 0.7
	}
	if r.TopP == 0 {
		r.TopP = 1.0
	}
}

// ChatCompletionResponse represents an OpenAI-compatible chat completion response.
type ChatCompletionResponse struct {
	ID                string   `json:"id"`
	Object            string   `json:"object"`
	Created           int64    `json:"created"`
	Model             string   `json:"model"`
	Choices           []Choice `json:"choices"`
	Usage             *Usage   `json:"usage,omitempty"`
	SystemFingerprint string   `json:"system_fingerprint,omitempty"`
}

// Choice represents a completion choice.
type Choice struct {
	Index        int     `json:"index"`
	Message      Message `json:"message"`
	FinishReason string  `json:"finish_reason"`
}

// Usage represents token usage statistics.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// ChatCompletionChunk represents a streaming response chunk.
type ChatCompletionChunk struct {
	ID                string        `json:"id"`
	Object            string        `json:"object"`
	Created           int64         `json:"created"`
	Model             string        `json:"model"`
	Choices           []ChunkChoice `json:"choices"`
	SystemFingerprint string        `json:"system_fingerprint,omitempty"`
}

// ChunkChoice represents a streaming choice.
type ChunkChoice struct {
	Index        int    `json:"index"`
	Delta        Delta  `json:"delta"`
	FinishReason string `json:"finish_reason,omitempty"`
}

// Delta represents the incremental content in a streaming response.
type Delta struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

// ModelInfo represents information about a model.
type ModelInfo struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

// ModelsResponse represents the response from /v1/models.
type ModelsResponse struct {
	Object string      `json:"object"`
	Data   []ModelInfo `json:"data"`
}

// ErrorResponse represents an API error response.
type ErrorResponse struct {
	Error ErrorDetail `json:"error"`
}

// ErrorDetail contains error information.
type ErrorDetail struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Param   string `json:"param,omitempty"`
	Code    string `json:"code,omitempty"`
}

// HealthResponse represents the health check response.
type HealthResponse struct {
	Status    string `json:"status"`
	Model     string `json:"model,omitempty"`
	Timestamp int64  `json:"timestamp"`
	Version   string `json:"version"`
}

// generateID creates a unique completion ID.
func generateID() string {
	bytes := make([]byte, 16)
	rand.Read(bytes)
	return "chatcmpl-" + hex.EncodeToString(bytes)
}

// NewChatCompletionResponse creates a new chat completion response.
func NewChatCompletionResponse(model string, content string, finishReason string) *ChatCompletionResponse {
	return &ChatCompletionResponse{
		ID:      generateID(),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   model,
		Choices: []Choice{
			{
				Index: 0,
				Message: Message{
					Role:    "assistant",
					Content: content,
				},
				FinishReason: finishReason,
			},
		},
	}
}

// NewChatCompletionChunk creates a new streaming chunk.
func NewChatCompletionChunk(model string, content string, finishReason string) *ChatCompletionChunk {
	chunk := &ChatCompletionChunk{
		ID:      generateID(),
		Object:  "chat.completion.chunk",
		Created: time.Now().Unix(),
		Model:   model,
		Choices: []ChunkChoice{
			{
				Index: 0,
				Delta: Delta{
					Content: content,
				},
			},
		},
	}

	if finishReason != "" {
		chunk.Choices[0].FinishReason = finishReason
	}

	return chunk
}

// NewErrorResponse creates an error response.
func NewErrorResponse(message, errorType, code string) *ErrorResponse {
	return &ErrorResponse{
		Error: ErrorDetail{
			Message: message,
			Type:    errorType,
			Code:    code,
		},
	}
}
