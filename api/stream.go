// Package api provides HTTP API handlers for the NeuroGrid inference engine.
package api

import (
	"encoding/json"
	"fmt"
	"net/http"
)

// SSEWriter wraps http.ResponseWriter for Server-Sent Events streaming.
// It handles the SSE protocol details including proper formatting and flushing.
type SSEWriter struct {
	w       http.ResponseWriter
	flusher http.Flusher
}

// NewSSEWriter creates a new SSEWriter that wraps the given ResponseWriter.
// It sets the required SSE headers and returns an error if streaming is not supported.
func NewSSEWriter(w http.ResponseWriter) (*SSEWriter, error) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		return nil, fmt.Errorf("streaming not supported: ResponseWriter does not implement http.Flusher")
	}

	// CRITICAL: Set headers before first write
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no") // Disable nginx buffering
	w.Header().Set("Transfer-Encoding", "chunked")

	// Send headers immediately
	w.WriteHeader(http.StatusOK)
	flusher.Flush()

	return &SSEWriter{w: w, flusher: flusher}, nil
}

// WriteEvent writes a JSON-encoded SSE event.
// Format: "data: {json}\n\n" following OpenAI's SSE format.
func (s *SSEWriter) WriteEvent(data interface{}) error {
	jsonData, err := json.Marshal(data)
	if err != nil {
		return fmt.Errorf("failed to marshal event data: %w", err)
	}

	// PATTERN: OpenAI format uses "data: " with space after colon
	if _, err := fmt.Fprintf(s.w, "data: %s\n\n", jsonData); err != nil {
		return fmt.Errorf("failed to write event: %w", err)
	}

	s.flusher.Flush()
	return nil
}

// WriteError writes an error event in SSE format.
func (s *SSEWriter) WriteError(message string, errorType string) error {
	errEvent := map[string]interface{}{
		"error": map[string]string{
			"message": message,
			"type":    errorType,
		},
	}
	return s.WriteEvent(errEvent)
}

// Close writes the [DONE] marker and flushes the stream.
// PATTERN: OpenAI sends literal "[DONE]" not JSON.
func (s *SSEWriter) Close() error {
	if _, err := fmt.Fprintf(s.w, "data: [DONE]\n\n"); err != nil {
		return fmt.Errorf("failed to write DONE marker: %w", err)
	}
	s.flusher.Flush()
	return nil
}

// Flush forces the stream to be flushed to the client.
func (s *SSEWriter) Flush() {
	s.flusher.Flush()
}

// StreamState holds the state for a streaming response.
// It ensures consistent ID and timestamp across all chunks.
type StreamState struct {
	ID      string
	Created int64
	Model   string
}

// NewStreamState creates a new stream state with a unique ID.
func NewStreamState(model string, created int64) *StreamState {
	return &StreamState{
		ID:      generateID(),
		Created: created,
		Model:   model,
	}
}

// CreateRoleChunk creates the initial chunk with the assistant role.
func (s *StreamState) CreateRoleChunk() *ChatCompletionChunk {
	return &ChatCompletionChunk{
		ID:      s.ID,
		Object:  "chat.completion.chunk",
		Created: s.Created,
		Model:   s.Model,
		Choices: []ChunkChoice{
			{
				Index: 0,
				Delta: Delta{Role: "assistant"},
			},
		},
	}
}

// CreateContentChunk creates a chunk with content.
func (s *StreamState) CreateContentChunk(content string) *ChatCompletionChunk {
	return &ChatCompletionChunk{
		ID:      s.ID,
		Object:  "chat.completion.chunk",
		Created: s.Created,
		Model:   s.Model,
		Choices: []ChunkChoice{
			{
				Index: 0,
				Delta: Delta{Content: content},
			},
		},
	}
}

// CreateFinalChunk creates the final chunk with finish reason.
func (s *StreamState) CreateFinalChunk(finishReason string) *ChatCompletionChunk {
	return &ChatCompletionChunk{
		ID:      s.ID,
		Object:  "chat.completion.chunk",
		Created: s.Created,
		Model:   s.Model,
		Choices: []ChunkChoice{
			{
				Index:        0,
				Delta:        Delta{},
				FinishReason: finishReason,
			},
		},
	}
}
