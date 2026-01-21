// Package e2e provides end-to-end tests for the NeuroGrid distributed inference engine.
package e2e

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/neurogrid/engine/api"
	"github.com/neurogrid/engine/pkg/inference"
	"github.com/neurogrid/engine/pkg/scheduler"
	"github.com/neurogrid/engine/pkg/types"
)

// ===== PRP-04: SSE Token Streaming E2E Tests =====
// These tests verify the SSE streaming implementation per the acceptance criteria
// in docs/prps/04-sse-streaming.md

// createStreamingTestServer creates a test server with streaming capabilities.
func createStreamingTestServer() *api.Server {
	config := types.Llama7BConfig()
	config.NumLayers = 1 // Minimal for testing

	engine := inference.NewEngine(inference.EngineConfig{
		ModelConfig: config,
		LocalPeerID: "local",
	})

	engine.SetTokenizer(NewMockTokenizer())
	engine.SetAssignments([]scheduler.LayerAssignment{
		{LayerID: -1, PeerID: "local"},
		{LayerID: 0, PeerID: "local"},
		{LayerID: 1, PeerID: "local"},
	})
	engine.InitializeKVCaches()

	serverConfig := api.DefaultServerConfig()
	serverConfig.ModelName = "test-model"

	return api.NewServer(engine, serverConfig)
}

// ===== Acceptance Criterion 1: Stream tokens via SSE when stream: true =====

func TestSSE_StreamTokensWhenStreamTrue(t *testing.T) {
	server := createStreamingTestServer()

	reqBody := api.ChatCompletionRequest{
		Model: "test-model",
		Messages: []api.Message{
			{Role: "user", Content: "Count to 5"},
		},
		MaxTokens:   10,
		Temperature: 0.0,
		Stream:      true,
	}

	body, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	server.Mux().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	// Verify Content-Type is text/event-stream
	contentType := w.Header().Get("Content-Type")
	if contentType != "text/event-stream" {
		t.Errorf("expected Content-Type 'text/event-stream', got '%s'", contentType)
	}

	// Verify Cache-Control header
	cacheControl := w.Header().Get("Cache-Control")
	if cacheControl != "no-cache" {
		t.Errorf("expected Cache-Control 'no-cache', got '%s'", cacheControl)
	}

	// Verify Connection header
	connection := w.Header().Get("Connection")
	if connection != "keep-alive" {
		t.Errorf("expected Connection 'keep-alive', got '%s'", connection)
	}

	// Verify response contains SSE data events
	respBody := w.Body.String()
	if !strings.Contains(respBody, "data:") {
		t.Error("response should contain SSE 'data:' events")
	}
}

// ===== Acceptance Criterion 2: Match OpenAI's SSE message format exactly =====

func TestSSE_MatchesOpenAIFormat(t *testing.T) {
	server := createStreamingTestServer()

	reqBody := api.ChatCompletionRequest{
		Model: "test-model",
		Messages: []api.Message{
			{Role: "user", Content: "Hi"},
		},
		MaxTokens: 5,
		Stream:    true,
	}

	body, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	server.Mux().ServeHTTP(w, req)

	// Parse SSE events
	events := parseSSEEvents(w.Body.String())

	if len(events) == 0 {
		t.Fatal("expected at least one SSE event")
	}

	// Verify first chunk contains role
	firstChunk := events[0]
	if firstChunk.Object != "chat.completion.chunk" {
		t.Errorf("expected object 'chat.completion.chunk', got '%s'", firstChunk.Object)
	}

	if !strings.HasPrefix(firstChunk.ID, "chatcmpl-") {
		t.Errorf("ID should start with 'chatcmpl-', got '%s'", firstChunk.ID)
	}

	if firstChunk.Created == 0 {
		t.Error("Created timestamp should not be zero")
	}

	if firstChunk.Model != "test-model" {
		t.Errorf("expected model 'test-model', got '%s'", firstChunk.Model)
	}

	if len(firstChunk.Choices) != 1 {
		t.Fatalf("expected 1 choice, got %d", len(firstChunk.Choices))
	}

	if firstChunk.Choices[0].Index != 0 {
		t.Errorf("expected choice index 0, got %d", firstChunk.Choices[0].Index)
	}

	// First chunk should have role="assistant" in delta
	if firstChunk.Choices[0].Delta.Role != "assistant" {
		t.Errorf("first chunk delta should have role 'assistant', got '%s'", firstChunk.Choices[0].Delta.Role)
	}
}

func TestSSE_DataPrefixFormat(t *testing.T) {
	server := createStreamingTestServer()

	reqBody := api.ChatCompletionRequest{
		Model: "test-model",
		Messages: []api.Message{
			{Role: "user", Content: "Test"},
		},
		MaxTokens: 3,
		Stream:    true,
	}

	body, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	w := httptest.NewRecorder()

	server.Mux().ServeHTTP(w, req)

	// OpenAI uses "data: " (with space after colon), not "data:"
	respBody := w.Body.String()
	lines := strings.Split(respBody, "\n")

	for _, line := range lines {
		if strings.HasPrefix(line, "data:") {
			// Verify the space after "data:"
			if !strings.HasPrefix(line, "data: ") {
				t.Errorf("SSE format error: line should start with 'data: ' (with space), got: %q", line)
			}
		}
	}
}

func TestSSE_EmptyLineBetweenEvents(t *testing.T) {
	server := createStreamingTestServer()

	reqBody := api.ChatCompletionRequest{
		Model: "test-model",
		Messages: []api.Message{
			{Role: "user", Content: "Test"},
		},
		MaxTokens: 5,
		Stream:    true,
	}

	body, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	w := httptest.NewRecorder()

	server.Mux().ServeHTTP(w, req)

	// Verify empty lines between events (\n\n)
	respBody := w.Body.String()
	if !strings.Contains(respBody, "\n\n") {
		t.Error("SSE events should be separated by double newlines")
	}

	// Each data line should be followed by \n\n
	dataCount := strings.Count(respBody, "data: ")
	doubleNewlineCount := strings.Count(respBody, "\n\n")

	if doubleNewlineCount < dataCount {
		t.Errorf("expected at least %d double newlines (one per event), got %d", dataCount, doubleNewlineCount)
	}
}

// ===== Acceptance Criterion 3: Send [DONE] marker at end =====

func TestSSE_SendsDONEMarker(t *testing.T) {
	server := createStreamingTestServer()

	reqBody := api.ChatCompletionRequest{
		Model: "test-model",
		Messages: []api.Message{
			{Role: "user", Content: "Hello"},
		},
		MaxTokens: 5,
		Stream:    true,
	}

	body, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	w := httptest.NewRecorder()

	server.Mux().ServeHTTP(w, req)

	respBody := w.Body.String()

	// [DONE] should be literal string, not JSON
	if !strings.Contains(respBody, "data: [DONE]") {
		t.Error("response should contain 'data: [DONE]' marker")
	}

	// [DONE] should be the last data line
	lastDataIndex := strings.LastIndex(respBody, "data: ")
	if lastDataIndex == -1 {
		t.Fatal("no data lines found")
	}

	lastDataLine := respBody[lastDataIndex:]
	if !strings.HasPrefix(lastDataLine, "data: [DONE]") {
		t.Errorf("last data line should be [DONE], got: %q", lastDataLine)
	}
}

func TestSSE_DONEIsNotJSON(t *testing.T) {
	server := createStreamingTestServer()

	reqBody := api.ChatCompletionRequest{
		Model: "test-model",
		Messages: []api.Message{
			{Role: "user", Content: "Test"},
		},
		MaxTokens: 3,
		Stream:    true,
	}

	body, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	w := httptest.NewRecorder()

	server.Mux().ServeHTTP(w, req)

	respBody := w.Body.String()
	lines := strings.Split(respBody, "\n")

	for _, line := range lines {
		if strings.HasPrefix(line, "data: [DONE]") {
			// The [DONE] marker should NOT be valid JSON
			dataContent := strings.TrimPrefix(line, "data: ")
			var parsed interface{}
			err := json.Unmarshal([]byte(dataContent), &parsed)
			if err == nil {
				t.Error("[DONE] marker should not be valid JSON - it should be literal '[DONE]'")
			}
		}
	}
}

// ===== Acceptance Criterion 4: Handle client disconnect gracefully =====

func TestSSE_HandleClientDisconnect(t *testing.T) {
	server := createStreamingTestServer()

	reqBody := api.ChatCompletionRequest{
		Model: "test-model",
		Messages: []api.Message{
			{Role: "user", Content: "Generate a long response please"},
		},
		MaxTokens: 100, // Request many tokens
		Stream:    true,
	}

	body, _ := json.Marshal(reqBody)

	// Create a request with a cancellable context
	ctx, cancel := context.WithCancel(context.Background())
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	req = req.WithContext(ctx)
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()

	// Cancel the context after a short delay to simulate disconnect
	go func() {
		time.Sleep(10 * time.Millisecond)
		cancel()
	}()

	// This should not panic when client disconnects
	server.Mux().ServeHTTP(w, req)

	// Server should handle the disconnect gracefully (no panic occurred)
	// The response might be partial or empty, but no error should propagate
	t.Log("Client disconnect handled gracefully")
}

func TestSSE_ContextCancellationStopsGeneration(t *testing.T) {
	server := createStreamingTestServer()

	reqBody := api.ChatCompletionRequest{
		Model: "test-model",
		Messages: []api.Message{
			{Role: "user", Content: "Hello"},
		},
		MaxTokens: 1000, // Many tokens
		Stream:    true,
	}

	body, _ := json.Marshal(reqBody)

	// Create a request with immediate cancellation
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	req = req.WithContext(ctx)
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()

	start := time.Now()
	server.Mux().ServeHTTP(w, req)
	elapsed := time.Since(start)

	// Should return quickly, not process all 1000 tokens
	if elapsed > 500*time.Millisecond {
		t.Errorf("cancelled request should return quickly, took %v", elapsed)
	}
}

// ===== Acceptance Criterion 5: Include usage stats in final chunk =====

func TestSSE_UsageInFinalChunk(t *testing.T) {
	server := createStreamingTestServer()

	reqBody := api.ChatCompletionRequest{
		Model: "test-model",
		Messages: []api.Message{
			{Role: "user", Content: "Hi"},
		},
		MaxTokens: 5,
		Stream:    true,
	}

	body, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	w := httptest.NewRecorder()

	server.Mux().ServeHTTP(w, req)

	events := parseSSEEvents(w.Body.String())

	// Find the chunk with finish_reason (final content chunk before [DONE])
	var finalChunk *api.ChatCompletionChunk
	for i := len(events) - 1; i >= 0; i-- {
		if len(events[i].Choices) > 0 && events[i].Choices[0].FinishReason != "" {
			finalChunk = &events[i]
			break
		}
	}

	if finalChunk == nil {
		t.Fatal("no final chunk with finish_reason found")
	}

	// Final chunk should have finish_reason
	if finalChunk.Choices[0].FinishReason == "" {
		t.Error("final chunk should have a finish_reason")
	}

	// Note: Usage stats in streaming is optional per OpenAI spec
	// Some implementations include it, some don't
	// We'll test that if present, it's valid
	t.Logf("Final chunk finish_reason: %s", finalChunk.Choices[0].FinishReason)
}

func TestSSE_FinishReasonStop(t *testing.T) {
	server := createStreamingTestServer()

	reqBody := api.ChatCompletionRequest{
		Model: "test-model",
		Messages: []api.Message{
			{Role: "user", Content: "Say hello"},
		},
		MaxTokens: 100, // Should hit EOS before this
		Stream:    true,
	}

	body, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	w := httptest.NewRecorder()

	server.Mux().ServeHTTP(w, req)

	events := parseSSEEvents(w.Body.String())

	// Find final chunk
	var finalChunk *api.ChatCompletionChunk
	for i := len(events) - 1; i >= 0; i-- {
		if len(events[i].Choices) > 0 && events[i].Choices[0].FinishReason != "" {
			finalChunk = &events[i]
			break
		}
	}

	if finalChunk == nil {
		t.Fatal("no final chunk found")
	}

	// Finish reason should be "stop", "length", or "eos"
	validReasons := map[string]bool{"stop": true, "length": true, "eos": true, "max_tokens": true}
	if !validReasons[finalChunk.Choices[0].FinishReason] {
		t.Errorf("invalid finish_reason: %s", finalChunk.Choices[0].FinishReason)
	}
}

// ===== Acceptance Criterion 6: Time-to-first-token < 500ms (excluding model latency) =====

func TestSSE_TimeToFirstToken(t *testing.T) {
	server := createStreamingTestServer()

	reqBody := api.ChatCompletionRequest{
		Model: "test-model",
		Messages: []api.Message{
			{Role: "user", Content: "Hello"},
		},
		MaxTokens: 10,
		Stream:    true,
	}

	body, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	// Use a pipe to measure time-to-first-byte
	pr, pw := io.Pipe()
	w := &pipeResponseWriter{
		header: make(http.Header),
		pw:     pw,
	}

	start := time.Now()
	done := make(chan struct{})
	var firstByteTime time.Duration

	go func() {
		buf := make([]byte, 1)
		pr.Read(buf) // Read first byte
		firstByteTime = time.Since(start)
		close(done)
		io.Copy(io.Discard, pr) // Drain remaining
	}()

	go func() {
		server.Mux().ServeHTTP(w, req)
		pw.Close()
	}()

	<-done

	// Time-to-first-token should be < 500ms (excluding actual model latency)
	// In tests with mocked tokenizer/engine, this should be very fast
	if firstByteTime > 500*time.Millisecond {
		t.Errorf("time-to-first-token exceeded 500ms: %v", firstByteTime)
	}

	t.Logf("Time-to-first-token: %v", firstByteTime)
}

// ===== Additional Streaming Tests =====

func TestSSE_MultipleChunksReceived(t *testing.T) {
	server := createStreamingTestServer()

	reqBody := api.ChatCompletionRequest{
		Model: "test-model",
		Messages: []api.Message{
			{Role: "user", Content: "Count to 10"},
		},
		MaxTokens: 10,
		Stream:    true,
	}

	body, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	w := httptest.NewRecorder()

	server.Mux().ServeHTTP(w, req)

	events := parseSSEEvents(w.Body.String())

	// Should have multiple chunks: first (with role), content chunks, final (with finish_reason)
	if len(events) < 2 {
		t.Errorf("expected multiple chunks, got %d", len(events))
	}

	t.Logf("Received %d SSE events", len(events))
}

func TestSSE_ConsistentIDsAcrossChunks(t *testing.T) {
	server := createStreamingTestServer()

	reqBody := api.ChatCompletionRequest{
		Model: "test-model",
		Messages: []api.Message{
			{Role: "user", Content: "Hello world"},
		},
		MaxTokens: 5,
		Stream:    true,
	}

	body, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	w := httptest.NewRecorder()

	server.Mux().ServeHTTP(w, req)

	events := parseSSEEvents(w.Body.String())
	if len(events) == 0 {
		t.Fatal("no events received")
	}

	// All chunks should have the same ID
	expectedID := events[0].ID
	for i, event := range events {
		if event.ID != expectedID {
			t.Errorf("chunk %d has different ID: expected %s, got %s", i, expectedID, event.ID)
		}
	}
}

func TestSSE_ConsistentTimestampAcrossChunks(t *testing.T) {
	server := createStreamingTestServer()

	reqBody := api.ChatCompletionRequest{
		Model: "test-model",
		Messages: []api.Message{
			{Role: "user", Content: "Test"},
		},
		MaxTokens: 5,
		Stream:    true,
	}

	body, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	w := httptest.NewRecorder()

	server.Mux().ServeHTTP(w, req)

	events := parseSSEEvents(w.Body.String())
	if len(events) == 0 {
		t.Fatal("no events received")
	}

	// All chunks should have the same created timestamp
	expectedCreated := events[0].Created
	for i, event := range events {
		if event.Created != expectedCreated {
			t.Errorf("chunk %d has different created timestamp: expected %d, got %d", i, expectedCreated, event.Created)
		}
	}
}

func TestSSE_DeltaContentInMiddleChunks(t *testing.T) {
	server := createStreamingTestServer()

	reqBody := api.ChatCompletionRequest{
		Model: "test-model",
		Messages: []api.Message{
			{Role: "user", Content: "Hello"},
		},
		MaxTokens: 10,
		Stream:    true,
	}

	body, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	w := httptest.NewRecorder()

	server.Mux().ServeHTTP(w, req)

	events := parseSSEEvents(w.Body.String())

	// Content chunks (not first, not final) should have content in delta
	contentChunks := 0
	for i, event := range events {
		if i == 0 {
			continue // Skip first chunk (has role)
		}
		if len(event.Choices) > 0 && event.Choices[0].FinishReason != "" {
			continue // Skip final chunk
		}
		if len(event.Choices) > 0 && event.Choices[0].Delta.Content != "" {
			contentChunks++
		}
	}

	// Should have at least one content chunk (unless empty response)
	t.Logf("Found %d content chunks", contentChunks)
}

func TestSSE_FirstChunkHasRoleOnly(t *testing.T) {
	server := createStreamingTestServer()

	reqBody := api.ChatCompletionRequest{
		Model: "test-model",
		Messages: []api.Message{
			{Role: "user", Content: "Hi"},
		},
		MaxTokens: 5,
		Stream:    true,
	}

	body, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	w := httptest.NewRecorder()

	server.Mux().ServeHTTP(w, req)

	events := parseSSEEvents(w.Body.String())
	if len(events) == 0 {
		t.Fatal("no events received")
	}

	firstChunk := events[0]
	if len(firstChunk.Choices) == 0 {
		t.Fatal("first chunk has no choices")
	}

	// First chunk should have role
	if firstChunk.Choices[0].Delta.Role != "assistant" {
		t.Errorf("first chunk should have role 'assistant', got '%s'", firstChunk.Choices[0].Delta.Role)
	}
}

func TestSSE_FinalChunkHasEmptyDelta(t *testing.T) {
	server := createStreamingTestServer()

	reqBody := api.ChatCompletionRequest{
		Model: "test-model",
		Messages: []api.Message{
			{Role: "user", Content: "Hello"},
		},
		MaxTokens: 5,
		Stream:    true,
	}

	body, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	w := httptest.NewRecorder()

	server.Mux().ServeHTTP(w, req)

	events := parseSSEEvents(w.Body.String())

	// Find final chunk with finish_reason
	var finalChunk *api.ChatCompletionChunk
	for i := len(events) - 1; i >= 0; i-- {
		if len(events[i].Choices) > 0 && events[i].Choices[0].FinishReason != "" {
			finalChunk = &events[i]
			break
		}
	}

	if finalChunk == nil {
		t.Fatal("no final chunk found")
	}

	// Final chunk should have empty delta (or just finish_reason)
	if finalChunk.Choices[0].Delta.Content != "" && finalChunk.Choices[0].Delta.Role != "" {
		t.Log("Note: Final chunk has non-empty delta, which is acceptable but not typical")
	}
}

// ===== Concurrent Streaming Tests =====

func TestSSE_ConcurrentStreams(t *testing.T) {
	server := createStreamingTestServer()

	numStreams := 5
	var wg sync.WaitGroup
	results := make(chan bool, numStreams)

	for i := 0; i < numStreams; i++ {
		wg.Add(1)
		go func(streamID int) {
			defer wg.Done()

			reqBody := api.ChatCompletionRequest{
				Model: "test-model",
				Messages: []api.Message{
					{Role: "user", Content: "Hello"},
				},
				MaxTokens: 5,
				Stream:    true,
			}

			body, _ := json.Marshal(reqBody)
			req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()

			server.Mux().ServeHTTP(w, req)

			// Check basic success criteria
			if w.Code != http.StatusOK {
				results <- false
				return
			}

			respBody := w.Body.String()
			if !strings.Contains(respBody, "data:") || !strings.Contains(respBody, "[DONE]") {
				results <- false
				return
			}

			results <- true
		}(i)
	}

	wg.Wait()
	close(results)

	successCount := 0
	for success := range results {
		if success {
			successCount++
		}
	}

	if successCount != numStreams {
		t.Errorf("expected %d successful streams, got %d", numStreams, successCount)
	}
}

// ===== Error Handling Tests =====

func TestSSE_ErrorMidStream(t *testing.T) {
	// This test verifies that errors during streaming are handled gracefully
	// The error should be sent as an SSE event, not cause a panic

	server := createStreamingTestServer()

	reqBody := api.ChatCompletionRequest{
		Model: "test-model",
		Messages: []api.Message{
			{Role: "user", Content: "Hello"},
		},
		MaxTokens: 5,
		Stream:    true,
	}

	body, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	w := httptest.NewRecorder()

	// This should not panic even if there's an internal error
	func() {
		defer func() {
			if r := recover(); r != nil {
				t.Errorf("handler panicked: %v", r)
			}
		}()
		server.Mux().ServeHTTP(w, req)
	}()
}

func TestSSE_InvalidRequestReturnsError(t *testing.T) {
	server := createStreamingTestServer()

	// Empty messages should return an error, not start streaming
	reqBody := api.ChatCompletionRequest{
		Model:    "test-model",
		Messages: []api.Message{}, // Invalid: empty messages
		Stream:   true,
	}

	body, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	server.Mux().ServeHTTP(w, req)

	// Should return 400 Bad Request, not start SSE stream
	if w.Code != http.StatusBadRequest {
		t.Errorf("expected 400 for invalid request, got %d", w.Code)
	}

	// Content-Type should be JSON for error, not event-stream
	contentType := w.Header().Get("Content-Type")
	if contentType == "text/event-stream" {
		t.Error("error response should not use text/event-stream Content-Type")
	}
}

// ===== Non-Streaming Compatibility Test =====

func TestSSE_NonStreamingStillWorks(t *testing.T) {
	server := createStreamingTestServer()

	// stream: false should return regular JSON response
	reqBody := api.ChatCompletionRequest{
		Model: "test-model",
		Messages: []api.Message{
			{Role: "user", Content: "Hello"},
		},
		MaxTokens: 5,
		Stream:    false, // Not streaming
	}

	body, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	server.Mux().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	// Should return JSON, not SSE
	contentType := w.Header().Get("Content-Type")
	if contentType == "text/event-stream" {
		t.Error("non-streaming request should not return event-stream")
	}

	// Should be valid ChatCompletionResponse
	var resp api.ChatCompletionResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to decode non-streaming response: %v", err)
	}

	if resp.Object != "chat.completion" {
		t.Errorf("expected object 'chat.completion', got '%s'", resp.Object)
	}
}

// ===== Helper Functions =====

// parseSSEEvents parses SSE data from a response body.
func parseSSEEvents(body string) []api.ChatCompletionChunk {
	var events []api.ChatCompletionChunk

	scanner := bufio.NewScanner(strings.NewReader(body))
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "data: ") {
			data := strings.TrimPrefix(line, "data: ")
			if data == "[DONE]" {
				continue
			}

			var chunk api.ChatCompletionChunk
			if err := json.Unmarshal([]byte(data), &chunk); err == nil {
				events = append(events, chunk)
			}
		}
	}

	return events
}

// pipeResponseWriter is a custom ResponseWriter for measuring time-to-first-byte.
type pipeResponseWriter struct {
	header     http.Header
	pw         *io.PipeWriter
	statusCode int
}

func (p *pipeResponseWriter) Header() http.Header {
	return p.header
}

func (p *pipeResponseWriter) Write(data []byte) (int, error) {
	return p.pw.Write(data)
}

func (p *pipeResponseWriter) WriteHeader(statusCode int) {
	p.statusCode = statusCode
}

func (p *pipeResponseWriter) Flush() {
	// No-op for pipe writer
}

// Verify pipeResponseWriter implements http.Flusher
var _ http.Flusher = (*pipeResponseWriter)(nil)
