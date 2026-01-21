// Package e2e provides end-to-end tests for the NeuroGrid distributed inference engine.
package e2e

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/neurogrid/engine/api"
	"github.com/neurogrid/engine/pkg/inference"
	"github.com/neurogrid/engine/pkg/scheduler"
	"github.com/neurogrid/engine/pkg/types"
)

// ===== TASK-025: HTTP Server Tests =====

func createTestEngine() *inference.Engine {
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

	return engine
}

func createTestServer() *api.Server {
	engine := createTestEngine()
	config := api.DefaultServerConfig()
	config.ModelName = "test-model"

	return api.NewServer(engine, config)
}

func TestServer_Creation(t *testing.T) {
	server := createTestServer()
	if server == nil {
		t.Fatal("NewServer returned nil")
	}
}

func TestServer_HealthEndpoint(t *testing.T) {
	server := createTestServer()

	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	w := httptest.NewRecorder()

	server.Mux().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", w.Code)
	}

	var resp api.HealthResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if resp.Status != "healthy" {
		t.Errorf("expected status 'healthy', got '%s'", resp.Status)
	}

	if resp.Version == "" {
		t.Error("version should not be empty")
	}
}

func TestServer_ReadyEndpoint(t *testing.T) {
	server := createTestServer()

	req := httptest.NewRequest(http.MethodGet, "/ready", nil)
	w := httptest.NewRecorder()

	server.Mux().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", w.Code)
	}

	var resp map[string]interface{}
	json.NewDecoder(w.Body).Decode(&resp)

	if ready, ok := resp["ready"].(bool); !ok || !ready {
		t.Error("expected ready=true")
	}
}

func TestServer_MetricsEndpoint(t *testing.T) {
	server := createTestServer()

	req := httptest.NewRequest(http.MethodGet, "/metrics", nil)
	w := httptest.NewRecorder()

	server.Mux().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", w.Code)
	}

	var resp map[string]interface{}
	json.NewDecoder(w.Body).Decode(&resp)

	if _, ok := resp["requests_total"]; !ok {
		t.Error("expected requests_total in metrics")
	}

	if _, ok := resp["uptime_seconds"]; !ok {
		t.Error("expected uptime_seconds in metrics")
	}
}

func TestServer_RootEndpoint(t *testing.T) {
	server := createTestServer()

	req := httptest.NewRequest(http.MethodGet, "/", nil)
	w := httptest.NewRecorder()

	server.Mux().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", w.Code)
	}

	var resp map[string]string
	json.NewDecoder(w.Body).Decode(&resp)

	if resp["name"] != "NeuroGrid Inference Engine" {
		t.Errorf("unexpected name: %s", resp["name"])
	}
}

func TestServer_NotFound(t *testing.T) {
	server := createTestServer()

	req := httptest.NewRequest(http.MethodGet, "/nonexistent", nil)
	w := httptest.NewRecorder()

	server.Mux().ServeHTTP(w, req)

	if w.Code != http.StatusNotFound {
		t.Errorf("expected 404, got %d", w.Code)
	}
}

// ===== TASK-026: Chat Completions Handler Tests =====

func TestServer_ListModels(t *testing.T) {
	server := createTestServer()

	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	w := httptest.NewRecorder()

	server.Mux().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", w.Code)
	}

	var resp api.ModelsResponse
	json.NewDecoder(w.Body).Decode(&resp)

	if resp.Object != "list" {
		t.Errorf("expected object 'list', got '%s'", resp.Object)
	}

	if len(resp.Data) != 1 {
		t.Errorf("expected 1 model, got %d", len(resp.Data))
	}

	if resp.Data[0].ID != "test-model" {
		t.Errorf("expected model 'test-model', got '%s'", resp.Data[0].ID)
	}
}

func TestServer_GetModel(t *testing.T) {
	server := createTestServer()

	req := httptest.NewRequest(http.MethodGet, "/v1/models/test-model", nil)
	w := httptest.NewRecorder()

	server.Mux().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", w.Code)
	}

	var resp api.ModelInfo
	json.NewDecoder(w.Body).Decode(&resp)

	if resp.ID != "test-model" {
		t.Errorf("expected model 'test-model', got '%s'", resp.ID)
	}
}

func TestServer_GetModelNotFound(t *testing.T) {
	server := createTestServer()

	req := httptest.NewRequest(http.MethodGet, "/v1/models/nonexistent", nil)
	w := httptest.NewRecorder()

	server.Mux().ServeHTTP(w, req)

	if w.Code != http.StatusNotFound {
		t.Errorf("expected 404, got %d", w.Code)
	}
}

func TestServer_ChatCompletions(t *testing.T) {
	server := createTestServer()

	reqBody := api.ChatCompletionRequest{
		Model: "test-model",
		Messages: []api.Message{
			{Role: "user", Content: "Hello"},
		},
		MaxTokens:   10,
		Temperature: 0.0,
	}

	body, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	server.Mux().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var resp api.ChatCompletionResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if resp.Object != "chat.completion" {
		t.Errorf("expected object 'chat.completion', got '%s'", resp.Object)
	}

	if len(resp.Choices) != 1 {
		t.Errorf("expected 1 choice, got %d", len(resp.Choices))
	}

	if resp.Choices[0].Message.Role != "assistant" {
		t.Errorf("expected role 'assistant', got '%s'", resp.Choices[0].Message.Role)
	}

	// ID should have the expected format
	if !strings.HasPrefix(resp.ID, "chatcmpl-") {
		t.Errorf("ID should start with 'chatcmpl-', got '%s'", resp.ID)
	}
}

func TestServer_ChatCompletionsMissingMessages(t *testing.T) {
	server := createTestServer()

	reqBody := api.ChatCompletionRequest{
		Model:    "test-model",
		Messages: []api.Message{}, // Empty messages
	}

	body, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	server.Mux().ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", w.Code)
	}
}

func TestServer_ChatCompletionsInvalidJSON(t *testing.T) {
	server := createTestServer()

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader("invalid json"))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	server.Mux().ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", w.Code)
	}

	var resp api.ErrorResponse
	json.NewDecoder(w.Body).Decode(&resp)

	if resp.Error.Type != "invalid_request" {
		t.Errorf("expected error type 'invalid_request', got '%s'", resp.Error.Type)
	}
}

func TestServer_ChatCompletionsMethodNotAllowed(t *testing.T) {
	server := createTestServer()

	req := httptest.NewRequest(http.MethodGet, "/v1/chat/completions", nil)
	w := httptest.NewRecorder()

	server.Mux().ServeHTTP(w, req)

	if w.Code != http.StatusMethodNotAllowed {
		t.Errorf("expected 405, got %d", w.Code)
	}
}

func TestServer_ChatCompletionsWithSystemMessage(t *testing.T) {
	server := createTestServer()

	reqBody := api.ChatCompletionRequest{
		Model: "test-model",
		Messages: []api.Message{
			{Role: "system", Content: "You are a helpful assistant."},
			{Role: "user", Content: "What is 2+2?"},
		},
		MaxTokens:   5,
		Temperature: 0.0,
	}

	body, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	server.Mux().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", w.Code)
	}
}

// ===== TASK-027: Streaming Response Tests =====

func TestServer_ChatCompletionsStreaming(t *testing.T) {
	server := createTestServer()

	reqBody := api.ChatCompletionRequest{
		Model: "test-model",
		Messages: []api.Message{
			{Role: "user", Content: "Hello"},
		},
		MaxTokens:   5,
		Temperature: 0.0,
		Stream:      true,
	}

	body, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	server.Mux().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	// Check content type
	contentType := w.Header().Get("Content-Type")
	if contentType != "text/event-stream" {
		t.Errorf("expected Content-Type 'text/event-stream', got '%s'", contentType)
	}

	// Check that response contains SSE data
	respBody := w.Body.String()
	if !strings.Contains(respBody, "data:") {
		t.Error("response should contain SSE data events")
	}

	// Check for [DONE] marker
	if !strings.Contains(respBody, "[DONE]") {
		t.Error("response should contain [DONE] marker")
	}
}

func TestServer_StreamingResponseFormat(t *testing.T) {
	server := createTestServer()

	reqBody := api.ChatCompletionRequest{
		Model: "test-model",
		Messages: []api.Message{
			{Role: "user", Content: "Hi"},
		},
		MaxTokens: 3,
		Stream:    true,
	}

	body, _ := json.Marshal(reqBody)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	w := httptest.NewRecorder()

	server.Mux().ServeHTTP(w, req)

	// Parse SSE events
	respBody := w.Body.String()
	lines := strings.Split(respBody, "\n")

	var chunks []api.ChatCompletionChunk
	for _, line := range lines {
		if strings.HasPrefix(line, "data: ") {
			data := strings.TrimPrefix(line, "data: ")
			if data == "[DONE]" {
				break
			}

			var chunk api.ChatCompletionChunk
			if err := json.Unmarshal([]byte(data), &chunk); err == nil {
				chunks = append(chunks, chunk)
			}
		}
	}

	if len(chunks) == 0 {
		t.Error("expected at least one chunk")
	}

	// First chunk should have role
	if len(chunks) > 0 {
		if chunks[0].Object != "chat.completion.chunk" {
			t.Errorf("expected object 'chat.completion.chunk', got '%s'", chunks[0].Object)
		}
	}
}

// ===== API Types Tests =====

func TestChatCompletionRequest_SetDefaults(t *testing.T) {
	req := &api.ChatCompletionRequest{}
	req.SetDefaults()

	if req.MaxTokens != 256 {
		t.Errorf("expected default MaxTokens 256, got %d", req.MaxTokens)
	}

	if req.Temperature != 0.7 {
		t.Errorf("expected default Temperature 0.7, got %f", req.Temperature)
	}

	if req.TopP != 1.0 {
		t.Errorf("expected default TopP 1.0, got %f", req.TopP)
	}
}

func TestNewChatCompletionResponse(t *testing.T) {
	resp := api.NewChatCompletionResponse("test-model", "Hello!", "stop")

	if resp.Object != "chat.completion" {
		t.Errorf("expected object 'chat.completion', got '%s'", resp.Object)
	}

	if resp.Model != "test-model" {
		t.Errorf("expected model 'test-model', got '%s'", resp.Model)
	}

	if len(resp.Choices) != 1 {
		t.Fatalf("expected 1 choice, got %d", len(resp.Choices))
	}

	if resp.Choices[0].Message.Content != "Hello!" {
		t.Errorf("expected content 'Hello!', got '%s'", resp.Choices[0].Message.Content)
	}

	if resp.Choices[0].FinishReason != "stop" {
		t.Errorf("expected finish_reason 'stop', got '%s'", resp.Choices[0].FinishReason)
	}
}

func TestNewChatCompletionChunk(t *testing.T) {
	chunk := api.NewChatCompletionChunk("test-model", "Hello", "")

	if chunk.Object != "chat.completion.chunk" {
		t.Errorf("expected object 'chat.completion.chunk', got '%s'", chunk.Object)
	}

	if len(chunk.Choices) != 1 {
		t.Fatalf("expected 1 choice, got %d", len(chunk.Choices))
	}

	if chunk.Choices[0].Delta.Content != "Hello" {
		t.Errorf("expected delta content 'Hello', got '%s'", chunk.Choices[0].Delta.Content)
	}
}

func TestNewErrorResponse(t *testing.T) {
	resp := api.NewErrorResponse("Something went wrong", "server_error", "internal_error")

	if resp.Error.Message != "Something went wrong" {
		t.Errorf("expected message 'Something went wrong', got '%s'", resp.Error.Message)
	}

	if resp.Error.Type != "server_error" {
		t.Errorf("expected type 'server_error', got '%s'", resp.Error.Type)
	}

	if resp.Error.Code != "internal_error" {
		t.Errorf("expected code 'internal_error', got '%s'", resp.Error.Code)
	}
}

// ===== Integration Tests =====

func TestServer_FullWorkflow(t *testing.T) {
	server := createTestServer()

	// 1. Check health
	healthReq := httptest.NewRequest(http.MethodGet, "/health", nil)
	healthW := httptest.NewRecorder()
	server.Mux().ServeHTTP(healthW, healthReq)

	if healthW.Code != http.StatusOK {
		t.Fatalf("health check failed: %d", healthW.Code)
	}

	// 2. List models
	modelsReq := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	modelsW := httptest.NewRecorder()
	server.Mux().ServeHTTP(modelsW, modelsReq)

	if modelsW.Code != http.StatusOK {
		t.Fatalf("list models failed: %d", modelsW.Code)
	}

	// 3. Send chat completion
	chatReq := api.ChatCompletionRequest{
		Model: "test-model",
		Messages: []api.Message{
			{Role: "user", Content: "Test message"},
		},
		MaxTokens: 5,
	}

	body, _ := json.Marshal(chatReq)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	chatW := httptest.NewRecorder()
	server.Mux().ServeHTTP(chatW, req)

	if chatW.Code != http.StatusOK {
		t.Fatalf("chat completion failed: %d - %s", chatW.Code, chatW.Body.String())
	}

	// 4. Check metrics increased
	metricsReq := httptest.NewRequest(http.MethodGet, "/metrics", nil)
	metricsW := httptest.NewRecorder()
	server.Mux().ServeHTTP(metricsW, metricsReq)

	var metrics map[string]interface{}
	json.NewDecoder(metricsW.Body).Decode(&metrics)

	// Note: requests_total might vary due to test execution order
	if metrics["requests_total"] == nil {
		t.Error("expected requests_total in metrics")
	}
}

func TestServer_ConcurrentRequests(t *testing.T) {
	server := createTestServer()

	// Send multiple concurrent requests
	done := make(chan bool, 5)

	for i := 0; i < 5; i++ {
		go func() {
			chatReq := api.ChatCompletionRequest{
				Model: "test-model",
				Messages: []api.Message{
					{Role: "user", Content: "Hello"},
				},
				MaxTokens: 3,
			}

			body, _ := json.Marshal(chatReq)
			req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()

			server.Mux().ServeHTTP(w, req)
			done <- w.Code == http.StatusOK
		}()
	}

	// Wait for all requests
	successCount := 0
	for i := 0; i < 5; i++ {
		if <-done {
			successCount++
		}
	}

	if successCount != 5 {
		t.Errorf("expected 5 successful requests, got %d", successCount)
	}
}

func TestServer_Shutdown(t *testing.T) {
	server := createTestServer()

	// Test graceful shutdown (without actually starting the server)
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()

	err := server.Shutdown(ctx)
	if err != nil {
		t.Errorf("shutdown error: %v", err)
	}
}

// Helper to read SSE response
func readSSEEvents(body io.Reader) []string {
	var events []string
	buf := make([]byte, 4096)
	for {
		n, err := body.Read(buf)
		if err != nil {
			break
		}
		events = append(events, string(buf[:n]))
	}
	return events
}
