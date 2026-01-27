// Package api provides HTTP API handlers for the NeuroGrid inference engine.
package api

import (
	"context"
	"encoding/json"
	"log"
	"net/http"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/neurogrid/engine/pkg/inference"
	"github.com/neurogrid/engine/pkg/metrics"
	"github.com/neurogrid/engine/pkg/model"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

const (
	// Version is the API version.
	Version = "1.0.0"
)

// ServerConfig holds server configuration.
type ServerConfig struct {
	Addr         string
	ReadTimeout  time.Duration
	WriteTimeout time.Duration
	ModelName    string
	EnableCORS   bool
}

// DefaultServerConfig returns default server configuration.
func DefaultServerConfig() ServerConfig {
	return ServerConfig{
		Addr:         ":8080",
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 120 * time.Second,
		ModelName:    "llama-7b",
		EnableCORS:   true,
	}
}

// ClusterInfoProvider provides cluster information.
type ClusterInfoProvider interface {
	GetClusterInfo() ClusterInfo
}

// ClusterInfo contains information about the cluster.
type ClusterInfo struct {
	PeerID           string            `json:"peer_id"`
	ConnectedPeers   int               `json:"connected_peers"`
	LayerAssignments map[int]string    `json:"layer_assignments"`
	MemoryUsage      map[string]uint64 `json:"memory_usage"`
	TotalLayers      int               `json:"total_layers"`
	LoadedLayers     int               `json:"loaded_layers"`
}

// Server is the HTTP API server.
type Server struct {
	engine          *inference.Engine
	config          ServerConfig
	server          *http.Server
	mux             *http.ServeMux
	startTime       time.Time
	reqCounter      uint64
	clusterProvider ClusterInfoProvider
}

// NewServer creates a new API server.
func NewServer(engine *inference.Engine, config ServerConfig) *Server {
	s := &Server{
		engine:    engine,
		config:    config,
		mux:       http.NewServeMux(),
		startTime: time.Now(),
	}

	s.setupRoutes()
	return s
}

// SetClusterInfoProvider sets the cluster info provider.
func (s *Server) SetClusterInfoProvider(provider ClusterInfoProvider) {
	s.clusterProvider = provider
}

// setupRoutes configures the HTTP routes.
func (s *Server) setupRoutes() {
	// API endpoints
	s.mux.HandleFunc("/v1/chat/completions", s.handleChatCompletions)
	s.mux.HandleFunc("/v1/models", s.handleListModels)
	s.mux.HandleFunc("/v1/models/", s.handleGetModel)

	// Cluster endpoints
	s.mux.HandleFunc("/v1/cluster/info", s.handleClusterInfo)

	// Health and status
	s.mux.HandleFunc("/health", s.handleHealth)
	s.mux.HandleFunc("/ready", s.handleReady)

	// Prometheus metrics endpoint
	s.mux.Handle("/metrics", promhttp.Handler())

	// Root
	s.mux.HandleFunc("/", s.handleRoot)
}

// Start starts the HTTP server.
func (s *Server) Start() error {
	var handler http.Handler = s.mux

	// Apply middleware
	handler = s.loggingMiddleware(handler)
	if s.config.EnableCORS {
		handler = s.corsMiddleware(handler)
	}
	handler = s.recoveryMiddleware(handler)

	s.server = &http.Server{
		Addr:         s.config.Addr,
		Handler:      handler,
		ReadTimeout:  s.config.ReadTimeout,
		WriteTimeout: s.config.WriteTimeout,
	}

	log.Printf("Starting server on %s", s.config.Addr)
	return s.server.ListenAndServe()
}

// Shutdown gracefully shuts down the server.
func (s *Server) Shutdown(ctx context.Context) error {
	if s.server == nil {
		return nil
	}
	return s.server.Shutdown(ctx)
}

// corsMiddleware adds CORS headers.
func (s *Server) corsMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusOK)
			return
		}

		next.ServeHTTP(w, r)
	})
}

// loggingMiddleware logs HTTP requests and records metrics.
func (s *Server) loggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		reqID := atomic.AddUint64(&s.reqCounter, 1)

		// Track active requests
		metrics.ActiveRequests.Inc()
		defer metrics.ActiveRequests.Dec()

		// Wrap response writer to capture status
		rw := &responseWriter{ResponseWriter: w, statusCode: http.StatusOK}

		next.ServeHTTP(rw, r)

		duration := time.Since(start)
		log.Printf("[%d] %s %s %d %v", reqID, r.Method, r.URL.Path, rw.statusCode, duration)

		// Record metrics (skip /metrics endpoint to avoid recursion)
		if r.URL.Path != "/metrics" {
			metrics.RequestsTotal.WithLabelValues(r.Method, r.URL.Path, strconv.Itoa(rw.statusCode)).Inc()
			metrics.RequestDuration.WithLabelValues(r.Method, r.URL.Path).Observe(duration.Seconds())
		}
	})
}

// recoveryMiddleware recovers from panics.
func (s *Server) recoveryMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if err := recover(); err != nil {
				log.Printf("Panic recovered: %v", err)
				s.sendError(w, "Internal server error", "server_error", "internal_error", http.StatusInternalServerError)
			}
		}()
		next.ServeHTTP(w, r)
	})
}

// responseWriter wraps http.ResponseWriter to capture status code.
type responseWriter struct {
	http.ResponseWriter
	statusCode int
}

func (rw *responseWriter) WriteHeader(code int) {
	rw.statusCode = code
	rw.ResponseWriter.WriteHeader(code)
}

// Flush implements http.Flusher interface.
// This is required for SSE streaming to work through the logging middleware.
func (rw *responseWriter) Flush() {
	if f, ok := rw.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

// handleRoot handles the root endpoint.
func (s *Server) handleRoot(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}

	resp := map[string]string{
		"name":    "NeuroGrid Inference Engine",
		"version": Version,
		"status":  "running",
	}

	s.sendJSON(w, resp, http.StatusOK)
}

// handleHealth handles the health check endpoint.
func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	resp := HealthResponse{
		Status:    "healthy",
		Model:     s.config.ModelName,
		Timestamp: time.Now().Unix(),
		Version:   Version,
	}

	s.sendJSON(w, resp, http.StatusOK)
}

// handleReady handles the readiness check endpoint.
func (s *Server) handleReady(w http.ResponseWriter, r *http.Request) {
	// Check if engine is ready
	if s.engine == nil {
		s.sendError(w, "Engine not initialized", "not_ready", "not_ready", http.StatusServiceUnavailable)
		return
	}

	resp := map[string]interface{}{
		"ready":  true,
		"uptime": time.Since(s.startTime).Seconds(),
		"model":  s.config.ModelName,
	}

	s.sendJSON(w, resp, http.StatusOK)
}

// handleClusterInfo handles GET /v1/cluster/info.
func (s *Server) handleClusterInfo(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		s.sendError(w, "Method not allowed", "invalid_request", "method_not_allowed", http.StatusMethodNotAllowed)
		return
	}

	if s.clusterProvider == nil {
		// Return empty cluster info if no provider
		resp := ClusterInfo{
			PeerID:           "",
			ConnectedPeers:   0,
			LayerAssignments: make(map[int]string),
			MemoryUsage:      make(map[string]uint64),
			TotalLayers:      0,
			LoadedLayers:     0,
		}
		s.sendJSON(w, resp, http.StatusOK)
		return
	}

	info := s.clusterProvider.GetClusterInfo()
	s.sendJSON(w, info, http.StatusOK)
}

// handleListModels handles GET /v1/models.
func (s *Server) handleListModels(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		s.sendError(w, "Method not allowed", "invalid_request", "method_not_allowed", http.StatusMethodNotAllowed)
		return
	}

	resp := ModelsResponse{
		Object: "list",
		Data: []ModelInfo{
			{
				ID:      s.config.ModelName,
				Object:  "model",
				Created: s.startTime.Unix(),
				OwnedBy: "neurogrid",
			},
		},
	}

	s.sendJSON(w, resp, http.StatusOK)
}

// handleGetModel handles GET /v1/models/{model_id}.
func (s *Server) handleGetModel(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		s.sendError(w, "Method not allowed", "invalid_request", "method_not_allowed", http.StatusMethodNotAllowed)
		return
	}

	modelID := strings.TrimPrefix(r.URL.Path, "/v1/models/")
	if modelID == "" || modelID != s.config.ModelName {
		s.sendError(w, "Model not found", "not_found", "model_not_found", http.StatusNotFound)
		return
	}

	resp := ModelInfo{
		ID:      s.config.ModelName,
		Object:  "model",
		Created: s.startTime.Unix(),
		OwnedBy: "neurogrid",
	}

	s.sendJSON(w, resp, http.StatusOK)
}

// handleChatCompletions handles POST /v1/chat/completions.
func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		s.sendError(w, "Method not allowed", "invalid_request", "method_not_allowed", http.StatusMethodNotAllowed)
		return
	}

	// Parse request
	var req ChatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		s.sendError(w, "Invalid JSON: "+err.Error(), "invalid_request", "invalid_json", http.StatusBadRequest)
		return
	}

	// Validate request
	if len(req.Messages) == 0 {
		s.sendError(w, "Messages array is required", "invalid_request", "missing_messages", http.StatusBadRequest)
		return
	}

	req.SetDefaults()

	// Check for streaming
	if req.Stream {
		s.handleChatCompletionsStream(w, r, &req)
		return
	}

	// Build prompt from messages using the model-specific chat template
	prompt := buildPromptForModel(req.Messages, s.config.ModelName)

	// Get stop strings for this model (prevents generating new turns)
	stopStrings := getStopStringsForModel(s.config.ModelName)

	// Generate response
	genReq := &inference.GenerateRequest{
		Prompt:      prompt,
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		StopStrings: stopStrings,
	}

	ctx := r.Context()
	genResp, err := s.engine.Generate(ctx, genReq)
	if err != nil {
		if ctx.Err() != nil {
			s.sendError(w, "Request cancelled", "cancelled", "cancelled", http.StatusGatewayTimeout)
			return
		}
		s.sendError(w, "Generation failed: "+err.Error(), "server_error", "generation_failed", http.StatusInternalServerError)
		return
	}

	// Clean up output - remove any leaked template tokens
	cleanedText := cleanModelOutput(genResp.Text, s.config.ModelName)

	// Build response
	resp := NewChatCompletionResponse(s.config.ModelName, cleanedText, genResp.StopReason)

	s.sendJSON(w, resp, http.StatusOK)
}

// handleChatCompletionsStream handles streaming chat completions.
func (s *Server) handleChatCompletionsStream(w http.ResponseWriter, r *http.Request, req *ChatCompletionRequest) {
	// Create SSE writer
	sse, err := NewSSEWriter(w)
	if err != nil {
		s.sendError(w, err.Error(), "server_error", "no_streaming", http.StatusInternalServerError)
		return
	}

	// Create stream state for consistent ID/timestamp across chunks
	streamState := NewStreamState(s.config.ModelName, time.Now().Unix())

	// Build prompt using model-specific chat template
	prompt := buildPromptForModel(req.Messages, s.config.ModelName)

	// Get stop strings for this model (prevents generating new turns)
	stopStrings := getStopStringsForModel(s.config.ModelName)

	// Build generation request
	genReq := &inference.GenerateRequest{
		Prompt:      prompt,
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		StopStrings: stopStrings,
	}

	ctx := r.Context()

	// Send first chunk with role
	if err := sse.WriteEvent(streamState.CreateRoleChunk()); err != nil {
		log.Printf("Error writing role chunk: %v", err)
		return
	}

	// Stream tokens as they are generated
	var lastStopReason string
	err = s.engine.GenerateStream(ctx, genReq, func(token inference.StreamToken) error {
		// Send content chunk for each token
		if token.Text != "" {
			if err := sse.WriteEvent(streamState.CreateContentChunk(token.Text)); err != nil {
				return err
			}
		}

		// Track stop reason for final chunk
		if token.IsFinal {
			lastStopReason = token.StopReason
		}

		return nil
	})

	if err != nil {
		log.Printf("Error during streaming generation: %v", err)
		sse.WriteError(err.Error(), "server_error")
		return
	}

	// Send final chunk with finish reason
	if err := sse.WriteEvent(streamState.CreateFinalChunk(lastStopReason)); err != nil {
		log.Printf("Error writing final chunk: %v", err)
		return
	}

	// Send [DONE] marker
	if err := sse.Close(); err != nil {
		log.Printf("Error closing SSE stream: %v", err)
	}
}

// buildPromptForModel builds a prompt using the appropriate chat template for the model.
// Different models use different chat formats (Llama 2, Llama 3, TinyLlama, Mistral, etc.)
func buildPromptForModel(messages []Message, modelName string) string {
	// Convert API messages to model messages
	modelMessages := make([]model.Message, len(messages))
	for i, msg := range messages {
		modelMessages[i] = model.Message{
			Role:    msg.Role,
			Content: msg.Content,
		}
	}

	// Get the appropriate chat template for this model
	chatTemplate := model.ChatTemplateFactory(modelName)

	// Format the messages using the model-specific template
	return chatTemplate.Format(modelMessages)
}

// buildLlamaPrompt is kept for backward compatibility.
// It delegates to buildPromptForModel with an empty model name (defaults to Llama 2).
func buildLlamaPrompt(messages []Message) string {
	return buildPromptForModel(messages, "")
}

// getStopStringsForModel returns stop strings that prevent the model from generating new turns.
// Different chat templates use different markers for role changes.
func getStopStringsForModel(modelName string) []string {
	// Return empty - we'll clean up the output instead of stopping early
	// This prevents issues with models that generate template tokens at the start
	return []string{}
}

// cleanModelOutput removes leaked template tokens from model output.
// Some models (especially smaller ones like TinyLlama) may generate template markers
// as part of their response. This function strips them out.
func cleanModelOutput(text string, modelName string) string {
	lower := strings.ToLower(modelName)

	// Define markers to remove based on model type
	var markers []string

	if strings.Contains(lower, "tinyllama") {
		markers = []string{"<|system|>", "<|user|>", "<|assistant|>", "</s>"}
	} else if strings.Contains(lower, "llama-3") || strings.Contains(lower, "llama3") {
		markers = []string{"<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>", "<|begin_of_text|>"}
	} else if strings.Contains(lower, "mistral") || strings.Contains(lower, "mixtral") {
		markers = []string{"[INST]", "[/INST]", "<<SYS>>", "<</SYS>>"}
	} else {
		// Llama 2 default
		markers = []string{"[INST]", "[/INST]", "<<SYS>>", "<</SYS>>", "</s>", "<s>"}
	}

	// Remove all markers
	result := text
	for _, marker := range markers {
		result = strings.ReplaceAll(result, marker, "")
	}

	// Remove role names that might appear after markers
	roleNames := []string{"system\n", "user\n", "assistant\n", "system", "user", "assistant"}
	for _, role := range roleNames {
		// Only remove if at the start of the text or after a newline
		if strings.HasPrefix(result, role) {
			result = strings.TrimPrefix(result, role)
		}
		result = strings.ReplaceAll(result, "\n"+role, "\n")
	}

	// Clean up multiple newlines and trim
	for strings.Contains(result, "\n\n\n") {
		result = strings.ReplaceAll(result, "\n\n\n", "\n\n")
	}
	result = strings.TrimSpace(result)

	return result
}

// sendJSON sends a JSON response.
func (s *Server) sendJSON(w http.ResponseWriter, data interface{}, status int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(data)
}

// sendError sends an error response.
func (s *Server) sendError(w http.ResponseWriter, message, errorType, code string, status int) {
	resp := NewErrorResponse(message, errorType, code)
	s.sendJSON(w, resp, status)
}

// Mux returns the server's HTTP mux for testing.
func (s *Server) Mux() *http.ServeMux {
	return s.mux
}
