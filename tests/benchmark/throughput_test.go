// Package benchmark provides performance benchmarks for the NeuroGrid engine.
package benchmark

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"runtime"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/neurogrid/engine/api"
	"github.com/neurogrid/engine/pkg/inference"
	"github.com/neurogrid/engine/pkg/scheduler"
	"github.com/neurogrid/engine/pkg/transport"
	"github.com/neurogrid/engine/pkg/types"
)

// BenchmarkConfig holds benchmark configuration parameters.
type BenchmarkConfig struct {
	MaxTokens        int
	Temperature      float32
	ConcurrentUsers  int
	WarmupIterations int
	TotalIterations  int
}

// DefaultBenchmarkConfig returns sensible defaults for benchmarks.
func DefaultBenchmarkConfig() BenchmarkConfig {
	return BenchmarkConfig{
		MaxTokens:        32,
		Temperature:      0.7,
		ConcurrentUsers:  10,
		WarmupIterations: 5,
		TotalIterations:  100,
	}
}

// BenchmarkThroughputSingleRequest benchmarks single request throughput.
func BenchmarkThroughputSingleRequest(b *testing.B) {
	_, server := setupBenchmarkEnvironment(b)

	req := api.ChatCompletionRequest{
		Model: "llama-7b",
		Messages: []api.Message{
			{Role: "user", Content: "Hello, how are you?"},
		},
		MaxTokens:   32,
		Temperature: 0.7,
	}

	body, _ := json.Marshal(req)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		httpReq := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(body))
		httpReq.Header.Set("Content-Type", "application/json")

		recorder := httptest.NewRecorder()
		server.Mux().ServeHTTP(recorder, httpReq)

		if recorder.Code != http.StatusOK {
			b.Fatalf("Request failed with status %d", recorder.Code)
		}
	}
}

// BenchmarkThroughputConcurrent benchmarks concurrent request throughput.
func BenchmarkThroughputConcurrent(b *testing.B) {
	_, server := setupBenchmarkEnvironment(b)

	concurrencyLevels := []int{1, 2, 4, 8, 16, 32}

	for _, concurrency := range concurrencyLevels {
		b.Run(fmt.Sprintf("Concurrency-%d", concurrency), func(b *testing.B) {
			req := api.ChatCompletionRequest{
				Model: "llama-7b",
				Messages: []api.Message{
					{Role: "user", Content: "Hello"},
				},
				MaxTokens:   16,
				Temperature: 0.7,
			}
			body, _ := json.Marshal(req)

			b.ResetTimer()
			b.ReportAllocs()

			var wg sync.WaitGroup
			requestsPerWorker := b.N / concurrency
			if requestsPerWorker == 0 {
				requestsPerWorker = 1
			}

			for w := 0; w < concurrency; w++ {
				wg.Add(1)
				go func() {
					defer wg.Done()
					for i := 0; i < requestsPerWorker; i++ {
						httpReq := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(body))
						httpReq.Header.Set("Content-Type", "application/json")

						recorder := httptest.NewRecorder()
						server.Mux().ServeHTTP(recorder, httpReq)
					}
				}()
			}
			wg.Wait()
		})
	}
}

// BenchmarkLatencyPerToken benchmarks latency per generated token.
func BenchmarkLatencyPerToken(b *testing.B) {
	_, server := setupBenchmarkEnvironment(b)

	tokenCounts := []int{8, 16, 32, 64, 128}

	for _, maxTokens := range tokenCounts {
		b.Run(fmt.Sprintf("Tokens-%d", maxTokens), func(b *testing.B) {
			req := api.ChatCompletionRequest{
				Model: "llama-7b",
				Messages: []api.Message{
					{Role: "user", Content: "Tell me a story"},
				},
				MaxTokens:   maxTokens,
				Temperature: 0.7,
			}
			body, _ := json.Marshal(req)

			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				httpReq := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(body))
				httpReq.Header.Set("Content-Type", "application/json")

				recorder := httptest.NewRecorder()
				server.Mux().ServeHTTP(recorder, httpReq)
			}
		})
	}
}

// BenchmarkTimeToFirstToken measures time to first token in streaming mode.
func BenchmarkTimeToFirstToken(b *testing.B) {
	_, server := setupBenchmarkEnvironment(b)

	req := api.ChatCompletionRequest{
		Model: "llama-7b",
		Messages: []api.Message{
			{Role: "user", Content: "Hello"},
		},
		MaxTokens:   32,
		Temperature: 0.7,
		Stream:      true,
	}
	body, _ := json.Marshal(req)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		httpReq := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(body))
		httpReq.Header.Set("Content-Type", "application/json")

		recorder := httptest.NewRecorder()
		server.Mux().ServeHTTP(recorder, httpReq)
	}
}

// BenchmarkSchedulerAssignment benchmarks layer assignment computation.
func BenchmarkSchedulerAssignment(b *testing.B) {
	configs := []struct {
		name   string
		config scheduler.ModelConfig
	}{
		{"Llama7B", scheduler.DefaultLlama7BConfig()},
		{"Llama13B", scheduler.DefaultLlama13BConfig()},
	}

	for _, cfg := range configs {
		b.Run(cfg.name, func(b *testing.B) {
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				sched := scheduler.NewScheduler(cfg.config)

				// Register 4 peers
				for j := 0; j < 4; j++ {
					peerID := fmt.Sprintf("peer-%d", j)
					sched.RegisterPeer(peerID, 8*1024*1024*1024, 1*1024*1024*1024)
				}

				_, err := sched.ComputeAssignments()
				if err != nil {
					b.Fatalf("Assignment failed: %v", err)
				}
			}
		})
	}
}

// BenchmarkTransportRouter benchmarks activation routing.
func BenchmarkTransportRouter(b *testing.B) {
	router := transport.NewTransportRouter()

	// Register transports
	for i := 0; i < 4; i++ {
		peerID := fmt.Sprintf("peer-%d", i)
		t, err := transport.NewCUDATransport(0, i)
		if err != nil {
			b.Fatalf("Failed to create transport: %v", err)
		}
		router.RegisterRemoteTransport(peerID, t)
	}

	// Assign layers
	for i := 0; i < 32; i++ {
		peerID := fmt.Sprintf("peer-%d", i%4)
		router.AssignLayerToPeer(i, peerID)
	}

	ctx := context.Background()
	activation := make([]byte, 4096*2) // Hidden state in FP16

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		layerID := i % 32
		_ = router.RouteActivation(ctx, layerID, uint64(i), activation)
	}
}

// BenchmarkEngineGenerate benchmarks the inference engine generate function.
func BenchmarkEngineGenerate(b *testing.B) {
	engine := setupBenchmarkEngine(b)

	ctx := context.Background()
	req := &inference.GenerateRequest{
		Prompt:      "Hello, how are you?",
		MaxTokens:   32,
		Temperature: 0.7,
		TopP:        0.9,
	}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, err := engine.Generate(ctx, req)
		if err != nil {
			b.Fatalf("Generate failed: %v", err)
		}
	}
}

// BenchmarkMemoryUsage benchmarks memory usage under load.
func BenchmarkMemoryUsage(b *testing.B) {
	_, server := setupBenchmarkEnvironment(b)

	req := api.ChatCompletionRequest{
		Model: "llama-7b",
		Messages: []api.Message{
			{Role: "user", Content: "Hello"},
		},
		MaxTokens:   32,
		Temperature: 0.7,
	}
	body, _ := json.Marshal(req)

	// Force GC before measurement
	runtime.GC()

	var memStatsBefore, memStatsAfter runtime.MemStats
	runtime.ReadMemStats(&memStatsBefore)

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		httpReq := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(body))
		httpReq.Header.Set("Content-Type", "application/json")

		recorder := httptest.NewRecorder()
		server.Mux().ServeHTTP(recorder, httpReq)
	}

	b.StopTimer()

	runtime.ReadMemStats(&memStatsAfter)

	allocatedBytes := memStatsAfter.TotalAlloc - memStatsBefore.TotalAlloc
	b.ReportMetric(float64(allocatedBytes)/float64(b.N), "bytes/op")
}

// BenchmarkContextSizes benchmarks different input context sizes.
func BenchmarkContextSizes(b *testing.B) {
	_, server := setupBenchmarkEnvironment(b)

	contextSizes := []int{1, 5, 10, 20, 50}

	for _, size := range contextSizes {
		b.Run(fmt.Sprintf("Context-%d-turns", size), func(b *testing.B) {
			messages := make([]api.Message, 0, size*2+1)
			messages = append(messages, api.Message{Role: "system", Content: "You are helpful."})

			for i := 0; i < size; i++ {
				messages = append(messages,
					api.Message{Role: "user", Content: fmt.Sprintf("Message %d", i)},
					api.Message{Role: "assistant", Content: fmt.Sprintf("Response %d", i)},
				)
			}
			messages = append(messages, api.Message{Role: "user", Content: "Final question"})

			req := api.ChatCompletionRequest{
				Model:       "llama-7b",
				Messages:    messages,
				MaxTokens:   16,
				Temperature: 0.7,
			}
			body, _ := json.Marshal(req)

			b.ResetTimer()
			b.ReportAllocs()

			for i := 0; i < b.N; i++ {
				httpReq := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(body))
				httpReq.Header.Set("Content-Type", "application/json")

				recorder := httptest.NewRecorder()
				server.Mux().ServeHTTP(recorder, httpReq)
			}
		})
	}
}

// BenchmarkHealthEndpoint benchmarks the health check endpoint.
func BenchmarkHealthEndpoint(b *testing.B) {
	_, server := setupBenchmarkEnvironment(b)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		req := httptest.NewRequest("GET", "/health", nil)
		recorder := httptest.NewRecorder()
		server.Mux().ServeHTTP(recorder, req)
	}
}

// BenchmarkModelsEndpoint benchmarks the models listing endpoint.
func BenchmarkModelsEndpoint(b *testing.B) {
	_, server := setupBenchmarkEnvironment(b)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		req := httptest.NewRequest("GET", "/v1/models", nil)
		recorder := httptest.NewRecorder()
		server.Mux().ServeHTTP(recorder, req)
	}
}

// TestThroughputMetrics runs detailed throughput measurements and reports.
func TestThroughputMetrics(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping throughput test in short mode")
	}

	_, server := setupTestEnvironment(t)

	config := DefaultBenchmarkConfig()

	// Warmup
	t.Log("Running warmup iterations...")
	for i := 0; i < config.WarmupIterations; i++ {
		makeRequest(server, config.MaxTokens, config.Temperature)
	}

	// Collect metrics
	var latencies []time.Duration
	var totalTokens int64
	var errors int64

	start := time.Now()

	var wg sync.WaitGroup
	var mu sync.Mutex

	for w := 0; w < config.ConcurrentUsers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()

			iterations := config.TotalIterations / config.ConcurrentUsers
			for i := 0; i < iterations; i++ {
				reqStart := time.Now()
				tokens, err := makeRequest(server, config.MaxTokens, config.Temperature)
				latency := time.Since(reqStart)

				mu.Lock()
				if err != nil {
					errors++
				} else {
					latencies = append(latencies, latency)
					totalTokens += int64(tokens)
				}
				mu.Unlock()
			}
		}()
	}

	wg.Wait()
	elapsed := time.Since(start)

	// Calculate metrics
	throughput := float64(len(latencies)) / elapsed.Seconds()
	tokensPerSecond := float64(totalTokens) / elapsed.Seconds()

	var totalLatency time.Duration
	var minLatency = time.Hour
	var maxLatency time.Duration

	for _, l := range latencies {
		totalLatency += l
		if l < minLatency {
			minLatency = l
		}
		if l > maxLatency {
			maxLatency = l
		}
	}

	var avgLatency time.Duration
	if len(latencies) > 0 {
		avgLatency = totalLatency / time.Duration(len(latencies))
	}

	// Report
	t.Logf("=== Throughput Metrics ===")
	t.Logf("Total Requests:    %d", len(latencies))
	t.Logf("Total Errors:      %d", errors)
	t.Logf("Elapsed Time:      %v", elapsed)
	t.Logf("Requests/sec:      %.2f", throughput)
	t.Logf("Tokens/sec:        %.2f", tokensPerSecond)
	t.Logf("Avg Latency:       %v", avgLatency)
	t.Logf("Min Latency:       %v", minLatency)
	t.Logf("Max Latency:       %v", maxLatency)
	t.Logf("Concurrent Users:  %d", config.ConcurrentUsers)
}

// TestLatencyPercentiles measures latency percentiles.
func TestLatencyPercentiles(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping latency percentile test in short mode")
	}

	_, server := setupTestEnvironment(t)

	iterations := 100
	latencies := make([]time.Duration, 0, iterations)

	// Warmup
	for i := 0; i < 5; i++ {
		makeRequest(server, 32, 0.7)
	}

	// Measure
	for i := 0; i < iterations; i++ {
		start := time.Now()
		makeRequest(server, 32, 0.7)
		latencies = append(latencies, time.Since(start))
	}

	// Sort for percentile calculation
	sortDurations(latencies)

	p50 := latencies[len(latencies)*50/100]
	p90 := latencies[len(latencies)*90/100]
	p95 := latencies[len(latencies)*95/100]
	p99 := latencies[len(latencies)*99/100]

	t.Logf("=== Latency Percentiles (%d samples) ===", iterations)
	t.Logf("P50:  %v", p50)
	t.Logf("P90:  %v", p90)
	t.Logf("P95:  %v", p95)
	t.Logf("P99:  %v", p99)
	t.Logf("Min:  %v", latencies[0])
	t.Logf("Max:  %v", latencies[len(latencies)-1])
}

// TestSustainedLoad tests performance under sustained load.
func TestSustainedLoad(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping sustained load test in short mode")
	}

	_, server := setupTestEnvironment(t)

	duration := 10 * time.Second
	concurrency := 5

	var requestCount int64
	var errorCount int64

	ctx, cancel := context.WithTimeout(context.Background(), duration)
	defer cancel()

	var wg sync.WaitGroup
	for w := 0; w < concurrency; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				select {
				case <-ctx.Done():
					return
				default:
					_, err := makeRequest(server, 16, 0.7)
					atomic.AddInt64(&requestCount, 1)
					if err != nil {
						atomic.AddInt64(&errorCount, 1)
					}
				}
			}
		}()
	}

	wg.Wait()

	t.Logf("=== Sustained Load Test (%v) ===", duration)
	t.Logf("Total Requests: %d", requestCount)
	t.Logf("Total Errors:   %d", errorCount)
	t.Logf("Requests/sec:   %.2f", float64(requestCount)/duration.Seconds())
	t.Logf("Concurrency:    %d", concurrency)
}

// Helper functions

func setupBenchmarkEnvironment(b *testing.B) (*inference.Engine, *api.Server) {
	b.Helper()

	config := types.Llama7BConfig()
	sched := scheduler.NewScheduler(scheduler.ModelConfig{
		HiddenSize:       int64(config.HiddenSize),
		IntermediateSize: int64(config.IntermediateSize),
		NumLayers:        config.NumLayers,
		NumKVHeads:       config.NumKVHeads,
		HeadDim:          config.HeadDim,
		MaxSeqLen:        config.MaxSeqLen,
		VocabSize:        int64(config.VocabSize),
	})

	sched.RegisterPeer("local", 8*1024*1024*1024, 1*1024*1024*1024)

	router := transport.NewTransportRouter()
	localTransport, _ := transport.NewCUDATransport(0, 0)
	router.RegisterLocalTransport(0, localTransport)
	router.RegisterRemoteTransport("local", localTransport)

	engineConfig := inference.EngineConfig{
		ModelConfig: config,
		LocalPeerID: "local",
	}

	engine := inference.NewEngine(engineConfig)
	engine.SetScheduler(sched)
	engine.SetRouter(router)

	assignments, _ := sched.ComputeAssignments()
	engine.SetAssignments(assignments)
	engine.SetTokenizer(NewBenchmarkTokenizer())

	serverConfig := api.ServerConfig{
		Addr:         ":0",
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 120 * time.Second,
		ModelName:    "llama-7b",
		EnableCORS:   true,
	}

	server := api.NewServer(engine, serverConfig)

	return engine, server
}

func setupBenchmarkEngine(b *testing.B) *inference.Engine {
	b.Helper()

	config := types.Llama7BConfig()
	sched := scheduler.NewScheduler(scheduler.ModelConfig{
		HiddenSize:       int64(config.HiddenSize),
		IntermediateSize: int64(config.IntermediateSize),
		NumLayers:        config.NumLayers,
		NumKVHeads:       config.NumKVHeads,
		HeadDim:          config.HeadDim,
		MaxSeqLen:        config.MaxSeqLen,
		VocabSize:        int64(config.VocabSize),
	})

	sched.RegisterPeer("local", 8*1024*1024*1024, 1*1024*1024*1024)

	router := transport.NewTransportRouter()
	localTransport, _ := transport.NewCUDATransport(0, 0)
	router.RegisterLocalTransport(0, localTransport)
	router.RegisterRemoteTransport("local", localTransport)

	engineConfig := inference.EngineConfig{
		ModelConfig: config,
		LocalPeerID: "local",
	}

	engine := inference.NewEngine(engineConfig)
	engine.SetScheduler(sched)
	engine.SetRouter(router)

	assignments, _ := sched.ComputeAssignments()
	engine.SetAssignments(assignments)
	engine.SetTokenizer(NewBenchmarkTokenizer())

	return engine
}

func setupTestEnvironment(t *testing.T) (*inference.Engine, *api.Server) {
	t.Helper()

	config := types.Llama7BConfig()
	sched := scheduler.NewScheduler(scheduler.ModelConfig{
		HiddenSize:       int64(config.HiddenSize),
		IntermediateSize: int64(config.IntermediateSize),
		NumLayers:        config.NumLayers,
		NumKVHeads:       config.NumKVHeads,
		HeadDim:          config.HeadDim,
		MaxSeqLen:        config.MaxSeqLen,
		VocabSize:        int64(config.VocabSize),
	})

	sched.RegisterPeer("local", 8*1024*1024*1024, 1*1024*1024*1024)

	router := transport.NewTransportRouter()
	localTransport, _ := transport.NewCUDATransport(0, 0)
	router.RegisterLocalTransport(0, localTransport)
	router.RegisterRemoteTransport("local", localTransport)

	engineConfig := inference.EngineConfig{
		ModelConfig: config,
		LocalPeerID: "local",
	}

	engine := inference.NewEngine(engineConfig)
	engine.SetScheduler(sched)
	engine.SetRouter(router)

	assignments, _ := sched.ComputeAssignments()
	engine.SetAssignments(assignments)
	engine.SetTokenizer(NewBenchmarkTokenizer())

	serverConfig := api.ServerConfig{
		Addr:         ":0",
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 120 * time.Second,
		ModelName:    "llama-7b",
		EnableCORS:   true,
	}

	server := api.NewServer(engine, serverConfig)

	return engine, server
}

func makeRequest(server *api.Server, maxTokens int, temperature float32) (int, error) {
	req := api.ChatCompletionRequest{
		Model: "llama-7b",
		Messages: []api.Message{
			{Role: "user", Content: "Hello"},
		},
		MaxTokens:   maxTokens,
		Temperature: temperature,
	}
	body, _ := json.Marshal(req)

	httpReq := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewReader(body))
	httpReq.Header.Set("Content-Type", "application/json")

	recorder := httptest.NewRecorder()
	server.Mux().ServeHTTP(recorder, httpReq)

	if recorder.Code != http.StatusOK {
		return 0, fmt.Errorf("request failed with status %d", recorder.Code)
	}

	var resp api.ChatCompletionResponse
	if err := json.Unmarshal(recorder.Body.Bytes(), &resp); err != nil {
		return 0, err
	}

	if resp.Usage != nil {
		return resp.Usage.CompletionTokens, nil
	}
	return maxTokens, nil
}

func sortDurations(d []time.Duration) {
	for i := 1; i < len(d); i++ {
		for j := i; j > 0 && d[j-1] > d[j]; j-- {
			d[j], d[j-1] = d[j-1], d[j]
		}
	}
}

// BenchmarkTokenizer for testing
type BenchmarkTokenizer struct {
	vocab map[string]int
}

func NewBenchmarkTokenizer() *BenchmarkTokenizer {
	return &BenchmarkTokenizer{
		vocab: map[string]int{
			"<s>":   1,
			"</s>":  2,
			"hello": 100,
			"world": 101,
			" ":     3,
		},
	}
}

func (t *BenchmarkTokenizer) Encode(text string) ([]int, error) {
	// Simple tokenization for benchmarking
	tokens := make([]int, 0, len(text)/4+1)
	tokens = append(tokens, 1) // BOS
	for i := 0; i < len(text); i += 4 {
		tokens = append(tokens, 100+i%100)
	}
	return tokens, nil
}

func (t *BenchmarkTokenizer) Decode(tokens []int) (string, error) {
	result := ""
	for range tokens {
		result += "word "
	}
	return result, nil
}

func (t *BenchmarkTokenizer) EOSToken() int  { return 2 }
func (t *BenchmarkTokenizer) BOSToken() int  { return 1 }
func (t *BenchmarkTokenizer) VocabSize() int { return 32000 }
