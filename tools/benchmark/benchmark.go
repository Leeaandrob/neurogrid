// Package main provides a comprehensive benchmark tool for the NeuroGrid inference engine.
// Measures TTFT, ITL, TPOT, TPS, and latency percentiles comparable to vLLM/SGLang benchmarks.
package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"net/http"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// ChatCompletionRequest is the OpenAI-compatible request format
type ChatCompletionRequest struct {
	Model       string    `json:"model"`
	Messages    []Message `json:"messages"`
	MaxTokens   int       `json:"max_tokens"`
	Temperature float64   `json:"temperature"`
	Stream      bool      `json:"stream"`
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ChatCompletionResponse is the OpenAI-compatible response format
type ChatCompletionResponse struct {
	ID      string   `json:"id"`
	Object  string   `json:"object"`
	Created int64    `json:"created"`
	Model   string   `json:"model"`
	Choices []Choice `json:"choices"`
	Usage   Usage    `json:"usage,omitempty"`
}

// StreamChunk for SSE streaming responses
type StreamChunk struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []StreamChoice `json:"choices"`
}

type StreamChoice struct {
	Index        int         `json:"index"`
	Delta        StreamDelta `json:"delta"`
	FinishReason string      `json:"finish_reason,omitempty"`
}

type StreamDelta struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

type Choice struct {
	Index        int     `json:"index"`
	Message      Message `json:"message"`
	FinishReason string  `json:"finish_reason"`
}

type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// BenchmarkResult holds metrics for a single request
type BenchmarkResult struct {
	TTFT             time.Duration   // Time to first token
	TotalLatency     time.Duration   // Total request latency
	OutputTokens     int             // Number of output tokens
	PromptTokens     int             // Number of input tokens
	TPOT             time.Duration   // Time per output token (excluding TTFT)
	ITLs             []time.Duration // Inter-token latencies (time between consecutive tokens)
	ITLMean          time.Duration   // Mean inter-token latency
	Success          bool
	Error            string
}

// BenchmarkStats holds aggregated statistics
type BenchmarkStats struct {
	TotalRequests     int
	SuccessfulReqs    int
	FailedReqs        int
	TotalOutputTokens int
	TotalPromptTokens int

	// TTFT stats (milliseconds)
	TTFTMin  float64
	TTFTMax  float64
	TTFTMean float64
	TTFTP50  float64
	TTFTP90  float64
	TTFTP95  float64
	TTFTP99  float64

	// ITL stats (milliseconds) - Inter-Token Latency
	ITLMin  float64
	ITLMax  float64
	ITLMean float64
	ITLP50  float64
	ITLP90  float64
	ITLP95  float64
	ITLP99  float64

	// TPOT stats (milliseconds per token)
	TPOTMin  float64
	TPOTMax  float64
	TPOTMean float64
	TPOTP50  float64
	TPOTP90  float64
	TPOTP95  float64
	TPOTP99  float64

	// Total latency stats (milliseconds)
	LatencyMin  float64
	LatencyMax  float64
	LatencyMean float64
	LatencyP50  float64
	LatencyP90  float64
	LatencyP95  float64
	LatencyP99  float64

	// Throughput
	TotalDuration      time.Duration
	RequestsPerSec     float64
	OutputTokensPerSec float64

	// Errors
	Errors []string
}

func main() {
	// Command line flags
	endpoint := flag.String("endpoint", "http://localhost:8089/v1/chat/completions", "API endpoint")
	model := flag.String("model", "tinyllama", "Model name")
	numRequests := flag.Int("n", 100, "Number of requests")
	concurrency := flag.Int("c", 1, "Concurrent requests")
	maxTokens := flag.Int("max-tokens", 50, "Max tokens to generate")
	promptLen := flag.String("prompt", "short", "Prompt length: short, medium, long")
	warmup := flag.Int("warmup", 3, "Warmup requests (not counted)")
	streaming := flag.Bool("stream", false, "Use streaming mode for accurate ITL measurement")
	outputJSON := flag.Bool("json", false, "Output results as JSON")
	flag.Parse()

	prompt := getPrompt(*promptLen)

	fmt.Println("╔══════════════════════════════════════════════════════════════╗")
	fmt.Println("║           NeuroGrid Inference Engine Benchmark               ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════╣")
	fmt.Printf("║ Endpoint:     %-47s ║\n", truncateStr(*endpoint, 47))
	fmt.Printf("║ Model:        %-47s ║\n", *model)
	fmt.Printf("║ Requests:     %-47d ║\n", *numRequests)
	fmt.Printf("║ Concurrency:  %-47d ║\n", *concurrency)
	fmt.Printf("║ Max Tokens:   %-47d ║\n", *maxTokens)
	fmt.Printf("║ Prompt:       %-47s ║\n", *promptLen)
	fmt.Printf("║ Streaming:    %-47v ║\n", *streaming)
	fmt.Println("╚══════════════════════════════════════════════════════════════╝")
	fmt.Println()

	// Warmup
	if *warmup > 0 {
		fmt.Printf("Warming up with %d requests...\n", *warmup)
		for i := 0; i < *warmup; i++ {
			if *streaming {
				_, _ = runStreamingRequest(*endpoint, *model, prompt, *maxTokens)
			} else {
				_, _ = runRequest(*endpoint, *model, prompt, *maxTokens)
			}
		}
		fmt.Println("Warmup complete.")
		fmt.Println()
	}

	// Run benchmark
	fmt.Printf("Running benchmark with %d requests (concurrency=%d)...\n", *numRequests, *concurrency)
	results := runBenchmark(*endpoint, *model, prompt, *maxTokens, *numRequests, *concurrency, *streaming)

	// Calculate statistics
	stats := calculateStats(results)

	// Output results
	if *outputJSON {
		outputJSONStats(stats)
	} else {
		printStats(stats, *streaming)
	}
}

func getPrompt(length string) string {
	switch length {
	case "short":
		return "The capital of France is"
	case "medium":
		return "Explain the concept of machine learning in simple terms. Machine learning is a subset of artificial intelligence that"
	case "long":
		return `Write a detailed explanation of how neural networks work, including the concepts of layers, weights, biases, activation functions, and backpropagation. Start with the basics and build up to more complex ideas. Neural networks are computational models inspired by`
	default:
		return "The capital of France is"
	}
}

func runRequest(endpoint, model, prompt string, maxTokens int) (BenchmarkResult, error) {
	result := BenchmarkResult{}

	reqBody := ChatCompletionRequest{
		Model:       model,
		Messages:    []Message{{Role: "user", Content: prompt}},
		MaxTokens:   maxTokens,
		Temperature: 0.7,
		Stream:      false,
	}

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		result.Error = err.Error()
		return result, err
	}

	startTime := time.Now()

	resp, err := http.Post(endpoint, "application/json", bytes.NewBuffer(jsonBody))
	if err != nil {
		result.Error = err.Error()
		result.TotalLatency = time.Since(startTime)
		return result, err
	}
	defer resp.Body.Close()

	// For non-streaming, TTFT ≈ total latency (we get all tokens at once)
	ttft := time.Since(startTime)

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		result.Error = err.Error()
		result.TotalLatency = time.Since(startTime)
		return result, err
	}

	totalLatency := time.Since(startTime)

	if resp.StatusCode != http.StatusOK {
		result.Error = fmt.Sprintf("HTTP %d: %s", resp.StatusCode, string(body))
		result.TotalLatency = totalLatency
		return result, fmt.Errorf(result.Error)
	}

	var chatResp ChatCompletionResponse
	if err := json.Unmarshal(body, &chatResp); err != nil {
		result.Error = err.Error()
		result.TotalLatency = totalLatency
		return result, err
	}

	// Count output tokens
	outputTokens := 0
	if chatResp.Usage.CompletionTokens > 0 {
		outputTokens = chatResp.Usage.CompletionTokens
	} else if len(chatResp.Choices) > 0 {
		words := len(bytes.Fields([]byte(chatResp.Choices[0].Message.Content)))
		outputTokens = int(float64(words) * 1.3)
		if outputTokens < 1 {
			outputTokens = 1
		}
	}

	promptTokens := int(float64(len(bytes.Fields([]byte(prompt)))) * 1.3)
	if promptTokens < 1 {
		promptTokens = 1
	}

	result.Success = true
	result.TTFT = ttft
	result.TotalLatency = totalLatency
	result.OutputTokens = outputTokens
	result.PromptTokens = promptTokens

	// TPOT = TotalLatency / OutputTokens (for non-streaming)
	if outputTokens > 0 {
		result.TPOT = totalLatency / time.Duration(outputTokens)
		// For non-streaming, ITL ≈ TPOT (approximation)
		result.ITLMean = result.TPOT
	}

	return result, nil
}

func runStreamingRequest(endpoint, model, prompt string, maxTokens int) (BenchmarkResult, error) {
	result := BenchmarkResult{}

	reqBody := ChatCompletionRequest{
		Model:       model,
		Messages:    []Message{{Role: "user", Content: prompt}},
		MaxTokens:   maxTokens,
		Temperature: 0.7,
		Stream:      true,
	}

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		result.Error = err.Error()
		return result, err
	}

	startTime := time.Now()

	resp, err := http.Post(endpoint, "application/json", bytes.NewBuffer(jsonBody))
	if err != nil {
		result.Error = err.Error()
		result.TotalLatency = time.Since(startTime)
		return result, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		result.Error = fmt.Sprintf("HTTP %d: %s", resp.StatusCode, string(body))
		result.TotalLatency = time.Since(startTime)
		return result, fmt.Errorf(result.Error)
	}

	// Read SSE stream and measure token timing
	reader := bufio.NewReader(resp.Body)
	var tokenTimes []time.Time
	var outputContent strings.Builder
	firstToken := true

	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				break
			}
			result.Error = err.Error()
			result.TotalLatency = time.Since(startTime)
			return result, err
		}

		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			break
		}

		var chunk StreamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}

		// Record token arrival time
		now := time.Now()
		if firstToken {
			result.TTFT = now.Sub(startTime)
			firstToken = false
		}
		tokenTimes = append(tokenTimes, now)

		// Accumulate content
		if len(chunk.Choices) > 0 && chunk.Choices[0].Delta.Content != "" {
			outputContent.WriteString(chunk.Choices[0].Delta.Content)
		}
	}

	result.TotalLatency = time.Since(startTime)

	// Calculate Inter-Token Latencies (ITL)
	if len(tokenTimes) > 1 {
		result.ITLs = make([]time.Duration, len(tokenTimes)-1)
		for i := 1; i < len(tokenTimes); i++ {
			result.ITLs[i-1] = tokenTimes[i].Sub(tokenTimes[i-1])
		}

		// Calculate mean ITL
		var totalITL time.Duration
		for _, itl := range result.ITLs {
			totalITL += itl
		}
		result.ITLMean = totalITL / time.Duration(len(result.ITLs))
	}

	// Count output tokens
	result.OutputTokens = len(tokenTimes)
	if result.OutputTokens == 0 {
		words := len(strings.Fields(outputContent.String()))
		result.OutputTokens = int(float64(words) * 1.3)
		if result.OutputTokens < 1 {
			result.OutputTokens = 1
		}
	}

	promptTokens := int(float64(len(strings.Fields(prompt))) * 1.3)
	if promptTokens < 1 {
		promptTokens = 1
	}
	result.PromptTokens = promptTokens

	// TPOT = (TotalLatency - TTFT) / OutputTokens
	if result.OutputTokens > 1 {
		generationTime := result.TotalLatency - result.TTFT
		result.TPOT = generationTime / time.Duration(result.OutputTokens-1)
	} else if result.OutputTokens == 1 {
		result.TPOT = result.TotalLatency
	}

	result.Success = true
	return result, nil
}

func runBenchmark(endpoint, model, prompt string, maxTokens, numRequests, concurrency int, streaming bool) []BenchmarkResult {
	results := make([]BenchmarkResult, numRequests)
	var wg sync.WaitGroup
	var completed int64

	semaphore := make(chan struct{}, concurrency)
	startTime := time.Now()

	// Progress ticker
	done := make(chan bool)
	go func() {
		ticker := time.NewTicker(time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				c := atomic.LoadInt64(&completed)
				elapsed := time.Since(startTime)
				rps := float64(c) / elapsed.Seconds()
				fmt.Printf("\rProgress: %d/%d (%.1f req/s)", c, numRequests, rps)
			case <-done:
				return
			}
		}
	}()

	for i := 0; i < numRequests; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			semaphore <- struct{}{}
			defer func() { <-semaphore }()

			var result BenchmarkResult
			if streaming {
				result, _ = runStreamingRequest(endpoint, model, prompt, maxTokens)
			} else {
				result, _ = runRequest(endpoint, model, prompt, maxTokens)
			}
			results[idx] = result
			atomic.AddInt64(&completed, 1)
		}(i)
	}

	wg.Wait()
	close(done)
	fmt.Printf("\rProgress: %d/%d - Complete!                    \n\n", numRequests, numRequests)

	return results
}

func calculateStats(results []BenchmarkResult) BenchmarkStats {
	stats := BenchmarkStats{
		TotalRequests: len(results),
	}

	var ttfts, tpots, latencies, allITLs []float64

	for _, r := range results {
		if r.Success {
			stats.SuccessfulReqs++
			stats.TotalOutputTokens += r.OutputTokens
			stats.TotalPromptTokens += r.PromptTokens

			ttfts = append(ttfts, float64(r.TTFT.Milliseconds()))
			if r.OutputTokens > 0 {
				tpots = append(tpots, float64(r.TPOT.Milliseconds()))
			}
			latencies = append(latencies, float64(r.TotalLatency.Milliseconds()))

			// Collect all ITLs
			for _, itl := range r.ITLs {
				allITLs = append(allITLs, float64(itl.Milliseconds()))
			}
			// If no individual ITLs but we have ITLMean (non-streaming)
			if len(r.ITLs) == 0 && r.ITLMean > 0 {
				allITLs = append(allITLs, float64(r.ITLMean.Milliseconds()))
			}
		} else {
			stats.FailedReqs++
			if r.Error != "" {
				stats.Errors = append(stats.Errors, r.Error)
			}
		}
	}

	// Calculate TTFT percentiles
	if len(ttfts) > 0 {
		sort.Float64s(ttfts)
		stats.TTFTMin = ttfts[0]
		stats.TTFTMax = ttfts[len(ttfts)-1]
		stats.TTFTMean = mean(ttfts)
		stats.TTFTP50 = percentile(ttfts, 50)
		stats.TTFTP90 = percentile(ttfts, 90)
		stats.TTFTP95 = percentile(ttfts, 95)
		stats.TTFTP99 = percentile(ttfts, 99)
	}

	// Calculate ITL percentiles
	if len(allITLs) > 0 {
		sort.Float64s(allITLs)
		stats.ITLMin = allITLs[0]
		stats.ITLMax = allITLs[len(allITLs)-1]
		stats.ITLMean = mean(allITLs)
		stats.ITLP50 = percentile(allITLs, 50)
		stats.ITLP90 = percentile(allITLs, 90)
		stats.ITLP95 = percentile(allITLs, 95)
		stats.ITLP99 = percentile(allITLs, 99)
	}

	// Calculate TPOT percentiles
	if len(tpots) > 0 {
		sort.Float64s(tpots)
		stats.TPOTMin = tpots[0]
		stats.TPOTMax = tpots[len(tpots)-1]
		stats.TPOTMean = mean(tpots)
		stats.TPOTP50 = percentile(tpots, 50)
		stats.TPOTP90 = percentile(tpots, 90)
		stats.TPOTP95 = percentile(tpots, 95)
		stats.TPOTP99 = percentile(tpots, 99)
	}

	// Calculate latency percentiles
	if len(latencies) > 0 {
		sort.Float64s(latencies)
		stats.LatencyMin = latencies[0]
		stats.LatencyMax = latencies[len(latencies)-1]
		stats.LatencyMean = mean(latencies)
		stats.LatencyP50 = percentile(latencies, 50)
		stats.LatencyP90 = percentile(latencies, 90)
		stats.LatencyP95 = percentile(latencies, 95)
		stats.LatencyP99 = percentile(latencies, 99)

		stats.TotalDuration = time.Duration(stats.LatencyMax) * time.Millisecond

		if stats.TotalDuration > 0 {
			stats.RequestsPerSec = float64(stats.SuccessfulReqs) / (float64(stats.LatencyMax) / 1000.0)
			stats.OutputTokensPerSec = float64(stats.TotalOutputTokens) / (float64(stats.LatencyMax) / 1000.0)
		}
	}

	return stats
}

func mean(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	return sum / float64(len(values))
}

func percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	idx := int(math.Ceil(float64(len(sorted)) * p / 100.0)) - 1
	if idx < 0 {
		idx = 0
	}
	if idx >= len(sorted) {
		idx = len(sorted) - 1
	}
	return sorted[idx]
}

func printStats(stats BenchmarkStats, streaming bool) {
	fmt.Println("╔══════════════════════════════════════════════════════════════╗")
	fmt.Println("║                     Benchmark Results                        ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════╣")
	fmt.Printf("║ Total Requests:     %-42d ║\n", stats.TotalRequests)
	fmt.Printf("║ Successful:         %-42d ║\n", stats.SuccessfulReqs)
	fmt.Printf("║ Failed:             %-42d ║\n", stats.FailedReqs)
	fmt.Printf("║ Total Output Tokens:%-42d ║\n", stats.TotalOutputTokens)
	fmt.Println("╠══════════════════════════════════════════════════════════════╣")
	fmt.Println("║                    TTFT (Time To First Token)                ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════╣")
	fmt.Printf("║ Min:     %10.2f ms                                       ║\n", stats.TTFTMin)
	fmt.Printf("║ Max:     %10.2f ms                                       ║\n", stats.TTFTMax)
	fmt.Printf("║ Mean:    %10.2f ms                                       ║\n", stats.TTFTMean)
	fmt.Printf("║ P50:     %10.2f ms                                       ║\n", stats.TTFTP50)
	fmt.Printf("║ P90:     %10.2f ms                                       ║\n", stats.TTFTP90)
	fmt.Printf("║ P95:     %10.2f ms                                       ║\n", stats.TTFTP95)
	fmt.Printf("║ P99:     %10.2f ms                                       ║\n", stats.TTFTP99)
	fmt.Println("╠══════════════════════════════════════════════════════════════╣")
	fmt.Println("║                   ITL (Inter-Token Latency)                  ║")
	if !streaming {
		fmt.Println("║           (Approximated - use --stream for accuracy)        ║")
	}
	fmt.Println("╠══════════════════════════════════════════════════════════════╣")
	fmt.Printf("║ Min:     %10.2f ms                                       ║\n", stats.ITLMin)
	fmt.Printf("║ Max:     %10.2f ms                                       ║\n", stats.ITLMax)
	fmt.Printf("║ Mean:    %10.2f ms                                       ║\n", stats.ITLMean)
	fmt.Printf("║ P50:     %10.2f ms                                       ║\n", stats.ITLP50)
	fmt.Printf("║ P90:     %10.2f ms                                       ║\n", stats.ITLP90)
	fmt.Printf("║ P95:     %10.2f ms                                       ║\n", stats.ITLP95)
	fmt.Printf("║ P99:     %10.2f ms                                       ║\n", stats.ITLP99)
	fmt.Println("╠══════════════════════════════════════════════════════════════╣")
	fmt.Println("║                 TPOT (Time Per Output Token)                 ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════╣")
	fmt.Printf("║ Min:     %10.2f ms/token                                 ║\n", stats.TPOTMin)
	fmt.Printf("║ Max:     %10.2f ms/token                                 ║\n", stats.TPOTMax)
	fmt.Printf("║ Mean:    %10.2f ms/token                                 ║\n", stats.TPOTMean)
	fmt.Printf("║ P50:     %10.2f ms/token                                 ║\n", stats.TPOTP50)
	fmt.Printf("║ P90:     %10.2f ms/token                                 ║\n", stats.TPOTP90)
	fmt.Printf("║ P95:     %10.2f ms/token                                 ║\n", stats.TPOTP95)
	fmt.Printf("║ P99:     %10.2f ms/token                                 ║\n", stats.TPOTP99)
	fmt.Println("╠══════════════════════════════════════════════════════════════╣")
	fmt.Println("║                    End-to-End Latency                        ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════╣")
	fmt.Printf("║ Min:     %10.2f ms                                       ║\n", stats.LatencyMin)
	fmt.Printf("║ Max:     %10.2f ms                                       ║\n", stats.LatencyMax)
	fmt.Printf("║ Mean:    %10.2f ms                                       ║\n", stats.LatencyMean)
	fmt.Printf("║ P50:     %10.2f ms                                       ║\n", stats.LatencyP50)
	fmt.Printf("║ P90:     %10.2f ms                                       ║\n", stats.LatencyP90)
	fmt.Printf("║ P95:     %10.2f ms                                       ║\n", stats.LatencyP95)
	fmt.Printf("║ P99:     %10.2f ms                                       ║\n", stats.LatencyP99)
	fmt.Println("╠══════════════════════════════════════════════════════════════╣")
	fmt.Println("║                       Throughput                             ║")
	fmt.Println("╠══════════════════════════════════════════════════════════════╣")
	fmt.Printf("║ Requests/sec:       %-42.2f ║\n", stats.RequestsPerSec)
	fmt.Printf("║ Output Tokens/sec:  %-42.2f ║\n", stats.OutputTokensPerSec)
	tps := 1000.0 / stats.ITLMean
	if stats.ITLMean > 0 {
		fmt.Printf("║ Generation TPS:     %-42.2f ║\n", tps)
	}
	fmt.Println("╚══════════════════════════════════════════════════════════════╝")

	if len(stats.Errors) > 0 {
		fmt.Println("\nErrors:")
		uniqueErrors := make(map[string]int)
		for _, e := range stats.Errors {
			uniqueErrors[e]++
		}
		for err, count := range uniqueErrors {
			fmt.Printf("  [%d] %s\n", count, truncateStr(err, 70))
		}
	}
}

func truncateStr(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen-3] + "..."
}

func outputJSONStats(stats BenchmarkStats) {
	tps := 0.0
	if stats.ITLMean > 0 {
		tps = 1000.0 / stats.ITLMean
	}

	output := map[string]interface{}{
		"summary": map[string]interface{}{
			"total_requests":      stats.TotalRequests,
			"successful_requests": stats.SuccessfulReqs,
			"failed_requests":     stats.FailedReqs,
			"total_output_tokens": stats.TotalOutputTokens,
			"total_prompt_tokens": stats.TotalPromptTokens,
		},
		"ttft_ms": map[string]float64{
			"min":  stats.TTFTMin,
			"max":  stats.TTFTMax,
			"mean": stats.TTFTMean,
			"p50":  stats.TTFTP50,
			"p90":  stats.TTFTP90,
			"p95":  stats.TTFTP95,
			"p99":  stats.TTFTP99,
		},
		"itl_ms": map[string]float64{
			"min":  stats.ITLMin,
			"max":  stats.ITLMax,
			"mean": stats.ITLMean,
			"p50":  stats.ITLP50,
			"p90":  stats.ITLP90,
			"p95":  stats.ITLP95,
			"p99":  stats.ITLP99,
		},
		"tpot_ms": map[string]float64{
			"min":  stats.TPOTMin,
			"max":  stats.TPOTMax,
			"mean": stats.TPOTMean,
			"p50":  stats.TPOTP50,
			"p90":  stats.TPOTP90,
			"p95":  stats.TPOTP95,
			"p99":  stats.TPOTP99,
		},
		"latency_ms": map[string]float64{
			"min":  stats.LatencyMin,
			"max":  stats.LatencyMax,
			"mean": stats.LatencyMean,
			"p50":  stats.LatencyP50,
			"p90":  stats.LatencyP90,
			"p95":  stats.LatencyP95,
			"p99":  stats.LatencyP99,
		},
		"throughput": map[string]float64{
			"requests_per_sec":      stats.RequestsPerSec,
			"output_tokens_per_sec": stats.OutputTokensPerSec,
			"generation_tps":        tps,
		},
	}

	jsonBytes, _ := json.MarshalIndent(output, "", "  ")
	fmt.Println(string(jsonBytes))
}
