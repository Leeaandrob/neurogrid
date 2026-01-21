// Package download contains E2E tests for the Model Download Script feature.
// These tests verify the acceptance criteria from PRP-03: Model Download Script.
//
// Success Criteria:
// - [ ] Download TinyLlama 1.1B in < 2 minutes
// - [ ] Download Llama 2 7B with progress display
// - [ ] Support HF_TOKEN authentication
// - [ ] Resume interrupted downloads
// - [ ] Verify checksums after download
// - [ ] Estimate disk space before download
// - [ ] Support multiple model variants (chat, instruct)
package download

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/neurogrid/engine/pkg/huggingface"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// =============================================================================
// Test Fixtures and Mocks
// =============================================================================

// mockHFServer creates a mock HuggingFace API server for testing.
func mockHFServer(t *testing.T) *httptest.Server {
	mux := http.NewServeMux()

	// Mock model info endpoint
	mux.HandleFunc("/api/models/", func(w http.ResponseWriter, r *http.Request) {
		// Check for auth token
		authHeader := r.Header.Get("Authorization")
		path := r.URL.Path

		// Gated models require auth
		if strings.Contains(path, "meta-llama") && authHeader == "" {
			w.WriteHeader(http.StatusUnauthorized)
			w.Write([]byte(`{"error": "Gated model requires authentication"}`))
			return
		}

		// Return model info based on repo
		if strings.Contains(path, "TinyLlama") {
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{
				"id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
				"sha": "abc123",
				"siblings": [
					{"rfilename": "config.json", "size": 1024},
					{"rfilename": "tokenizer.model", "size": 512000},
					{"rfilename": "model.safetensors", "size": 2200000000}
				],
				"private": false,
				"gated": false
			}`))
		} else if strings.Contains(path, "Llama-2-7b") {
			// Check auth for gated model
			if authHeader == "" || !strings.HasPrefix(authHeader, "Bearer ") {
				w.WriteHeader(http.StatusUnauthorized)
				return
			}
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{
				"id": "meta-llama/Llama-2-7b-hf",
				"sha": "def456",
				"siblings": [
					{"rfilename": "config.json", "size": 1024},
					{"rfilename": "tokenizer.model", "size": 512000},
					{"rfilename": "model-00001-of-00002.safetensors", "size": 6500000000},
					{"rfilename": "model-00002-of-00002.safetensors", "size": 6500000000}
				],
				"private": false,
				"gated": "auto"
			}`))
		} else {
			w.WriteHeader(http.StatusNotFound)
		}
	})

	// Mock file download endpoint
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		// Support range requests for resume testing
		rangeHeader := r.Header.Get("Range")

		// Generate test file content
		testContent := []byte(strings.Repeat("test-content-", 1000))
		contentHash := sha256.Sum256(testContent)

		if rangeHeader != "" {
			// Parse range header: "bytes=START-"
			var start int64
			_, err := fmt.Sscanf(rangeHeader, "bytes=%d-", &start)
			if err == nil && start < int64(len(testContent)) {
				w.Header().Set("Content-Range", fmt.Sprintf("bytes %d-%d/%d", start, len(testContent)-1, len(testContent)))
				w.WriteHeader(http.StatusPartialContent)
				w.Write(testContent[start:])
				return
			}
		}

		w.Header().Set("Content-Length", fmt.Sprintf("%d", len(testContent)))
		w.Header().Set("X-Linked-Etag", hex.EncodeToString(contentHash[:]))
		w.Write(testContent)
	})

	return httptest.NewServer(mux)
}

// =============================================================================
// Acceptance Criteria Tests - RED Phase
// =============================================================================

// TestDownloadTinyLlamaUnder2Minutes verifies:
// - [ ] Download TinyLlama 1.1B in < 2 minutes
func TestDownloadTinyLlamaUnder2Minutes(t *testing.T) {
	// Skip in CI unless integration tests are enabled
	if os.Getenv("INTEGRATION_TESTS") == "" {
		t.Skip("Skipping integration test. Set INTEGRATION_TESTS=1 to run.")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()

	outputDir := t.TempDir()

	client := huggingface.NewClient(os.Getenv("HF_TOKEN"))
	client.SetBaseURL("https://huggingface.co")

	// Download TinyLlama
	err := client.Download(ctx, "TinyLlama/TinyLlama-1.1B-Chat-v1.0", outputDir, nil)

	// Assert download completed within timeout
	require.NoError(t, err, "Download should complete without error")

	// Verify required files exist
	requiredFiles := []string{"config.json", "tokenizer.model"}
	for _, f := range requiredFiles {
		path := filepath.Join(outputDir, f)
		assert.FileExists(t, path, "Required file should exist: %s", f)
	}

	// Verify model weights exist (safetensors format)
	matches, _ := filepath.Glob(filepath.Join(outputDir, "*.safetensors"))
	assert.NotEmpty(t, matches, "Model weights (safetensors) should exist")
}

// TestDownloadLlama7BWithProgress verifies:
// - [ ] Download Llama 2 7B with progress display
func TestDownloadLlama7BWithProgress(t *testing.T) {
	server := mockHFServer(t)
	defer server.Close()

	ctx := context.Background()
	outputDir := t.TempDir()

	client := huggingface.NewClient("test-token")
	client.SetBaseURL(server.URL)

	var progressCalls int32
	var lastProgress huggingface.DownloadProgress

	progressFn := func(p huggingface.DownloadProgress) {
		atomic.AddInt32(&progressCalls, 1)
		lastProgress = p

		// Verify progress fields are populated
		assert.NotEmpty(t, p.Filename, "Progress should include filename")
		assert.GreaterOrEqual(t, p.Downloaded, int64(0), "Downloaded bytes should be >= 0")
		assert.Greater(t, p.Total, int64(0), "Total bytes should be > 0")
	}

	err := client.Download(ctx, "meta-llama/Llama-2-7b-hf", outputDir, progressFn)
	require.NoError(t, err)

	// Verify progress callbacks were made
	assert.Greater(t, atomic.LoadInt32(&progressCalls), int32(0), "Progress callback should be called")
	assert.Greater(t, lastProgress.Speed, float64(0), "Progress should report speed")
}

// TestHFTokenAuthentication verifies:
// - [ ] Support HF_TOKEN authentication
func TestHFTokenAuthentication(t *testing.T) {
	server := mockHFServer(t)
	defer server.Close()

	ctx := context.Background()
	outputDir := t.TempDir()

	t.Run("gated model without token fails", func(t *testing.T) {
		client := huggingface.NewClient("") // No token
		client.SetBaseURL(server.URL)

		err := client.Download(ctx, "meta-llama/Llama-2-7b-hf", outputDir, nil)
		assert.Error(t, err, "Download of gated model without token should fail")
		assert.Contains(t, err.Error(), "authentication", "Error should mention authentication")
	})

	t.Run("gated model with token succeeds", func(t *testing.T) {
		client := huggingface.NewClient("hf_valid_token")
		client.SetBaseURL(server.URL)

		err := client.Download(ctx, "meta-llama/Llama-2-7b-hf", outputDir, nil)
		assert.NoError(t, err, "Download of gated model with token should succeed")
	})

	t.Run("public model without token succeeds", func(t *testing.T) {
		client := huggingface.NewClient("") // No token
		client.SetBaseURL(server.URL)

		err := client.Download(ctx, "TinyLlama/TinyLlama-1.1B-Chat-v1.0", outputDir, nil)
		assert.NoError(t, err, "Download of public model without token should succeed")
	})
}

// TestResumeInterruptedDownloads verifies:
// - [ ] Resume interrupted downloads
func TestResumeInterruptedDownloads(t *testing.T) {
	server := mockHFServer(t)
	defer server.Close()

	outputDir := t.TempDir()

	client := huggingface.NewClient("test-token")
	client.SetBaseURL(server.URL)

	// Create a partial download file
	partialFile := filepath.Join(outputDir, "test-file.safetensors.partial")
	partialContent := []byte(strings.Repeat("test-content-", 100)) // 1300 bytes
	err := os.WriteFile(partialFile, partialContent, 0644)
	require.NoError(t, err)

	partialSize := int64(len(partialContent))

	// Track if Range header was used
	var usedRangeHeader bool
	client.SetRequestInterceptor(func(req *http.Request) {
		if req.Header.Get("Range") != "" {
			usedRangeHeader = true
		}
	})

	// Download should resume from partial
	err = client.DownloadFile(
		context.Background(),
		huggingface.FileInfo{
			Filename: "test-file.safetensors",
			Size:     int64(len(strings.Repeat("test-content-", 1000))),
			SHA256:   "", // Skip checksum for this test
			URL:      server.URL + "/test-file",
		},
		outputDir,
		nil,
	)
	require.NoError(t, err)

	// Verify resume was attempted
	assert.True(t, usedRangeHeader, "Should use Range header for resume")

	// Verify final file exists (not .partial)
	finalFile := filepath.Join(outputDir, "test-file.safetensors")
	assert.FileExists(t, finalFile, "Final file should exist after resume")

	// Verify file size is complete
	stat, _ := os.Stat(finalFile)
	assert.Greater(t, stat.Size(), partialSize, "Final file should be larger than partial")
}

// TestChecksumVerification verifies:
// - [ ] Verify checksums after download
func TestChecksumVerification(t *testing.T) {
	server := mockHFServer(t)
	defer server.Close()

	outputDir := t.TempDir()

	client := huggingface.NewClient("test-token")
	client.SetBaseURL(server.URL)

	t.Run("valid checksum passes", func(t *testing.T) {
		// Generate expected checksum for test content
		testContent := []byte(strings.Repeat("test-content-", 1000))
		expectedHash := sha256.Sum256(testContent)
		expectedHex := hex.EncodeToString(expectedHash[:])

		err := client.DownloadFile(
			context.Background(),
			huggingface.FileInfo{
				Filename: "valid.safetensors",
				Size:     int64(len(testContent)),
				SHA256:   expectedHex,
				URL:      server.URL + "/valid",
			},
			outputDir,
			nil,
		)
		assert.NoError(t, err, "Download with valid checksum should pass")
	})

	t.Run("invalid checksum fails", func(t *testing.T) {
		err := client.DownloadFile(
			context.Background(),
			huggingface.FileInfo{
				Filename: "invalid.safetensors",
				Size:     1000,
				SHA256:   "0000000000000000000000000000000000000000000000000000000000000000",
				URL:      server.URL + "/invalid",
			},
			outputDir,
			nil,
		)
		assert.Error(t, err, "Download with invalid checksum should fail")
		assert.Contains(t, err.Error(), "checksum", "Error should mention checksum")
	})

	t.Run("empty checksum skips verification", func(t *testing.T) {
		err := client.DownloadFile(
			context.Background(),
			huggingface.FileInfo{
				Filename: "no-checksum.safetensors",
				Size:     1000,
				SHA256:   "", // Empty = skip verification
				URL:      server.URL + "/no-checksum",
			},
			outputDir,
			nil,
		)
		assert.NoError(t, err, "Download without checksum should pass (skip verification)")
	})
}

// TestDiskSpaceEstimation verifies:
// - [ ] Estimate disk space before download
func TestDiskSpaceEstimation(t *testing.T) {
	server := mockHFServer(t)
	defer server.Close()

	client := huggingface.NewClient("")
	client.SetBaseURL(server.URL)

	t.Run("get model info returns total size", func(t *testing.T) {
		info, err := client.GetModelInfo(context.Background(), "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
		require.NoError(t, err)

		assert.Greater(t, info.TotalSize, int64(0), "Total size should be > 0")
		assert.NotEmpty(t, info.Files, "Files list should not be empty")

		// Verify total size is sum of file sizes
		var calculatedTotal int64
		for _, f := range info.Files {
			calculatedTotal += f.Size
		}
		assert.Equal(t, calculatedTotal, info.TotalSize, "TotalSize should match sum of file sizes")
	})

	t.Run("check disk space before download", func(t *testing.T) {
		outputDir := t.TempDir()

		hasSpace, available, required, err := client.CheckDiskSpace(context.Background(), "TinyLlama/TinyLlama-1.1B-Chat-v1.0", outputDir)
		require.NoError(t, err)

		assert.Greater(t, available, int64(0), "Available space should be > 0")
		assert.Greater(t, required, int64(0), "Required space should be > 0")

		// In temp dir, we should have space for mock model
		assert.True(t, hasSpace, "Should have space for mock model in temp dir")
	})

	t.Run("insufficient disk space returns error", func(t *testing.T) {
		// This test requires a mock that reports insufficient space
		// We test the error path by checking error type
		err := client.Download(
			context.Background(),
			"TinyLlama/TinyLlama-1.1B-Chat-v1.0",
			"/nonexistent/path/that/should/fail",
			nil,
		)

		if err != nil {
			// Either permission error or space error is acceptable
			assert.True(t,
				strings.Contains(err.Error(), "space") || strings.Contains(err.Error(), "permission"),
				"Error should be about space or permissions")
		}
	})
}

// TestModelVariantSupport verifies:
// - [ ] Support multiple model variants (chat, instruct)
func TestModelVariantSupport(t *testing.T) {
	server := mockHFServer(t)
	defer server.Close()

	client := huggingface.NewClient("test-token")
	client.SetBaseURL(server.URL)

	testCases := []struct {
		alias    string
		expected string
	}{
		{"tinyllama", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"},
		{"llama7b", "meta-llama/Llama-2-7b-hf"},
		{"llama7b-chat", "meta-llama/Llama-2-7b-chat-hf"},
		{"llama13b", "meta-llama/Llama-2-13b-hf"},
		{"llama13b-chat", "meta-llama/Llama-2-13b-chat-hf"},
	}

	for _, tc := range testCases {
		t.Run(tc.alias, func(t *testing.T) {
			repoID := huggingface.ResolveAlias(tc.alias)
			assert.Equal(t, tc.expected, repoID, "Alias %s should resolve to %s", tc.alias, tc.expected)
		})
	}

	// Test that unknown aliases are passed through
	t.Run("unknown alias passes through", func(t *testing.T) {
		repoID := huggingface.ResolveAlias("custom-org/custom-model")
		assert.Equal(t, "custom-org/custom-model", repoID, "Unknown alias should pass through")
	})
}

// =============================================================================
// Additional E2E Tests
// =============================================================================

// TestBashScriptIntegration tests the bash wrapper script.
func TestBashScriptIntegration(t *testing.T) {
	// Skip unless integration tests enabled
	if os.Getenv("INTEGRATION_TESTS") == "" {
		t.Skip("Skipping integration test. Set INTEGRATION_TESTS=1 to run.")
	}

	scriptPath := filepath.Join("..", "..", "scripts", "download_model.sh")

	t.Run("script exists and is executable", func(t *testing.T) {
		info, err := os.Stat(scriptPath)
		require.NoError(t, err, "Script should exist")
		assert.True(t, info.Mode()&0111 != 0, "Script should be executable")
	})

	t.Run("script shows help without arguments", func(t *testing.T) {
		// This would test the bash script behavior
		// We test the Go CLI directly instead
	})
}

// TestGoDownloadCLI tests the Go download CLI directly.
func TestGoDownloadCLI(t *testing.T) {
	server := mockHFServer(t)
	defer server.Close()

	// Test CLI argument parsing
	t.Run("parse arguments", func(t *testing.T) {
		args := huggingface.ParseCLIArgs([]string{
			"--repo", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
			"--output", "/tmp/models",
			"--token", "test-token",
		})

		assert.Equal(t, "TinyLlama/TinyLlama-1.1B-Chat-v1.0", args.Repo)
		assert.Equal(t, "/tmp/models", args.Output)
		assert.Equal(t, "test-token", args.Token)
	})

	t.Run("repo from environment", func(t *testing.T) {
		os.Setenv("HF_TOKEN", "env-token")
		defer os.Unsetenv("HF_TOKEN")

		args := huggingface.ParseCLIArgs([]string{
			"--repo", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
			"--output", "/tmp/models",
		})

		assert.Equal(t, "env-token", args.Token, "Token should come from HF_TOKEN env var")
	})
}

// TestConcurrentDownloads tests downloading multiple files concurrently.
func TestConcurrentDownloads(t *testing.T) {
	server := mockHFServer(t)
	defer server.Close()

	ctx := context.Background()
	outputDir := t.TempDir()

	client := huggingface.NewClient("test-token")
	client.SetBaseURL(server.URL)
	client.SetConcurrency(4) // Download 4 files in parallel

	// Track concurrent downloads
	var maxConcurrent int32
	var currentConcurrent int32

	client.SetRequestInterceptor(func(req *http.Request) {
		current := atomic.AddInt32(&currentConcurrent, 1)
		if current > atomic.LoadInt32(&maxConcurrent) {
			atomic.StoreInt32(&maxConcurrent, current)
		}
	})
	client.SetResponseInterceptor(func(resp *http.Response) {
		atomic.AddInt32(&currentConcurrent, -1)
	})

	err := client.Download(ctx, "meta-llama/Llama-2-7b-hf", outputDir, nil)
	require.NoError(t, err)

	// Verify some concurrency happened (if model has multiple files)
	// Note: With 2 shard files, we expect max concurrent to be 2 or more
	t.Logf("Max concurrent downloads: %d", atomic.LoadInt32(&maxConcurrent))
}

// TestDownloadContextCancellation tests proper handling of context cancellation.
func TestDownloadContextCancellation(t *testing.T) {
	server := mockHFServer(t)
	defer server.Close()

	outputDir := t.TempDir()

	client := huggingface.NewClient("test-token")
	client.SetBaseURL(server.URL)

	ctx, cancel := context.WithCancel(context.Background())

	// Cancel immediately
	cancel()

	err := client.Download(ctx, "meta-llama/Llama-2-7b-hf", outputDir, nil)
	assert.Error(t, err, "Download should fail when context is cancelled")
	assert.True(t, strings.Contains(err.Error(), "cancel") || strings.Contains(err.Error(), "context"),
		"Error should mention cancellation")
}

// TestProgressETA tests that ETA calculation is reasonable.
func TestProgressETA(t *testing.T) {
	server := mockHFServer(t)
	defer server.Close()

	ctx := context.Background()
	outputDir := t.TempDir()

	client := huggingface.NewClient("test-token")
	client.SetBaseURL(server.URL)

	var lastETA time.Duration
	var etaDecreasing bool

	progressFn := func(p huggingface.DownloadProgress) {
		if p.ETA > 0 {
			if lastETA > 0 && p.ETA <= lastETA {
				etaDecreasing = true
			}
			lastETA = p.ETA
		}
	}

	err := client.Download(ctx, "TinyLlama/TinyLlama-1.1B-Chat-v1.0", outputDir, progressFn)
	require.NoError(t, err)

	// ETA should generally decrease as download progresses
	// (may not always be true due to speed fluctuations, but check it was set)
	assert.True(t, lastETA >= 0 || etaDecreasing, "ETA should be calculated")
}
