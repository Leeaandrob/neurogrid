package huggingface

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"
)

// Client is a HuggingFace Hub API client.
type Client struct {
	token               string
	baseURL             string
	httpClient          *http.Client
	concurrency         int
	requestInterceptor  func(*http.Request)
	responseInterceptor func(*http.Response)
}

// NewClient creates a new HuggingFace Hub client.
func NewClient(token string) *Client {
	return &Client{
		token:       token,
		baseURL:     "https://huggingface.co",
		httpClient:  &http.Client{Timeout: 30 * time.Minute},
		concurrency: 1,
	}
}

// SetBaseURL sets the base URL for API requests.
// This is useful for testing with a mock server.
func (c *Client) SetBaseURL(url string) {
	c.baseURL = url
}

// SetConcurrency sets the number of concurrent downloads.
func (c *Client) SetConcurrency(n int) {
	c.concurrency = n
}

// SetRequestInterceptor sets a function to be called before each request.
// This is useful for testing.
func (c *Client) SetRequestInterceptor(fn func(*http.Request)) {
	c.requestInterceptor = fn
}

// SetResponseInterceptor sets a function to be called after each response.
// This is useful for testing.
func (c *Client) SetResponseInterceptor(fn func(*http.Response)) {
	c.responseInterceptor = fn
}

// hfModelResponse represents the response from the HuggingFace API.
type hfModelResponse struct {
	ID       string `json:"id"`
	SHA      string `json:"sha"`
	Siblings []struct {
		RFilename string `json:"rfilename"`
		Size      int64  `json:"size"`
	} `json:"siblings"`
	Private bool        `json:"private"`
	Gated   interface{} `json:"gated"` // Can be bool or string
}

// GetModelInfo retrieves information about a model from the HuggingFace Hub.
func (c *Client) GetModelInfo(ctx context.Context, repoID string) (*ModelInfo, error) {
	url := fmt.Sprintf("%s/api/models/%s", c.baseURL, repoID)

	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	if c.token != "" {
		req.Header.Set("Authorization", "Bearer "+c.token)
	}

	if c.requestInterceptor != nil {
		c.requestInterceptor(req)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch model info: %w", err)
	}
	defer resp.Body.Close()

	if c.responseInterceptor != nil {
		c.responseInterceptor(resp)
	}

	if resp.StatusCode == http.StatusUnauthorized {
		return nil, fmt.Errorf("authentication required for gated model: %s", repoID)
	}

	if resp.StatusCode == http.StatusNotFound {
		return nil, fmt.Errorf("model not found: %s", repoID)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}

	var hfResp hfModelResponse
	if err := json.NewDecoder(resp.Body).Decode(&hfResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	// Convert to ModelInfo
	info := &ModelInfo{
		RepoID:   hfResp.ID,
		Revision: hfResp.SHA,
		Files:    make([]FileInfo, 0, len(hfResp.Siblings)),
	}

	// Check if auth is required
	switch v := hfResp.Gated.(type) {
	case bool:
		info.RequiresAuth = v
	case string:
		info.RequiresAuth = v != "" && v != "false"
	}

	// Build file list with download URLs
	for _, sibling := range hfResp.Siblings {
		fileURL := fmt.Sprintf("%s/%s/resolve/main/%s", c.baseURL, repoID, sibling.RFilename)
		info.Files = append(info.Files, FileInfo{
			Filename: sibling.RFilename,
			Size:     sibling.Size,
			URL:      fileURL,
		})
		info.TotalSize += sibling.Size
	}

	return info, nil
}

// Download downloads all files from a model repository.
func (c *Client) Download(ctx context.Context, repoID, outputDir string, progress ProgressFunc) error {
	// Check context first
	select {
	case <-ctx.Done():
		return fmt.Errorf("context cancelled: %w", ctx.Err())
	default:
	}

	// Get model info
	info, err := c.GetModelInfo(ctx, repoID)
	if err != nil {
		return err
	}

	// Create output directory
	if err := os.MkdirAll(outputDir, 0755); err != nil {
		return fmt.Errorf("failed to create output directory: %w", err)
	}

	// Check disk space
	hasSpace, available, required, err := c.checkDiskSpaceInternal(info, outputDir)
	if err != nil {
		return fmt.Errorf("failed to check disk space: %w", err)
	}
	if !hasSpace {
		return fmt.Errorf("insufficient disk space: need %d bytes, have %d bytes", required, available)
	}

	// Download files
	if c.concurrency <= 1 {
		// Sequential download
		for _, file := range info.Files {
			select {
			case <-ctx.Done():
				return fmt.Errorf("context cancelled: %w", ctx.Err())
			default:
			}

			if err := c.DownloadFile(ctx, file, outputDir, progress); err != nil {
				return fmt.Errorf("failed to download %s: %w", file.Filename, err)
			}
		}
	} else {
		// Concurrent download
		errChan := make(chan error, len(info.Files))
		semaphore := make(chan struct{}, c.concurrency)
		var wg sync.WaitGroup

		for _, file := range info.Files {
			wg.Add(1)
			go func(f FileInfo) {
				defer wg.Done()

				select {
				case <-ctx.Done():
					errChan <- fmt.Errorf("context cancelled: %w", ctx.Err())
					return
				case semaphore <- struct{}{}:
					defer func() { <-semaphore }()
				}

				if err := c.DownloadFile(ctx, f, outputDir, progress); err != nil {
					errChan <- fmt.Errorf("failed to download %s: %w", f.Filename, err)
				}
			}(file)
		}

		wg.Wait()
		close(errChan)

		// Check for errors
		for err := range errChan {
			if err != nil {
				return err
			}
		}
	}

	return nil
}

// DownloadFile downloads a single file with resume support.
func (c *Client) DownloadFile(ctx context.Context, file FileInfo, outputDir string, progress ProgressFunc) error {
	outPath := filepath.Join(outputDir, file.Filename)
	partialPath := outPath + ".partial"

	// Check for partial download to resume
	var startByte int64
	if stat, err := os.Stat(partialPath); err == nil {
		startByte = stat.Size()
	}

	// Create request
	req, err := http.NewRequestWithContext(ctx, "GET", file.URL, nil)
	if err != nil {
		return fmt.Errorf("failed to create request: %w", err)
	}

	// Add auth header
	if c.token != "" {
		req.Header.Set("Authorization", "Bearer "+c.token)
	}

	// Add range header for resume
	if startByte > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", startByte))
	}

	if c.requestInterceptor != nil {
		c.requestInterceptor(req)
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("failed to download file: %w", err)
	}
	defer resp.Body.Close()

	if c.responseInterceptor != nil {
		c.responseInterceptor(resp)
	}

	// Check response status
	if resp.StatusCode == http.StatusUnauthorized {
		return fmt.Errorf("authentication required for file: %s", file.Filename)
	}

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusPartialContent {
		return fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}

	// Open file for writing (append if resuming)
	flags := os.O_CREATE | os.O_WRONLY
	if startByte > 0 && resp.StatusCode == http.StatusPartialContent {
		flags |= os.O_APPEND
	} else {
		flags |= os.O_TRUNC
		startByte = 0 // Reset if not resuming
	}

	out, err := os.OpenFile(partialPath, flags, 0644)
	if err != nil {
		return fmt.Errorf("failed to open file: %w", err)
	}
	defer out.Close()

	// Create hash for checksum
	hash := sha256.New()

	// If resuming, we need to read existing content for hash
	if startByte > 0 {
		existingFile, err := os.Open(partialPath)
		if err == nil {
			io.Copy(hash, existingFile)
			existingFile.Close()
		}
	}

	// Copy with progress tracking
	buf := make([]byte, 32*1024) // 32KB buffer
	downloaded := startByte
	start := time.Now()

	for {
		select {
		case <-ctx.Done():
			return fmt.Errorf("context cancelled: %w", ctx.Err())
		default:
		}

		n, readErr := resp.Body.Read(buf)
		if n > 0 {
			// Write to file
			if _, writeErr := out.Write(buf[:n]); writeErr != nil {
				return fmt.Errorf("failed to write file: %w", writeErr)
			}

			// Update hash
			hash.Write(buf[:n])

			downloaded += int64(n)

			// Report progress
			if progress != nil {
				elapsed := time.Since(start).Seconds()
				if elapsed > 0 {
					speed := float64(downloaded-startByte) / elapsed
					remaining := file.Size - downloaded
					eta := time.Duration(0)
					if speed > 0 {
						eta = time.Duration(float64(remaining)/speed) * time.Second
					}

					progress(DownloadProgress{
						Filename:   file.Filename,
						Downloaded: downloaded,
						Total:      file.Size,
						Speed:      speed,
						ETA:        eta,
					})
				}
			}
		}

		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			return fmt.Errorf("failed to read response: %w", readErr)
		}
	}

	// Verify checksum if provided
	if file.SHA256 != "" {
		// For checksum verification, we need to read the entire file
		out.Close()
		actualHash, err := hashFile(partialPath)
		if err != nil {
			return fmt.Errorf("failed to compute checksum: %w", err)
		}

		if !strings.EqualFold(actualHash, file.SHA256) {
			os.Remove(partialPath)
			return fmt.Errorf("checksum mismatch for %s: expected %s, got %s", file.Filename, file.SHA256, actualHash)
		}
	}

	// Rename to final path
	if err := os.Rename(partialPath, outPath); err != nil {
		return fmt.Errorf("failed to rename file: %w", err)
	}

	return nil
}

// CheckDiskSpace checks if there is enough disk space to download a model.
func (c *Client) CheckDiskSpace(ctx context.Context, repoID, outputDir string) (bool, int64, int64, error) {
	info, err := c.GetModelInfo(ctx, repoID)
	if err != nil {
		return false, 0, 0, err
	}

	return c.checkDiskSpaceInternal(info, outputDir)
}

func (c *Client) checkDiskSpaceInternal(info *ModelInfo, outputDir string) (bool, int64, int64, error) {
	// Get available disk space
	var stat syscall.Statfs_t
	if err := syscall.Statfs(outputDir, &stat); err != nil {
		// Try parent directory
		parentDir := filepath.Dir(outputDir)
		if err := syscall.Statfs(parentDir, &stat); err != nil {
			return false, 0, info.TotalSize, fmt.Errorf("failed to get disk space: %w", err)
		}
	}

	available := int64(stat.Bavail) * int64(stat.Bsize)
	required := info.TotalSize

	// Add 10% buffer
	requiredWithBuffer := int64(float64(required) * 1.1)

	return available >= requiredWithBuffer, available, required, nil
}

// hashFile computes the SHA256 hash of a file.
func hashFile(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()

	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}

	return hex.EncodeToString(h.Sum(nil)), nil
}
