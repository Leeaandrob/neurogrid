package huggingface

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// Downloader handles file downloads with progress tracking and resume support.
type Downloader struct {
	client   *http.Client
	token    string
	progress ProgressFunc

	// Interceptors for testing
	requestInterceptor  func(*http.Request)
	responseInterceptor func(*http.Response)
}

// NewDownloader creates a new downloader with the given HTTP client and token.
func NewDownloader(client *http.Client, token string) *Downloader {
	return &Downloader{
		client: client,
		token:  token,
	}
}

// SetProgress sets the progress callback function.
func (d *Downloader) SetProgress(fn ProgressFunc) {
	d.progress = fn
}

// SetInterceptors sets request/response interceptors for testing.
func (d *Downloader) SetInterceptors(req func(*http.Request), resp func(*http.Response)) {
	d.requestInterceptor = req
	d.responseInterceptor = resp
}

// DownloadFiles downloads multiple files concurrently.
func (d *Downloader) DownloadFiles(ctx context.Context, files []FileInfo, outputDir string, concurrency int) error {
	if concurrency <= 1 {
		return d.downloadSequential(ctx, files, outputDir)
	}
	return d.downloadConcurrent(ctx, files, outputDir, concurrency)
}

func (d *Downloader) downloadSequential(ctx context.Context, files []FileInfo, outputDir string) error {
	for _, file := range files {
		select {
		case <-ctx.Done():
			return fmt.Errorf("context cancelled: %w", ctx.Err())
		default:
		}

		if err := d.DownloadFile(ctx, file, outputDir); err != nil {
			return fmt.Errorf("failed to download %s: %w", file.Filename, err)
		}
	}
	return nil
}

func (d *Downloader) downloadConcurrent(ctx context.Context, files []FileInfo, outputDir string, concurrency int) error {
	errChan := make(chan error, len(files))
	semaphore := make(chan struct{}, concurrency)
	var wg sync.WaitGroup

	for _, file := range files {
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

			if err := d.DownloadFile(ctx, f, outputDir); err != nil {
				errChan <- fmt.Errorf("failed to download %s: %w", f.Filename, err)
			}
		}(file)
	}

	wg.Wait()
	close(errChan)

	// Return first error encountered
	for err := range errChan {
		if err != nil {
			return err
		}
	}
	return nil
}

// DownloadFile downloads a single file with resume support.
func (d *Downloader) DownloadFile(ctx context.Context, file FileInfo, outputDir string) error {
	outPath := filepath.Join(outputDir, file.Filename)
	partialPath := outPath + ".partial"

	// Ensure output directory exists
	if err := os.MkdirAll(filepath.Dir(outPath), 0755); err != nil {
		return fmt.Errorf("failed to create output directory: %w", err)
	}

	// Check for partial download to resume
	startByte := d.getPartialSize(partialPath)

	// Create and execute request
	resp, actualStart, err := d.executeRequest(ctx, file.URL, startByte)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	// Open file for writing
	out, err := d.openOutputFile(partialPath, actualStart > 0 && resp.StatusCode == http.StatusPartialContent)
	if err != nil {
		return err
	}
	defer out.Close()

	// Reset startByte if not resuming
	if resp.StatusCode != http.StatusPartialContent {
		actualStart = 0
	}

	// Get total size from Content-Length if not available from API
	totalSize := file.Size
	if totalSize == 0 && resp.ContentLength > 0 {
		totalSize = resp.ContentLength
		if actualStart > 0 {
			// If resuming, add the startByte to get actual total
			totalSize += actualStart
		}
	}

	// Download with progress tracking
	if err := d.copyWithProgress(ctx, out, resp.Body, file, actualStart, totalSize); err != nil {
		return err
	}

	// Verify checksum if provided
	if file.SHA256 != "" {
		out.Close()
		if err := d.verifyChecksum(partialPath, file.SHA256); err != nil {
			os.Remove(partialPath)
			return err
		}
	}

	// Rename to final path
	return os.Rename(partialPath, outPath)
}

func (d *Downloader) getPartialSize(path string) int64 {
	if stat, err := os.Stat(path); err == nil {
		return stat.Size()
	}
	return 0
}

func (d *Downloader) executeRequest(ctx context.Context, url string, startByte int64) (*http.Response, int64, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, 0, fmt.Errorf("failed to create request: %w", err)
	}

	if d.token != "" {
		req.Header.Set("Authorization", "Bearer "+d.token)
	}

	if startByte > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", startByte))
	}

	if d.requestInterceptor != nil {
		d.requestInterceptor(req)
	}

	resp, err := d.client.Do(req)
	if err != nil {
		return nil, 0, fmt.Errorf("failed to download file: %w", err)
	}

	if d.responseInterceptor != nil {
		d.responseInterceptor(resp)
	}

	if resp.StatusCode == http.StatusUnauthorized {
		resp.Body.Close()
		return nil, 0, fmt.Errorf("authentication required")
	}

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusPartialContent {
		resp.Body.Close()
		return nil, 0, fmt.Errorf("unexpected status code: %d", resp.StatusCode)
	}

	return resp, startByte, nil
}

func (d *Downloader) openOutputFile(path string, append bool) (*os.File, error) {
	flags := os.O_CREATE | os.O_WRONLY
	if append {
		flags |= os.O_APPEND
	} else {
		flags |= os.O_TRUNC
	}
	return os.OpenFile(path, flags, 0644)
}

func (d *Downloader) copyWithProgress(ctx context.Context, out io.Writer, in io.Reader, file FileInfo, startByte int64, totalSize int64) error {
	buf := make([]byte, 32*1024) // 32KB buffer
	downloaded := startByte
	start := time.Now()

	for {
		select {
		case <-ctx.Done():
			return fmt.Errorf("context cancelled: %w", ctx.Err())
		default:
		}

		n, readErr := in.Read(buf)
		if n > 0 {
			if _, writeErr := out.Write(buf[:n]); writeErr != nil {
				return fmt.Errorf("failed to write file: %w", writeErr)
			}

			downloaded += int64(n)
			d.reportProgress(file, downloaded, startByte, start, totalSize)
		}

		if readErr == io.EOF {
			break
		}
		if readErr != nil {
			return fmt.Errorf("failed to read response: %w", readErr)
		}
	}

	return nil
}

func (d *Downloader) reportProgress(file FileInfo, downloaded, startByte int64, start time.Time, totalSize int64) {
	if d.progress == nil {
		return
	}

	elapsed := time.Since(start).Seconds()
	if elapsed <= 0 {
		return
	}

	// Use provided totalSize, fall back to file.Size
	total := totalSize
	if total <= 0 {
		total = file.Size
	}

	speed := float64(downloaded-startByte) / elapsed
	remaining := total - downloaded
	if remaining < 0 {
		remaining = 0
	}
	eta := time.Duration(0)
	if speed > 0 && remaining > 0 {
		eta = time.Duration(float64(remaining)/speed) * time.Second
	}

	d.progress(DownloadProgress{
		Filename:   file.Filename,
		Downloaded: downloaded,
		Total:      total,
		Speed:      speed,
		ETA:        eta,
	})
}

func (d *Downloader) verifyChecksum(path, expected string) error {
	actual, err := HashFile(path)
	if err != nil {
		return fmt.Errorf("failed to compute checksum: %w", err)
	}

	if !strings.EqualFold(actual, expected) {
		return fmt.Errorf("checksum mismatch: expected %s, got %s", expected, actual)
	}

	return nil
}

// HashFile computes the SHA256 hash of a file.
func HashFile(path string) (string, error) {
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
