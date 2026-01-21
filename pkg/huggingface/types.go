// Package huggingface provides a client for the HuggingFace Hub API.
// This package enables downloading models from HuggingFace with support for
// authentication, progress tracking, resume, and checksum verification.
package huggingface

import (
	"time"
)

// ModelInfo contains information about a model from HuggingFace Hub.
type ModelInfo struct {
	RepoID       string     `json:"repo_id"`
	Revision     string     `json:"revision"`
	Files        []FileInfo `json:"files"`
	TotalSize    int64      `json:"total_size"`
	RequiresAuth bool       `json:"requires_auth"`
}

// FileInfo contains information about a file in a model repository.
type FileInfo struct {
	Filename string `json:"filename"`
	Size     int64  `json:"size"`
	SHA256   string `json:"sha256"`
	URL      string `json:"url"`
}

// DownloadProgress reports the progress of a file download.
type DownloadProgress struct {
	Filename   string
	Downloaded int64
	Total      int64
	Speed      float64       // bytes per second
	ETA        time.Duration // estimated time remaining
}

// CLIArgs represents the command-line arguments for the download CLI.
type CLIArgs struct {
	Repo   string
	Output string
	Token  string
}

// ProgressFunc is a callback function for reporting download progress.
type ProgressFunc func(DownloadProgress)
