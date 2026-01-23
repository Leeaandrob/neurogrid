// Package main provides a CLI tool for downloading models from HuggingFace Hub.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"github.com/neurogrid/engine/pkg/huggingface"
)

func main() {
	// Parse flags
	repo := flag.String("repo", "", "HuggingFace repo ID or alias (required)")
	output := flag.String("output", "./models", "Output directory")
	token := flag.String("token", os.Getenv("HF_TOKEN"), "HuggingFace token (or set HF_TOKEN env var)")
	concurrency := flag.Int("concurrency", 4, "Number of concurrent downloads")
	flag.Parse()

	// Validate required args
	if *repo == "" {
		fmt.Fprintln(os.Stderr, "Error: --repo is required")
		fmt.Fprintln(os.Stderr, "")
		fmt.Fprintln(os.Stderr, "Usage: download --repo <model> [options]")
		fmt.Fprintln(os.Stderr, "")
		fmt.Fprintln(os.Stderr, "Examples:")
		fmt.Fprintln(os.Stderr, "  download --repo tinyllama")
		fmt.Fprintln(os.Stderr, "  download --repo llama7b --output ./models/llama")
		fmt.Fprintln(os.Stderr, "  download --repo meta-llama/Llama-2-7b-hf --token $HF_TOKEN")
		fmt.Fprintln(os.Stderr, "")
		fmt.Fprintln(os.Stderr, "Supported aliases:")
		fmt.Fprintln(os.Stderr, "  tinyllama          -> TinyLlama/TinyLlama-1.1B-Chat-v1.0")
		fmt.Fprintln(os.Stderr, "  llama7b            -> meta-llama/Llama-2-7b-hf")
		fmt.Fprintln(os.Stderr, "  llama7b-chat       -> meta-llama/Llama-2-7b-chat-hf")
		fmt.Fprintln(os.Stderr, "  llama13b           -> meta-llama/Llama-2-13b-hf")
		fmt.Fprintln(os.Stderr, "  llama13b-chat      -> meta-llama/Llama-2-13b-chat-hf")
		fmt.Fprintln(os.Stderr, "  mistral7b          -> mistralai/Mistral-7B-v0.3")
		fmt.Fprintln(os.Stderr, "  mistral7b-instruct -> mistralai/Mistral-7B-Instruct-v0.3")
		os.Exit(1)
	}

	// Resolve alias
	repoID := huggingface.ResolveAlias(*repo)

	// Create client
	client := huggingface.NewClient(*token)
	client.SetConcurrency(*concurrency)

	// Setup context with cancellation
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Handle signals for graceful shutdown
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigChan
		fmt.Fprintln(os.Stderr, "\nInterrupted. Cancelling download...")
		cancel()
	}()

	// Get model info first
	fmt.Printf("NeuroGrid Model Downloader\n")
	fmt.Printf("==========================\n\n")
	fmt.Printf("Model: %s\n", repoID)
	fmt.Printf("Output: %s\n\n", *output)

	fmt.Printf("Fetching model info...\n")
	info, err := client.GetModelInfo(ctx, repoID)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		if info == nil && *token == "" {
			fmt.Fprintln(os.Stderr, "")
			fmt.Fprintln(os.Stderr, "Hint: If this is a gated model, you need to:")
			fmt.Fprintln(os.Stderr, "  1. Accept the license at https://huggingface.co/"+repoID)
			fmt.Fprintln(os.Stderr, "  2. Set HF_TOKEN environment variable or use --token")
		}
		os.Exit(1)
	}

	// Show model info
	fmt.Printf("\nModel: %s\n", info.RepoID)
	fmt.Printf("Total size: %s\n", humanizeBytes(info.TotalSize))
	fmt.Printf("Files: %d\n\n", len(info.Files))

	// Check disk space
	hasSpace, available, required, err := client.CheckDiskSpace(ctx, repoID, *output)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Warning: Could not check disk space: %v\n", err)
	} else {
		fmt.Printf("Required: %s\n", humanizeBytes(required))
		fmt.Printf("Available: %s\n", humanizeBytes(available))
		if !hasSpace {
			fmt.Fprintf(os.Stderr, "Error: Insufficient disk space\n")
			os.Exit(1)
		}
		fmt.Printf("\n")
	}

	// Download with progress
	fmt.Printf("Downloading...\n\n")

	lastFile := ""
	progress := func(p huggingface.DownloadProgress) {
		if p.Filename != lastFile {
			if lastFile != "" {
				fmt.Printf("\n")
			}
			lastFile = p.Filename
		}
		percent := float64(p.Downloaded) / float64(p.Total) * 100
		fmt.Printf("\r  %s: %.1f%% (%s/%s) @ %s/s ETA: %s",
			p.Filename,
			percent,
			humanizeBytes(p.Downloaded),
			humanizeBytes(p.Total),
			humanizeBytes(int64(p.Speed)),
			formatDuration(p.ETA),
		)
	}

	if err := client.Download(ctx, repoID, *output, progress); err != nil {
		fmt.Fprintf(os.Stderr, "\nError: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("\n\nDownload complete!\n")
	fmt.Printf("Files saved to: %s\n", *output)
}

func humanizeBytes(bytes int64) string {
	const (
		KB = 1024
		MB = 1024 * KB
		GB = 1024 * MB
		TB = 1024 * GB
	)

	switch {
	case bytes >= TB:
		return fmt.Sprintf("%.2f TB", float64(bytes)/float64(TB))
	case bytes >= GB:
		return fmt.Sprintf("%.2f GB", float64(bytes)/float64(GB))
	case bytes >= MB:
		return fmt.Sprintf("%.2f MB", float64(bytes)/float64(MB))
	case bytes >= KB:
		return fmt.Sprintf("%.2f KB", float64(bytes)/float64(KB))
	default:
		return fmt.Sprintf("%d B", bytes)
	}
}

func formatDuration(d interface{}) string {
	switch v := d.(type) {
	case int64:
		return formatSeconds(v)
	default:
		// Handle time.Duration
		if dur, ok := d.(interface{ Seconds() float64 }); ok {
			return formatSeconds(int64(dur.Seconds()))
		}
		return "N/A"
	}
}

func formatSeconds(seconds int64) string {
	if seconds < 0 {
		return "N/A"
	}
	if seconds < 60 {
		return fmt.Sprintf("%ds", seconds)
	}
	if seconds < 3600 {
		return fmt.Sprintf("%dm %ds", seconds/60, seconds%60)
	}
	return fmt.Sprintf("%dh %dm", seconds/3600, (seconds%3600)/60)
}
