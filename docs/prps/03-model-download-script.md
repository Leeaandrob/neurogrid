# PRP: Model Download Script (HuggingFace)

## Document Information

| Field | Value |
|-------|-------|
| **Feature Name** | Model Download Script |
| **PRP Version** | 1.0 |
| **Created** | 2026-01-21 |
| **Priority** | Short-Term (Critical Path) |
| **Status** | Ready for Implementation |
| **Confidence Score** | 9/10 |

---

## Discovery Summary

### Initial Task Analysis

Create scripts to download Llama models from HuggingFace, including authentication handling, progress display, and verification. Support both full models and quantized versions.

### User Clarifications Received

- **Question**: Which models to support initially?
- **Answer**: TinyLlama (testing), Llama 2 7B, Llama 2 13B
- **Impact**: Script must handle various model sizes

### Missing Requirements Identified

- HuggingFace token authentication
- Resume interrupted downloads
- Checksum verification
- Space estimation before download

---

## Goal

Create `scripts/download_model.sh` and supporting Go CLI tool to download Llama models from HuggingFace Hub with authentication, progress tracking, and verification.

## Why

- **Critical path**: Need model weights to run inference
- **User experience**: Easy one-command download
- **Reliability**: Resume support for large downloads

## What

### Success Criteria

- [ ] Download TinyLlama 1.1B in < 2 minutes
- [ ] Download Llama 2 7B with progress display
- [ ] Support HF_TOKEN authentication
- [ ] Resume interrupted downloads
- [ ] Verify checksums after download
- [ ] Estimate disk space before download
- [ ] Support multiple model variants (chat, instruct)

---

## All Needed Context

### Research Phase Summary

- **Codebase patterns found**: `scripts/generate_golden.py` for script patterns
- **External research needed**: Yes - HuggingFace Hub API
- **Knowledge gaps identified**: HF Hub authentication flow

### Documentation & References

```yaml
- url: https://huggingface.co/docs/huggingface_hub/
  why: HuggingFace Hub API documentation

- url: https://huggingface.co/docs/hub/models-downloading
  why: Model download patterns and authentication

- file: scripts/generate_golden.py
  why: Script pattern to follow

- doc: https://huggingface.co/meta-llama/Llama-2-7b
  section: Files and versions
  critical: File structure and naming
```

### Current Codebase tree

```
scripts/
├── generate_golden.py     # PyTorch reference generation
└── test_e2e.sh           # E2E test runner
```

### Desired Codebase tree

```
scripts/
├── download_model.sh      # Main download script (bash wrapper)
├── generate_golden.py
└── test_e2e.sh

cmd/
├── download/              # Go CLI for downloading
│   └── main.go
```

### Known Gotchas

```bash
# CRITICAL: Meta models require license acceptance on HuggingFace
# CRITICAL: HF_TOKEN must be set for gated models
# CRITICAL: safetensors.index.json contains shard mapping
# CRITICAL: Large models (70B) need 140GB+ free space
```

---

## Implementation Blueprint

### Data Models

```go
// cmd/download/main.go

type ModelInfo struct {
    RepoID       string   `json:"repo_id"`
    Revision     string   `json:"revision"`
    Files        []FileInfo `json:"files"`
    TotalSize    int64    `json:"total_size"`
    RequiresAuth bool     `json:"requires_auth"`
}

type FileInfo struct {
    Filename string `json:"filename"`
    Size     int64  `json:"size"`
    SHA256   string `json:"sha256"`
    URL      string `json:"url"`
}

type DownloadProgress struct {
    Filename    string
    Downloaded  int64
    Total       int64
    Speed       float64
    ETA         time.Duration
}
```

### Task List

```yaml
Task 1: Create bash wrapper script
  CREATE scripts/download_model.sh:
    - Parse model name argument
    - Check for HF_TOKEN
    - Call Go download CLI
    - Handle common models by alias

Task 2: Create Go download CLI
  CREATE cmd/download/main.go:
    - Parse arguments (model, output dir)
    - Fetch model info from HF Hub API
    - Display size estimate
    - Download with progress

Task 3: Implement HF Hub API client
  CREATE pkg/huggingface/client.go:
    - GetModelInfo endpoint
    - Authentication handling
    - File listing

Task 4: Implement download with resume
  CREATE pkg/huggingface/download.go:
    - HTTP Range header for resume
    - Progress callback
    - Checksum verification
    - Concurrent downloads

Task 5: Add model aliases
  CREATE configs/models.yaml:
    - Alias mappings (tinyllama → TinyLlama/TinyLlama-1.1B-Chat-v1.0)
    - Default variants
    - Size estimates

Task 6: Add Makefile targets
  MODIFY Makefile:
    - download-tinyllama target
    - download-llama7b target
    - download-llama13b target

Task 7: Add tests
  CREATE tests/download/download_test.go:
    - Test model info fetch
    - Test download resume
    - Test checksum verification
```

### Per-Task Pseudocode

```bash
# Task 1: Bash wrapper
#!/bin/bash
# scripts/download_model.sh

MODEL=${1:-tinyllama}
OUTPUT=${2:-./models}

# Check HF_TOKEN for gated models
if [[ "$MODEL" == llama* ]] && [[ -z "$HF_TOKEN" ]]; then
    echo "Error: HF_TOKEN required for Llama models"
    echo "Get token at: https://huggingface.co/settings/tokens"
    exit 1
fi

# Map aliases
case $MODEL in
    tinyllama)  REPO="TinyLlama/TinyLlama-1.1B-Chat-v1.0" ;;
    llama7b)    REPO="meta-llama/Llama-2-7b-hf" ;;
    llama7b-chat) REPO="meta-llama/Llama-2-7b-chat-hf" ;;
    llama13b)   REPO="meta-llama/Llama-2-13b-hf" ;;
    *)          REPO="$MODEL" ;;
esac

# Run Go download tool
go run ./cmd/download --repo "$REPO" --output "$OUTPUT" --token "$HF_TOKEN"
```

```go
// Task 2: Go download CLI
func main() {
    repo := flag.String("repo", "", "HuggingFace repo ID")
    output := flag.String("output", "./models", "Output directory")
    token := flag.String("token", os.Getenv("HF_TOKEN"), "HuggingFace token")
    flag.Parse()

    client := huggingface.NewClient(*token)

    // Get model info
    info, err := client.GetModelInfo(*repo)
    if err != nil {
        log.Fatalf("Failed to get model info: %v", err)
    }

    // Show size estimate
    fmt.Printf("Model: %s\n", *repo)
    fmt.Printf("Total size: %s\n", humanize.Bytes(uint64(info.TotalSize)))
    fmt.Printf("Files: %d\n", len(info.Files))

    // Check disk space
    if !hasSpace(info.TotalSize) {
        log.Fatal("Insufficient disk space")
    }

    // Download files
    for _, file := range info.Files {
        if err := client.DownloadFile(file, *output, progressCallback); err != nil {
            log.Fatalf("Download failed: %v", err)
        }
    }

    fmt.Println("Download complete!")
}

// Task 4: Download with resume
func (c *Client) DownloadFile(file FileInfo, outputDir string, progress func(DownloadProgress)) error {
    outPath := filepath.Join(outputDir, file.Filename)

    // Check for partial download
    var startByte int64
    if stat, err := os.Stat(outPath + ".partial"); err == nil {
        startByte = stat.Size()
    }

    // Create request with Range header
    req, _ := http.NewRequest("GET", file.URL, nil)
    if startByte > 0 {
        req.Header.Set("Range", fmt.Sprintf("bytes=%d-", startByte))
    }
    if c.token != "" {
        req.Header.Set("Authorization", "Bearer "+c.token)
    }

    // Download with progress
    resp, err := c.httpClient.Do(req)
    if err != nil {
        return err
    }
    defer resp.Body.Close()

    // Write to .partial file
    out, _ := os.OpenFile(outPath+".partial", os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
    defer out.Close()

    // Copy with progress tracking
    buf := make([]byte, 32*1024)
    downloaded := startByte
    start := time.Now()

    for {
        n, err := resp.Body.Read(buf)
        if n > 0 {
            out.Write(buf[:n])
            downloaded += int64(n)

            elapsed := time.Since(start).Seconds()
            speed := float64(downloaded-startByte) / elapsed
            eta := time.Duration(float64(file.Size-downloaded)/speed) * time.Second

            progress(DownloadProgress{
                Filename:   file.Filename,
                Downloaded: downloaded,
                Total:      file.Size,
                Speed:      speed,
                ETA:        eta,
            })
        }
        if err == io.EOF {
            break
        }
        if err != nil {
            return err
        }
    }

    // Verify checksum
    if err := verifyChecksum(outPath+".partial", file.SHA256); err != nil {
        return fmt.Errorf("checksum mismatch: %w", err)
    }

    // Rename to final
    return os.Rename(outPath+".partial", outPath)
}
```

### Integration Points

```yaml
MAKEFILE:
  - add targets: download-tinyllama, download-llama7b, download-llama13b
  - pattern: 'go run ./cmd/download --repo $(REPO) --output models/'

CONFIG:
  - add: configs/models.yaml with model aliases
  - usage: Script reads aliases for common models

ENVIRONMENT:
  - HF_TOKEN: Required for gated models
  - MODELS_DIR: Optional, default ./models
```

---

## Validation Loop

### Level 1: Syntax & Style

```bash
shellcheck scripts/download_model.sh
go fmt ./cmd/download/...
go vet ./cmd/download/...

# Expected: No errors
```

### Level 2: Unit Tests

```bash
go test -v ./pkg/huggingface/...

# Expected: All tests pass
```

### Level 3: Integration Test

```bash
# Download small model
./scripts/download_model.sh tinyllama ./models/test

# Verify files
ls -la models/test/
# Expected: tokenizer.model, config.json, model.safetensors

# Verify model loads
go test -v -run TestLoadTinyLlama ./tests/model/...
```

---

## Final Validation Checklist

- [ ] TinyLlama downloads successfully
- [ ] Llama 2 7B downloads with HF_TOKEN
- [ ] Progress display works
- [ ] Resume works after interruption
- [ ] Checksums verified
- [ ] Disk space check works
- [ ] Makefile targets work

---

## Anti-Patterns to Avoid

- ❌ Don't store HF_TOKEN in scripts
- ❌ Don't skip checksum verification
- ❌ Don't download to final path directly (use .partial)
- ❌ Don't assume network is reliable
- ❌ Don't hardcode model URLs

---

## Performance Targets

| Metric | Target |
|--------|--------|
| TinyLlama download | < 2 min |
| Llama 7B download | < 30 min (on 100Mbps) |
| Resume overhead | < 1 second |

---

## Supported Models

| Alias | HuggingFace Repo | Size |
|-------|------------------|------|
| tinyllama | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 2.2 GB |
| llama7b | meta-llama/Llama-2-7b-hf | 13 GB |
| llama7b-chat | meta-llama/Llama-2-7b-chat-hf | 13 GB |
| llama13b | meta-llama/Llama-2-13b-hf | 26 GB |
| llama13b-chat | meta-llama/Llama-2-13b-chat-hf | 26 GB |

---

**PRP Confidence Score: 9/10**

**Rationale**:
- +3: HuggingFace Hub API well-documented
- +2: Simple HTTP downloads
- +2: Clear requirements
- +2: Standard tools (bash, curl/Go http)
- -1: Auth flow for gated models needs testing
