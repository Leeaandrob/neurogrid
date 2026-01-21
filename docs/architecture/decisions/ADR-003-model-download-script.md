# ADR-003: Model Download Script Architecture

## Status

Accepted

## Context

NeuroGrid Engine needs to download Llama models from HuggingFace Hub. The download process has several requirements:

1. Support for both public models (TinyLlama) and gated models (Llama 2)
2. HuggingFace token authentication for gated models
3. Resume interrupted downloads for large files (13GB+)
4. Progress tracking with speed and ETA
5. Checksum verification for data integrity
6. Disk space estimation before download
7. Support for model aliases (e.g., "llama7b" -> "meta-llama/Llama-2-7b-hf")

## Decision

We implement a Go-based download solution with a bash wrapper script.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface                            │
├─────────────────────────────────────────────────────────────┤
│  scripts/download_model.sh   │   make download-tinyllama     │
│  (Bash wrapper)              │   (Makefile targets)          │
└───────────────┬──────────────┴──────────────┬───────────────┘
                │                              │
                v                              v
        ┌───────────────────────────────────────────┐
        │           cmd/download/main.go            │
        │           (Go CLI Application)            │
        │   - Argument parsing                      │
        │   - Progress display                      │
        │   - Signal handling                       │
        └───────────────────┬───────────────────────┘
                            │
                            v
        ┌───────────────────────────────────────────┐
        │         pkg/huggingface/client.go         │
        │         (HuggingFace Hub Client)          │
        │   - GetModelInfo()                        │
        │   - Download()                            │
        │   - CheckDiskSpace()                      │
        └───────────────────┬───────────────────────┘
                            │
            ┌───────────────┼───────────────┐
            v               v               v
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│  aliases.go   │ │  download.go  │ │   types.go    │
│  - Model      │ │  - Downloader │ │  - ModelInfo  │
│    aliases    │ │  - Resume     │ │  - FileInfo   │
│  - CLI args   │ │  - Checksum   │ │  - Progress   │
└───────────────┘ └───────────────┘ └───────────────┘
```

### Component Responsibilities

1. **scripts/download_model.sh**
   - User-friendly bash wrapper
   - HF_TOKEN validation for gated models
   - Fallback to Python huggingface_hub if Go unavailable

2. **cmd/download/main.go**
   - CLI argument parsing
   - Human-readable progress display
   - Graceful shutdown on SIGINT/SIGTERM

3. **pkg/huggingface/client.go**
   - HuggingFace Hub API client
   - Model info retrieval
   - Disk space checking
   - Concurrent download orchestration

4. **pkg/huggingface/download.go**
   - HTTP Range header for resume
   - SHA256 checksum verification
   - Progress callback support

5. **pkg/huggingface/aliases.go**
   - Model alias resolution
   - CLI argument parsing

### Key Design Decisions

1. **Go over Python**: Go provides better performance for large file downloads and simpler deployment (single binary).

2. **Resume support via HTTP Range headers**: Allows resuming multi-GB downloads after interruption.

3. **Partial file naming**: Uses `.partial` suffix during download, renamed on success.

4. **Concurrent downloads**: Configurable concurrency for downloading multiple files (sharded models).

5. **Checksum verification**: SHA256 verification after download with partial file deletion on mismatch.

## Consequences

### Positive

- Fast downloads with progress tracking
- Reliable resume for large models
- Simple Makefile integration
- Works without Python if Go available
- Graceful error handling for gated models

### Negative

- Requires Go installation for optimal experience
- HuggingFace API changes may require updates
- Large models still take significant time/space

### Neutral

- Python fallback maintains backward compatibility
- Model aliases need manual updates for new models

## Alternatives Considered

1. **Python-only (huggingface_hub)**
   - Pros: Well-tested, official library
   - Cons: Slower, requires Python environment

2. **Direct curl/wget**
   - Pros: Universal availability
   - Cons: No progress tracking, manual auth handling

3. **Git LFS**
   - Pros: Native HuggingFace support
   - Cons: Slower, requires git-lfs installation

## References

- [HuggingFace Hub API](https://huggingface.co/docs/huggingface_hub/)
- [HTTP Range Requests](https://developer.mozilla.org/en-US/docs/Web/HTTP/Range_requests)
- [PRP-03: Model Download Script](../prps/03-model-download-script.md)
