# Changelog

All notable changes to NeuroGrid Inference Engine will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-01-21

### Added

#### Core Inference Engine
- Full GPU-accelerated inference pipeline using CUDA and cuBLAS
- Support for TinyLlama-1.1B and Llama-architecture models
- BF16 to FP16 automatic weight conversion during model loading
- SafeTensors format support for model weights
- SentencePiece tokenizer integration

#### GPU Acceleration
- cuBLAS-based FP16 GEMM operations for matrix multiplication
- INT8 quantized weight support with per-row dequantization scales
- GPU memory management with automatic allocation/deallocation
- Multi-GPU context support (foundation for distributed inference)
- RoPE (Rotary Position Embedding) implementation
- RMSNorm layer normalization
- SiLU activation function
- Flash-style attention implementation

#### Network Layer
- libp2p-based peer-to-peer networking
- mDNS peer discovery for local network
- CUDA transport for local GPU-to-GPU communication
- Layer-to-peer assignment routing
- Single-node local-only mode (`--min-peers 0`)

#### API
- OpenAI-compatible REST API (`/v1/chat/completions`)
- HTTP server with configurable port
- JSON request/response format
- Model configuration auto-detection from `config.json`

#### Scheduler
- Fair layer distribution across peers
- VRAM-aware scheduling
- Single-GPU mode for local-only execution

#### Testing & Benchmarking
- E2E real inference tests with coherence validation
- TTFT (Time To First Token) tests
- GPU GEMM unit tests (FP16, INT8, transpose operations)
- Comprehensive benchmark tool with metrics:
  - TTFT (Time To First Token)
  - ITL (Inter-Token Latency)
  - TPOT (Time Per Output Token)
  - TPS (Tokens Per Second)
  - Latency percentiles (P50, P90, P95, P99)

### Performance Metrics (TinyLlama-1.1B, Single GPU)
- TTFT: ~1.3s (P50)
- TPOT: ~145ms/token (P50)
- Generation TPS: ~6.9 tokens/sec
- Throughput: ~7 requests/sec (sequential)

### Known Limitations
- Streaming responses not yet implemented
- No KV cache optimization (continuous batching)
- No Flash Attention integration
- No CUDA graphs optimization
- No speculative decoding

### Technical Details
- Language: Go with CGO for CUDA bindings
- CUDA: Requires CUDA 11.x+ with cuBLAS
- Models: Llama-architecture models in SafeTensors format
- Precision: FP16 inference with BF16 weight conversion

## [Unreleased]

### Planned
- Streaming response support (SSE)
- KV cache with continuous batching
- Flash Attention integration
- CUDA graphs for reduced kernel launch overhead
- Speculative decoding
- Multi-GPU distributed inference
- INT4 quantization support
- OpenAI-compatible streaming format
