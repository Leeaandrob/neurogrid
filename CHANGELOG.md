# Changelog

All notable changes to NeuroGrid Inference Engine will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.1] - 2026-01-26

### Added

#### Distributed Inference Improvements
- **`-disable-mdns` flag** - Disable mDNS peer discovery for controlled environments
  - Prevents "ghost peer" issues from stale k8s pods or old processes
  - Workers connect explicitly via `-bootstrap` address
  - Recommended for production deployments

#### Weight Distributed Memory
- Workers can now load model weights locally with `-model` flag
- Combined with `-skip-weight-transfer` on coordinator for instant startup
- No P2P weight distribution overhead when workers have local models

#### Heterogeneous GPU Support
- **P2P GPU Info Protocol** - Workers report actual VRAM to coordinator
  - `MsgTypeGPUInfo` (0x08) - Worker sends GPU info
  - `MsgTypeGPUInfoRequest` (0x09) - Coordinator requests GPU info
- Scheduler now uses **real VRAM values** from each worker
  - No longer assumes all workers have same VRAM as coordinator
  - Enables proper layer distribution across GPUs with different capacities
- Workers proactively send GPU info when connecting to coordinator

### Fixed
- **Ghost Peer Issue** - Stale peers from k8s pods or previous sessions no longer interfere
  - Root cause: mDNS discovering cached/old peer announcements
  - Solution: Use `-disable-mdns` with explicit bootstrap addresses
- **Heterogeneous GPU OOM** - Scheduler now correctly estimates VRAM for each worker
  - Root cause: Coordinator was using local GPU's VRAM for all workers
  - Solution: Workers report actual VRAM via P2P protocol
- **Scheduler Memory Estimation** - Improved for distributed inference across heterogeneous GPUs
- **CRITICAL: Garbage Output in Distributed Inference** - RopeTheta mismatch between coordinator and workers
  - Root cause: Coordinator's `getModelConfigFromPath` was not passing `RopeTheta` from config.json
  - Impact: Coordinator used default RopeTheta=10000 while workers used model's actual value (e.g., 1000000 for Mistral Nemo)
  - Result: RoPE position encodings were incompatible, causing completely garbled output
  - Solution: Coordinator now correctly reads and uses `rope_theta` from model's config.json

### Changed
- Better documentation for distributed mode setup
- Coordinator logs now show when mDNS is disabled vs enabled
- Coordinator logs GPU info received from each worker
- Enhanced model config logging on both coordinator and workers (now includes RopeTheta)

---

## [0.3.0] - 2025-01-23

### Added

#### Distributed Inference
- **Coordinator/Worker Architecture** - Pipeline parallelism across multiple GPUs/machines
- **P2P Weight Distribution** - Automatic layer weight transfer between peers via libp2p
- **Remote Layer Execution** - Execute transformer layers on remote workers
- **Network Notifee** - Automatic detection of incoming worker connections
- `--skip-weight-transfer` flag for workers with pre-loaded weights
- `--min-peers` flag to wait for minimum number of workers before starting
- `--bootstrap` flag for explicit peer connection

#### Model Support
- **Mistral 7B Support** - Chat template, RoPE configuration, model config detection
- **Llama 2 13B Support** - Configuration alias and benchmark results
- Model family auto-detection (Llama 2, Llama 3, Mistral, TinyLlama)

#### Build System
- `make run` - Auto-detect model and start server
- `make run-coordinator` / `make run-worker` - Distributed mode targets
- `make download REPO=org/model` - Generic HuggingFace model download
- `make download-mistral7b-instruct` - Mistral 7B Instruct download
- Model auto-detection from `./models/` directory

#### Testing
- Comprehensive distributed inference E2E tests (1311 lines)
- Model loader unit tests (688 lines)
- Router unit tests with full coverage
- Scheduler E2E tests

### Changed
- Improved peer discovery timeout (5 minutes)
- GPU buffer preallocation for better performance
- Disabled CUDA debug prints in production builds
- Refactored chat template system for multi-model support

### Fixed
- PrefetchCoordinator blocking issue resolved
- Local weight upload to GPU in worker mode
- Layer routing for local vs remote execution
- Model loading issues with various SafeTensors formats

### Performance
- **Llama 2 7B** (Single RTX 4090):
  - TTFT: ~2.1s (P50)
  - Generation: ~5.2 tokens/sec
- **Llama 2 13B** (Distributed: RTX 4090 + GH200):
  - TTFT: ~3.8s (P50)
  - Generation: ~3.1 tokens/sec

---

## [0.2.0] - 2025-01-22

### Added

#### Streaming & API
- **SSE Streaming** - Server-Sent Events for real-time token streaming
- Fixed `WriteHeader` and `Transfer-Encoding` for proper streaming
- OpenAI-compatible streaming format

#### Observability
- **Prometheus Metrics** - Full metrics registry with `promauto`
- Request duration histograms
- TTFT (Time To First Token) metrics
- Token generation counters
- GPU memory and utilization gauges
- Cluster peer metrics

#### Networking
- **NAT Traversal** - AutoNAT, hole punching, and relay support
- **Peer Reconnection** - Exponential backoff with configurable max retries
- Peer manager for connection lifecycle management

#### Infrastructure
- `pkg/metrics/registry.go` - Centralized Prometheus metrics
- `p2p/peer_manager.go` - Peer connection management
- `pkg/inference/engine_nogpu.go` - Stub for non-CUDA builds
- `types.TinyLlamaConfig` - Smaller test configuration

### Fixed
- `fmt.Errorf` non-constant format strings
- Divide by zero in throughput tests
- Engine initialization with router and scheduler

---

## [0.1.0] - 2025-01-21

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
- No KV cache optimization (continuous batching)
- No Flash Attention integration
- No CUDA graphs optimization
- No speculative decoding

### Technical Details
- Language: Go with CGO for CUDA bindings
- CUDA: Requires CUDA 11.x+ with cuBLAS
- Models: Llama-architecture models in SafeTensors format
- Precision: FP16 inference with BF16 weight conversion

---

## [Unreleased]

### Planned
- Observability infrastructure (OpenTelemetry, Prometheus dashboards)
- Tensor checkpoints for debugging
- KV cache with continuous batching
- Flash Attention integration
- CUDA graphs for reduced kernel launch overhead
- Speculative decoding
- INT4 quantization support
