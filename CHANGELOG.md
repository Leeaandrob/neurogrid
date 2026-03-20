# Changelog

All notable changes to NeuroGrid Inference Engine will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.12.0] - 2026-03-20

### Continuous Batching Phase 2 — Batched Multi-Sequence Decode

Multiple sequences processed in a single batched forward pass with
per-sequence state isolation.

#### CUDA Kernels
- **cuda_decode_step_batched**: orchestrates all 16 layers for N sequences
- **cuda_paged_attention_batched**: N sequences in one kernel launch
  (grid: batch_size × num_heads, per-sequence block tables + seq_lens)
- Per-sequence conv state via contiguous GPU buffer `[num_conv, batch, H, K]`
- Per-sequence paged attention via cuda_layer_forward_bf16_paged

#### Architecture
- Conv layers: per-sequence (contiguous state buffer, state swap per decode step)
- Attention layers: per-sequence paged attention (reads from per-sequence KV cache)
- Embeddings: batched concatenation
- Block tables: per-sequence update before each attention layer

#### Correctness
- No cross-contamination between concurrent sequences
- Per-sequence conv state save/restore via contiguous GPU buffer
- H2O, Paris, 2+2 — all verified independently in concurrent mode

---

## [0.11.0] - 2026-03-20

### Continuous Batching Phase 1 — Concurrent Request Processing

Multiple requests processed concurrently via BatchScheduler. Each request
has independent lifecycle (prefill → decode → finish) with isolated state.

#### Architecture
- **BatchScheduler**: admit → buffer → prefill → decode (round-robin) → sample → cleanup
- **Per-sequence state**: conv state save/restore, paged KV cache blocks, position tracking
- **Phase isolation**: prefill only runs when no sequences are decoding (prevents GPU state corruption)
- **Auto-enabled**: 8 max concurrent sequences after GPU init

#### Performance (GH200 480GB)
| Config | Time | Speedup |
|--------|------|---------|
| 3 sequential | 18.9s | 1x |
| **3 concurrent** | **10.4s** | **1.8x** |
| Single request | 0.5s (Paris) | — |

Correctness verified: Paris, H₂O, 2+2 — all independent, no cross-contamination.

---

## [0.10.0] - 2026-03-20

### Batched Prefill + Prefix Caching Infrastructure

Process all prefill tokens through attention layers in a single batched pass
instead of N sequential passes. KV cache populated via per-token writes
matching the decode path (vLLM pattern).

#### Batched Prefill
- **Conv layers**: processed sequentially (matches decode conv state exactly)
- **Attention layers**: batched with `basic_attention_gqa` (causal mask)
- **Q/K/V transpose**: `[seq,heads,dim] → [heads,seq,dim]` before attention
- **Per-token KV cache write**: using `cuda_paged_kvcache_update` (same as decode)
- **Batch embedding gather**: `cuda_gather_embeddings` for N tokens at once

#### New CUDA Kernels
- `cuda_transpose_shd_to_hsd_fp16` / `_hsd_to_shd`: attention layout transpose
- `cuda_reshape_and_cache`: vLLM-style KV cache write via slot_mapping
- `cuda_gather_embeddings`: batch token embedding lookup
- `cuda_prefill_batch`: batched forward through all layers
- `cuda_workspace_bf16_get_kv_fp16`: workspace K/V accessor

#### Prefix Caching (infrastructure, disabled)
- `PrefixCache`: SHA256 block hashing for KV cache reuse
- `AllocateWithPrefix` / `FreeWithPrefix`: prefix-aware block allocation
- Conv state save/restore (240KB per snapshot)
- Cache hits verified: 2-3 blocks reused across requests

#### Critical Bug Fixes
- **SetDecodePagedCache nil guard**: Go binding skipped `use_paged=true` when
  `pagedCache` was nil (always the case with per-layer caches). KV cache writes
  in batch prefill never executed. THE root cause of the batch prefill bug.
- **Missing forward declaration**: `cuda_paged_kvcache_update` undefined in
  decode_all.cu — build silently used stale .o file without KV write code.

#### Benchmark (LFM2.5-1.2B, BF16 native, GH200 480GB)
| Config | tok/s |
|--------|-------|
| Sequential prefill | 362 |
| **Batched prefill** | **348** |

Correctness: Paris, 4, H₂O, gravity — all verified with EOS.

---

## [0.9.0] - 2026-03-19

### BF16 Native Compute — Precision-Preserving Inference

Full BF16 native compute pipeline for LFM2 attention layers, eliminating
FP16 precision loss that caused model overthinking (infinite `<think>` loop).

#### BF16 Native Pipeline
- **BF16 attention weights**: loaded directly from safetensors without FP16 conversion
- **BF16 GEMM** via cublasGemmEx (CUDA_R_16BF, CUBLAS_COMPUTE_32F)
- **BF16 RMSNorm, SiLU, RoPE**: all intermediate ops in BF16 with FP32 accumulation
- **FP16↔BF16 conversion**: at attention boundary (Q/K/V to FP16 for paged attention kernel)
- **BF16→FP16 post-decode**: automatic conversion for LM head compatibility

#### Critical Bug Fixes
- **Double-forward bug**: first decode step was running layers on prefill output (already processed), computing `layers(layers(embed))` instead of `layers(embed)`. Fixed by applying LM head directly to prefill output at i=0.
- **Missing RMSNorm in ForwardFromGPU**: GPU-pointer LM head path skipped final layernorm, producing flat logits (~0.5 max vs ~30 expected). This was the root cause of GPU-resident decode producing garbage output.
- **BF16 hidden state never populated**: `cuda_decode_step` copied FP16 to `hidden_a` but never converted to `bf16_hidden_a` for the BF16 native path. Added FP16→BF16 conversion in both decode paths.

#### Production Hardening
- **Request serialization**: exclusive mutex prevents concurrent request corruption
- **KV cache + conv state reset**: cleared between requests (paged blocks freed, conv state zeroed)
- **CUDA Graph invalidation**: graph destroyed and warmup counter reset on cache clear
- **Max tokens cap**: increased from 512 to 2048 (BF16 reduces overthinking)

#### Benchmark (LFM2.5-1.2B, BF16 native, 512 tokens)
| GPU | tok/s | vs vLLM |
|-----|-------|---------|
| RTX 4090 | 256 | 73% |
| **GH200 480GB** | **357** | **102%** |
| vLLM 0.17.1 (RTX 4090) | 352 | 100% |

CUDA Graph: 302 nodes captured.
Correctness: factual Q&A verified (Paris, Tokyo, Newton, H₂O, 8 spider legs).
Production: deployed on GH200 (192.222.50.16:8090).

---

## [0.8.0] - 2026-03-19

### CUDA Graphs — Full Kernel Launch Replay

Captures all 298 kernel launches as a CUDA graph during warmup,
then replays the entire decode step in ~50μs instead of individual launches.

#### What Changed
- **Global stream routing** (`ng_get_stream()`): all 71 kernels use configurable stream
- **Zero D2H copies**: all dynamic params (position, seq_len, kv_len) via GPU buffers
- **Graph-safe kernels**: `add_one_kernel` computes kv_len on GPU; `cudaMemcpyAsync` for D2D
- **Conv workspace**: pre-allocated BF16 buffers (no cudaMalloc during capture)
- **Per-layer paged caches**: properly set on decode context for graph compatibility

#### Benchmark (LFM2.5-1.2B, RTX 4090, 128 tokens)
| Engine | tok/s | vs vLLM |
|--------|-------|---------|
| HuggingFace 5.3 | 184 | 53% |
| NeuroGrid v0.7 (paged) | 216 | 62% |
| **NeuroGrid v0.8 (graphs)** | **225** | **64%** |
| vLLM 0.17.1 | 350 | 100% |

CUDA Graph: 298 nodes captured + replaying.
Golden Test: PASS — `<think>` (token 64400).
22% faster than HuggingFace transformers.

---

## [0.7.0] - 2026-03-19

### Paged KV Cache (PagedAttention)

Inspired by vLLM's PagedAttention — KV cache allocated in fixed-size blocks
instead of contiguous buffers. Eliminates memory fragmentation and enables
future continuous batching.

#### Block Allocator
- `BlockPool`: O(1) alloc/free via LIFO free list
- `PagedKVCacheManager`: per-sequence block table management
- `BlockSize=16` tokens per block, on-demand allocation
- `CalculateNumBlocks`: auto-size from available VRAM

#### Paged Attention CUDA Kernel
- `paged_attention_v1_kernel`: reads K/V via block_table indirection
- GQA support (num_heads != num_kv_heads)
- Online softmax with FP32 accumulation
- Works on CC >= 7.0 (RTX 2080 Ti and RTX 4090)
- Per-layer PagedKVCache (one cache per attention layer)

#### Benchmark (LFM2.5-1.2B, RTX 4090, 128 tokens)
| Config | tok/s | vs vLLM |
|--------|-------|---------|
| Contiguous KV | 204 | 58% |
| **Paged KV** | **216** | **62%** |
| vLLM 0.17.1 | 350 | 100% |

### Fixed
- CUDA Graph capture removed (caused stream corruption with paged attention)
- GPU-resident decode path disabled when paged cache active (incompatible)

---

## [0.6.0] - 2026-03-19

### Performance Optimizations

#### Flash Decode Attention Kernel
- Custom CUDA kernel with online softmax (no full score materialization)
- Tiled KV access (FLASH_TILE_SIZE=32) with FP32 accumulation
- Works on all GPUs CC >= 7.0 (RTX 2080 Ti and 4090)
- Drop-in replacement for naive attention in decode path

#### Single CUDA Call Decode (DecodeContext)
- All 16 layers execute in a single C function call
- Eliminates ~90 Go↔CUDA boundary crossings per token
- Ping-pong GPU buffers for zero-allocation layer traversal
- **Best measured: 230 tok/s (+9% over baseline)**

#### GPU-Resident Decode Path
- Hidden state stays on GPU between tokens (no Host↔Device copy)
- GPU→GPU embedding lookup via `SetHiddenFromGPU`
- LM head reads directly from GPU pointer via `ForwardFromGPU`
- Automatic fallback to host-copy path for distributed mode

#### Fused CUDA Kernels
- `cuda_silu_mul`: fused SwiGLU (SiLU + element-wise Mul) in single pass
- `cuda_add_rmsnorm`: fused Residual Add + RMSNorm in single pass
- Pre-allocated workspace buffers for attention and conv layers

#### Benchmark Results (LFM2.5-1.2B, RTX 4090, 128 tokens)
| Engine | tok/s | vs vLLM |
|--------|-------|---------|
| HuggingFace 5.3 | 201 | 57% |
| NeuroGrid v0.6 | 204 | 58% |
| vLLM 0.17.1 | 350 | 100% |

### Infrastructure
- `cublas_set_stream()` for future CUDA Graph support
- `cuda_flash_attention_supported()` runtime capability check
- Benchmark script: `scripts/benchmark_all.py`

---

## [0.5.0] - 2026-03-18

### Added

#### LFM2.5-1.2B-Thinking — First Non-Transformer Architecture
- Hybrid conv+attention architecture (10 conv + 6 attention layers, interleaved)
- BF16 CUDA kernels: RMSNorm, SiLU, Add, Mul, RoPE, GEMM (cublasGemmEx)
- Depthwise causal conv1d kernel (prefill + decode with FP32 state)
- LIV conv layer forward pass with FP16↔BF16 boundary conversion
- FP16-pure attention forward (no INT8 quantization) with QK LayerNorm
- Per-head RMSNorm on Q/K before RoPE (critical for accuracy)
- ChatML template with `<think>` reasoning token support
- GPT-2 byte-to-unicode encoding in BPE tokenizer
- Golden test data validated against HuggingFace reference (100% match)

#### Performance
- LFM2 1.2B: ~210 tok/s on RTX 4090 (matches HuggingFace transformers)
- First token matches golden reference: `<think>` (token 64400)

### Fixed
- GPT-2 tokenizer: byte-to-unicode mapping for control chars (\n → Ċ)
- Byte overflow in GPT-2 mapping table (infinite loop crash)
- SwiGLU intermediate_size: use 2/3 of block_ff_dim from config
- LFM2 tensor naming: out_proj, feed_forward, operator_norm, embedding_norm
- BF16/FP16 dtype boundary at conv↔attention layer transitions

---

## [0.4.0] - 2026-03-18

### Added

#### Distributed Demo Tooling
- **`demo_distributed.sh`** — One-command setup for distributed inference demo
- **Makefile targets**: `make demo`, `make demo-stream`, `make demo-stop`, `make demo-status`
- Supports RTX 2080 Ti (coordinator) + RTX 4090 (worker) topology

#### `--peer-vram-gb` Flag
- Override remote worker GPU VRAM when GPU info protocol times out
- Enables correct layer scheduling on heterogeneous clusters
- Example: `--peer-vram-gb 24` for RTX 4090 workers

#### LFM2 Architecture Support (branch `feat/lfm2-support`)
- **First non-Llama model**: LiquidAI LFM2.5-1.2B-Thinking (hybrid conv+attention)
- BF16 CUDA kernels: RMSNorm, SiLU, Add, Mul, RoPE, GEMM via cublasGemmEx
- Depthwise causal conv1d kernel (prefill + decode with FP32 state)
- LIV conv layer forward pass (conv_layer.cu)
- FP16-pure attention layer (no INT8 quantization) with configurable RoPE style
- FP16/BF16 conversion kernels at conv/attention layer boundaries
- ChatML template with `<think>` token support
- BPE tokenizer: array-of-arrays merges format, `<|im_end|>` EOS detection
- HuggingFace golden test data generator

#### Validated Distributed Configurations
| Model | GPUs | VRAM Used | Status |
|-------|------|-----------|--------|
| TinyLlama 1.1B | 2080 Ti + 4090 | ~3 GB | Tested |
| Mistral 7B Instruct | 2080 Ti + 4090 | ~14 GB | Tested |
| LFM2.5-1.2B-Thinking | 4090 only | ~3 GB | Tested (BF16) |

### Fixed
- Tokenizer: support merges as `[["a","b"],...]` format (LFM2/Qwen)
- Config: LFM2 SwiGLU intermediate_size adjustment (2/3 of block_ff_dim)
- Empty responses on some model configurations
- Chat template detection for ChatML-style models

---

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
- Continuous batching (process multiple requests simultaneously)
- Prefill batching (batch input tokens instead of 1-by-1)
- Prefix caching (share KV cache for common system prompts)
- INT4 quantization support
