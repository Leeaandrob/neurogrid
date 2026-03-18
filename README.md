# NeuroGrid Engine

<p align="center">
  <strong>GPU-Accelerated Distributed LLM Inference Engine</strong>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#supported-models">Models</a> •
  <a href="#api">API</a> •
  <a href="#distributed-mode">Distributed</a> •
  <a href="#benchmarks">Benchmarks</a>
</p>

---

NeuroGrid is a high-performance inference engine for Large Language Models (LLMs), built from scratch in **Go + CUDA**. Designed for both single-GPU and distributed inference across multiple machines.

## Quick Start

```bash
# 1. Download a model
make download-tinyllama          # Small model for testing (~2.2GB)

# 2. Build and run
make run

# 3. Test the API
curl -X POST http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "tinyllama", "messages": [{"role": "user", "content": "Hello!"}]}'
```

That's it! The server auto-detects the model and starts on port `8090`.

## Requirements

| Requirement | Version |
|-------------|---------|
| Go | 1.21+ |
| CUDA Toolkit | 11.x or 12.x |
| GPU | NVIDIA with Compute 7.0+ (RTX 20/30/40/50) |
| OS | Linux (Ubuntu 22.04/24.04) |

## Supported Models

| Model | Size | VRAM | Download Command |
|-------|------|------|------------------|
| TinyLlama 1.1B | ~2.2GB | ~3GB | `make download-tinyllama` |
| Mistral 7B Instruct | ~15GB | ~14GB | `make download-mistral7b-instruct` |
| Llama 2 7B | ~13GB | ~14GB | `make download-llama7b` ¹ |
| Llama 2 13B | ~26GB | ~26GB | `make download-llama13b` ¹ |
| LFM2.5-1.2B-Thinking | ~2.5GB | ~3GB | `make download-lfm2-thinking` ² |

¹ Requires `HF_TOKEN` environment variable (get token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens))

² Hybrid conv+attention architecture (branch `feat/lfm2-support`). Requires Ampere+ GPU (RTX 3090/4090) for BF16.

### Download Any HuggingFace Model

```bash
# Generic download - works with any public model
make download REPO=mistralai/Mistral-Nemo-Instruct-2407
make download REPO=Qwen/Qwen2.5-7B-Instruct
make download REPO=google/gemma-2-9b-it

# For gated models (Llama, etc.)
export HF_TOKEN=your_token
make download REPO=meta-llama/Llama-3.3-70B-Instruct
```

## Running the Server

### Single Node (Recommended for most users)

```bash
# Auto-detect model and run
make run

# Or run with specific model
make run-mistral      # Mistral 7B Instruct
make run-tinyllama    # TinyLlama 1.1B
make run-llama7b      # Llama 2 7B

# Custom configuration
make run HTTP_PORT=8080 GPU_ID=1 LOG_LEVEL=debug
```

### Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `HTTP_PORT` | 8090 | API server port |
| `GPU_ID` | 0 | CUDA device ID |
| `LOG_LEVEL` | info | Log verbosity (debug, info, warn, error) |

## API

### OpenAI-Compatible Endpoint

```
POST /v1/chat/completions
```

### Request

```bash
curl -X POST http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral-7b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### Response

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1769027418,
  "model": "mistral-7b",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "The capital of France is Paris."
    },
    "finish_reason": "eos"
  }]
}
```

### Health Check

```bash
curl http://localhost:8090/health
# {"status":"healthy","model":"mistral-7b","timestamp":1769027418,"version":"1.0.0"}
```

## Distributed Mode

For multi-GPU inference across machines using Pipeline Parallelism.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       Coordinator (GH200)                        │
│  - HTTP API endpoint                                             │
│  - Orchestrates inference                                        │
│  - Layers 0-13 (local)                                           │
└───────────────────────────┬─────────────────────────────────────┘
                            │ P2P (libp2p)
          ┌─────────────────┴─────────────────┐
          │                                   │
┌─────────┴───────────┐          ┌────────────┴──────────┐
│   Worker (RTX 4090) │          │   Worker (RTX 2080)   │
│   Layers 14-26      │          │   Layers 27-39        │
└─────────────────────┘          └───────────────────────┘
```

### Mode 1: Workers with Local Models (Recommended)

When workers have the model downloaded locally, skip weight transfer for faster startup.

**Important**: Use `-wait-for-assignment` on workers so they only load layers assigned by the coordinator. This is critical for heterogeneous GPU clusters to prevent OOM errors.

```bash
# Coordinator (GH200 - start first)
LD_LIBRARY_PATH=./build ./build/neurogrid \
  -model models/mistral-nemo-instruct-2407 \
  -http-port 8090 \
  -p2p-port 9000 \
  -gpu 0 \
  -min-peers 2 \
  -max-seq-len 4096 \
  -skip-weight-transfer \
  -disable-mdns

# Worker 1 (RTX 4090 - connect via bootstrap)
LD_LIBRARY_PATH=./build ./build/worker \
  -bootstrap /ip4/<COORDINATOR_IP>/tcp/9000/p2p/<COORDINATOR_PEER_ID> \
  -model models/mistral-nemo-instruct-2407 \
  -gpu 0 \
  -port 9001 \
  -wait-for-assignment

# Worker 2 (RTX 2080 - connect via bootstrap)
LD_LIBRARY_PATH=./build ./build/worker \
  -bootstrap /ip4/<COORDINATOR_IP>/tcp/9000/p2p/<COORDINATOR_PEER_ID> \
  -model /path/to/models/mistral-nemo-instruct-2407 \
  -gpu 0 \
  -port 9002 \
  -wait-for-assignment
```

**What happens:**
1. Workers connect and report their GPU info (VRAM, name) to coordinator
2. Coordinator computes layer assignments based on actual VRAM of each GPU
3. Coordinator sends layer requests to workers
4. Workers load only their assigned layers from local storage

### Mode 2: Auto-Discovery (LAN only)

For simple LAN setups, workers can be discovered automatically via mDNS:

```bash
# Machine 1: Worker with GPU 0
make run-worker GPU_ID=0 P2P_PORT=9001

# Machine 2: Worker with GPU 0
make run-worker GPU_ID=0 P2P_PORT=9002

# Machine 3: Coordinator (connects to workers automatically via mDNS)
make run-coordinator MIN_PEERS=2
```

### Distributed Mode Flags

#### Coordinator Flags

| Flag | Default | Description |
|------|---------|-------------|
| `-min-peers` | 0 | Minimum workers to wait for (0 = local only) |
| `-skip-weight-transfer` | false | Skip P2P weight distribution (workers have local models) |
| `-disable-mdns` | false | Disable mDNS discovery (use explicit bootstrap) |
| `-max-seq-len` | 4096 | Max sequence length (caps KV cache size) |
| `-peer-vram-gb` | 0 | Override worker GPU VRAM in GB (workaround for GPU info timeout) |

#### Worker Flags

| Flag | Default | Description |
|------|---------|-------------|
| `-bootstrap` | "" | Coordinator address for explicit connection |
| `-model` | "" | Path to local model weights |
| `-wait-for-assignment` | false | **Critical for heterogeneous clusters**: Wait for coordinator to assign layers before loading |
| `-max-seq-len` | 4096 | Max sequence length (caps KV cache size) |
| `-port` | 9000 | P2P listen port |
| `-gpu` | 0 | GPU device ID |

### Heterogeneous GPU Support

NeuroGrid supports clusters with different GPU types (e.g., GH200 + RTX 4090 + RTX 2080). The system:

1. **GPU Info Protocol**: Workers report actual VRAM to coordinator on connect
2. **VRAM-Aware Scheduling**: Scheduler assigns layers based on each GPU's available memory
3. **On-Demand Loading**: With `-wait-for-assignment`, workers load only their assigned layers

This prevents OOM errors on smaller GPUs that would occur if all GPUs tried to load the same layers.

### Troubleshooting Distributed Mode

**Ghost Peer Issue**: If you see connections to unknown peers, use `-disable-mdns` and connect workers via explicit `-bootstrap` addresses.

**Out of Memory on Workers**:
- **Use `-wait-for-assignment`** on workers (most common fix)
- Reduce `-max-seq-len` to use less KV cache memory
- The scheduler automatically assigns fewer layers to GPUs with less VRAM

**Workers not receiving layer assignments**: Ensure coordinator has `-min-peers` set to the number of expected workers.

## Building from Source

### Prerequisites

```bash
# Verify CUDA installation
nvcc --version
nvidia-smi

# Verify Go installation
go version
```

### Build Commands

```bash
# Build everything (CUDA library + binaries)
make build-all

# Build CUDA library only
make cuda

# Build specific binary
make build-coordinator
make build-worker

# Clean build artifacts
make clean
```

### Run Tests

```bash
make test           # CUDA tests
make test-e2e       # End-to-end tests (no CUDA required)
make test-all       # All tests
```

## Benchmarks

### Performance (Mistral 7B on RTX 4090)

| Metric | Value |
|--------|-------|
| TTFT (Time to First Token) | ~1.3s |
| Generation Speed | ~6.9 tokens/sec |
| GPU Memory | ~14GB |

### Run Benchmarks

```bash
make bench-quick     # Quick benchmark
make bench-full      # Full benchmark suite
```

## Project Structure

```
neurogrid-engine/
├── cmd/
│   ├── neurogrid/     # Main server (coordinator)
│   ├── worker/        # Distributed worker node
│   └── download/      # Model download utility
├── gpu/
│   ├── cuda/          # CUDA kernels
│   ├── engine/        # C++ layer implementation
│   └── bindings/      # Go ↔ CUDA bindings
├── pkg/
│   ├── inference/     # Inference engine
│   ├── model/         # Model loading & tokenizer
│   ├── scheduler/     # Layer distribution
│   └── huggingface/   # HF model downloader
├── api/               # HTTP API server
├── p2p/               # libp2p networking
└── Makefile
```

## Troubleshooting

### CUDA Library Not Found

If you see `libgpu_engine.so: cannot open shared object file`:

```bash
# Option 1: Use make (handles LD_LIBRARY_PATH automatically)
make run

# Option 2: Set LD_LIBRARY_PATH manually
export LD_LIBRARY_PATH=$(pwd)/build:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
./build/neurogrid --model ./models/tinyllama --model-name tinyllama
```

### No Model Found

```bash
# Check available models
ls ./models/

# Download a model
make download-tinyllama
```

### GPU Out of Memory

Try a smaller model:
```bash
make download-tinyllama
make run-tinyllama
```

## Help

```bash
make help   # Show all available commands
```

## License

**Source Available License with Academic & Educational Use Grant**

- ✅ Free for students, researchers, and academic use
- ✅ Free for personal learning and non-commercial projects
- ❌ Commercial use requires a license

Contact: **leandrobar93@gmail.com**

## Acknowledgments

- [cuBLAS](https://developer.nvidia.com/cublas) - NVIDIA's GPU-accelerated BLAS
- [libp2p](https://libp2p.io/) - Peer-to-peer networking
- [HuggingFace](https://huggingface.co/) - Model hub

---

<p align="center">
  <strong>NeuroGrid Engine v0.4.0</strong><br>
  Built with Go + CUDA
</p>
