# NeuroGrid Engine

<p align="center">
  <strong>GPU-Accelerated Distributed LLM Inference Engine</strong>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#benchmarks">Benchmarks</a> •
  <a href="#api">API</a> •
  <a href="#license">License</a>
</p>

---

NeuroGrid is a high-performance inference engine for Large Language Models (LLMs), built from scratch in **Go + CUDA**. Designed for distributed inference across federated GPU networks using Pipeline Parallelism.

## Features

### v0.1.0 - Current Release

- **GPU-Accelerated Inference** - Full CUDA/cuBLAS pipeline for FP16 inference
- **Llama Architecture Support** - TinyLlama-1.1B, Llama-2, and compatible models
- **BF16 → FP16 Conversion** - Automatic weight conversion during loading
- **SafeTensors Support** - Native loading of HuggingFace model format
- **SentencePiece Tokenizer** - Full tokenization pipeline
- **OpenAI-Compatible API** - Drop-in replacement for `/v1/chat/completions`
- **P2P Networking Foundation** - libp2p with mDNS discovery
- **Single-GPU Local Mode** - Standalone operation with `--min-peers 0`

### GPU Kernels

| Kernel | Description | Precision |
|--------|-------------|-----------|
| GEMM | Matrix multiplication (cuBLAS) | FP16/INT8 |
| RMSNorm | Root Mean Square Normalization | FP16 |
| SiLU | Sigmoid Linear Unit activation | FP16 |
| RoPE | Rotary Position Embeddings | FP16 |
| Attention | Multi-head attention | FP16 |
| Quantize | INT8 per-row quantization | INT8 |

## Requirements

- **Go:** 1.21+
- **CUDA Toolkit:** 11.x or 12.x
- **cuBLAS:** Included with CUDA Toolkit
- **GPU:** NVIDIA GPU with Compute Capability 7.0+ (RTX 20/30/40/50 series)
- **OS:** Linux (tested on Ubuntu 22.04/24.04)
- **RAM:** 8GB+ (for model loading)
- **VRAM:** 6GB+ (for TinyLlama-1.1B)

## Quick Start

### 1. Install Dependencies

```bash
# Ubuntu/Debian
sudo apt-get install build-essential

# CUDA Toolkit (if not installed)
# Download from: https://developer.nvidia.com/cuda-downloads
```

### 2. Build

```bash
# Clone repository
git clone https://github.com/Leeaandrob/neurogrid.git
cd neurogrid

# Build CUDA library and Go binaries
make build

# Verify build
./build/neurogrid --help
```

### 3. Download Model

```bash
# Download TinyLlama-1.1B (recommended for testing)
./scripts/download_model.sh tinyllama

# Or manually:
# Place SafeTensors files in ./models/tinyllama/
```

### 4. Run Server

```bash
# Start inference server (single GPU, local mode)
make run-coordinator

# Or with explicit options:
LD_LIBRARY_PATH=./build:/usr/local/cuda/lib64 \
  ./build/neurogrid \
  --http-port 8080 \
  --gpu 0 \
  --model ./models/tinyllama \
  --model-name tinyllama \
  --min-peers 0
```

### 5. Test Inference

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tinyllama",
    "messages": [{"role": "user", "content": "The capital of France is"}],
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

## Benchmarks

### Performance Metrics (TinyLlama-1.1B, RTX GPU)

| Metric | Value | Description |
|--------|-------|-------------|
| **TTFT** | ~1.3s (P50) | Time To First Token |
| **TPOT** | ~145ms/token | Time Per Output Token |
| **ITL** | ~145ms | Inter-Token Latency |
| **TPS** | ~6.9 tokens/sec | Generation throughput |

### Run Benchmarks

```bash
# Build benchmark tool
go build -o build/benchmark ./tools/benchmark/

# Run benchmark
./build/benchmark -n 100 -c 1 -max-tokens 50

# Options:
#   -n            Number of requests (default: 100)
#   -c            Concurrency (default: 1)
#   -max-tokens   Max tokens to generate (default: 50)
#   -prompt       Prompt length: short, medium, long
#   -json         Output as JSON
#   -warmup       Warmup requests (default: 3)
```

### Sample Output

```
╔══════════════════════════════════════════════════════════════╗
║                     Benchmark Results                        ║
╠══════════════════════════════════════════════════════════════╣
║ Total Requests:     100                                      ║
║ Successful:         100                                      ║
║ Failed:             0                                        ║
╠══════════════════════════════════════════════════════════════╣
║                    TTFT (Time To First Token)                ║
╠══════════════════════════════════════════════════════════════╣
║ P50:        1307.00 ms                                       ║
║ P90:        2208.00 ms                                       ║
║ P99:        2790.00 ms                                       ║
╠══════════════════════════════════════════════════════════════╣
║                       Throughput                             ║
╠══════════════════════════════════════════════════════════════╣
║ Requests/sec:       7.17                                     ║
║ Output Tokens/sec:  92.83                                    ║
║ Generation TPS:     6.86                                     ║
╚══════════════════════════════════════════════════════════════╝
```

## API

### OpenAI-Compatible Endpoint

```
POST /v1/chat/completions
```

### Request Format

```json
{
  "model": "tinyllama",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
  "max_tokens": 100,
  "temperature": 0.7
}
```

### Response Format

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1769027418,
  "model": "tinyllama",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Hello! How can I help you today?"
    },
    "finish_reason": "eos"
  }]
}
```

## Project Structure

```
neurogrid-engine/
├── cmd/
│   ├── neurogrid/           # Main server binary
│   ├── download/            # Model download utility
│   └── worker/              # Worker node (distributed mode)
├── api/                     # HTTP API server
├── gpu/
│   ├── cuda/                # CUDA kernels (.cu/.h)
│   ├── engine/              # C++ layer implementation
│   └── bindings/            # CGO Go ↔ CUDA bindings
├── pkg/
│   ├── inference/           # Inference engine
│   ├── model/               # Model loading & tokenizer
│   ├── scheduler/           # Layer scheduling
│   ├── transport/           # GPU transport layer
│   └── types/               # Core types (Tensor, Config)
├── p2p/                     # libp2p networking
├── tests/                   # Test suites
├── tools/
│   └── benchmark/           # Benchmark tool
├── docs/                    # Documentation
├── Makefile
└── go.mod
```

## Configuration

### Supported Models

| Model | Parameters | VRAM Required |
|-------|------------|---------------|
| TinyLlama-1.1B | 1.1B | ~3GB |
| Llama-2-7B | 7B | ~14GB |
| Llama-2-13B | 13B | ~26GB |

### Environment Variables

```bash
export CUDA_PATH=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:./build:$LD_LIBRARY_PATH
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--http-port` | 8080 | HTTP API port |
| `--p2p-port` | 9000 | P2P networking port |
| `--gpu` | 0 | GPU device ID |
| `--model` | - | Path to model directory |
| `--model-name` | - | Model identifier |
| `--min-peers` | 0 | Minimum peers (0 = local only) |
| `--cors` | true | Enable CORS headers |

## Building from Source

### Prerequisites

```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# Check Go installation
go version
```

### Build Commands

```bash
# Full build (CUDA + Go)
make build

# CUDA library only
make cuda

# Run tests
make test

# Run specific test
CGO_LDFLAGS="-L$(pwd)/build -lgpu_engine" \
  go test -v -tags cuda ./tests/... -run TestGEMMFP16

# Clean
make clean
```

## Documentation

- [Architecture Overview](docs/architecture/)
- [Architecture Decision Records](docs/architecture/decisions/)
- [Product Requirements](docs/prps/)
- [Changelog](CHANGELOG.md)

## Roadmap

### v0.2.0 (Planned)
- [ ] Streaming responses (SSE)
- [ ] KV cache optimization
- [ ] Flash Attention integration

### v0.3.0 (Planned)
- [ ] CUDA graphs optimization
- [ ] Speculative decoding
- [ ] Multi-GPU distributed inference

### Future
- [ ] INT4 quantization
- [ ] Tensor parallelism
- [ ] Production monitoring (Prometheus/Grafana)

## License

**Source Available License with Academic & Educational Use Grant**

- ✅ Free for students, researchers, and academic use
- ✅ Free for personal learning and non-commercial projects
- ✅ Free for non-commercial open source projects
- ❌ Commercial use requires a license

For commercial licensing inquiries, contact: **leandrobar93@gmail.com**

See [LICENSE](LICENSE) for full terms.

## Contributing

Contributions are welcome for non-commercial purposes. By contributing, you agree that your contributions will be subject to the project's license terms.

## Acknowledgments

- [cuBLAS](https://developer.nvidia.com/cublas) - NVIDIA's GPU-accelerated BLAS library
- [libp2p](https://libp2p.io/) - Modular peer-to-peer networking stack
- [SentencePiece](https://github.com/google/sentencepiece) - Tokenization library
- [SafeTensors](https://github.com/huggingface/safetensors) - Safe tensor serialization

## Contact

- **Author:** Leandro Barbosa
- **Email:** leandrobar93@gmail.com
- **Commercial Inquiries:** leandrobar93@gmail.com

---

<p align="center">
  <strong>NeuroGrid Engine v0.1.0</strong><br>
  Built with ❤️ and CUDA
</p>
