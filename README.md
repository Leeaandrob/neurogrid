# NeuroGrid Engine

**Phase 1: CUDA Foundation Layer**

Distributed inference engine for LLMs using Pipeline Parallelism over federated P2P GPU networks.

## Overview

NeuroGrid is a custom inference engine built with Go + CUDA, designed for:
- Pipeline Parallelism across multiple nodes via libp2p
- INT8 quantization for efficient inference
- OpenAI-compatible API
- Federated P2P network of consumer GPUs (RTX 4090/5090)

This repository contains Phase 1: the CUDA foundation layer.

## Requirements

- **Go:** 1.21+
- **CUDA Toolkit:** 12.x
- **GPU:** NVIDIA GPU with Compute Capability 7.5+ (RTX 20/30/40/50 series)
- **OS:** Linux (tested on Ubuntu 22.04)

## Quick Start

```bash
# Clone and enter directory
cd neurogrid-engine

# Check CUDA installation
make check-cuda

# Build CUDA library and Go bindings
make build

# Run tests (requires GPU)
make test
```

## Project Structure

```
neurogrid-engine/
├── cmd/
│   └── test-layer/          # CLI for testing single layer
├── pkg/
│   └── types/               # Go types (Tensor, Config)
├── gpu/
│   ├── cuda/                # CUDA kernels
│   │   ├── memory.cu        # Memory management
│   │   ├── kernels.cu       # RMSNorm, SiLU, RoPE
│   │   ├── matmul.cu        # cuBLAS GEMM
│   │   ├── quantize.cu      # INT8 quantization
│   │   └── attention.cu     # Basic attention
│   ├── engine/              # C++ layer implementation
│   │   └── layer.cpp        # Transformer layer forward
│   └── bindings/            # CGO bindings
│       └── gpu.go           # Go ↔ CUDA interface
├── models/
│   └── llama/               # Llama model configs
├── scripts/
│   ├── generate_golden.py   # Generate PyTorch reference data
│   └── download_model.sh    # Download Llama weights
├── tests/                   # Go tests
│   └── golden/              # Reference outputs
├── Makefile
└── go.mod
```

## Building

### Prerequisites

1. Install CUDA Toolkit 12.x:
```bash
# Ubuntu
wget https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda_12.4.1_550.54.15_linux.run
sudo sh cuda_12.4.1_550.54.15_linux.run
```

2. Set environment:
```bash
export CUDA_PATH=/usr/local/cuda
export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
```

### Build Commands

```bash
# Build CUDA shared library only
make cuda

# Build everything (CUDA + Go)
make build

# Run all tests
make test

# Run specific test
make test-RMSNorm

# Run benchmarks
make bench

# Format code
make fmt

# Clean build artifacts
make clean
```

## Testing

### Unit Tests

```bash
# Run all tests
make test

# Run kernel tests
make test-RMSNorm
make test-SiLU
make test-GEMM
```

### Golden Data Validation

```bash
# Generate reference data from PyTorch
python scripts/generate_golden.py --kernels-only --output tests/golden/

# Run validation
make test-SingleLayerForward
```

### Benchmarks

```bash
# Run all benchmarks
make bench

# Expected results (RTX 4090):
# BenchmarkCGOOverhead: < 10μs
# BenchmarkLayerForward: ~1ms for 7B single token
```

## Implementation Details

### Kernels

| Kernel | Description | Precision |
|--------|-------------|-----------|
| RMSNorm | Root Mean Square Normalization | FP16 |
| SiLU | Sigmoid Linear Unit activation | FP16 |
| RoPE | Rotary Position Embeddings | FP16 |
| GEMM | Matrix multiplication (cuBLAS) | FP16/INT8 |
| Attention | Basic multi-head attention | FP16 |

### Quantization

- **Weight quantization:** INT8 per-column
- **Activation:** FP16
- **Dequantization:** On-the-fly during GEMM

### Memory Management

- Lazy allocation with device pinning
- FP32 ↔ FP16 conversion kernels
- KV cache with position-based updates

## Configuration

### Llama 7B Parameters

| Parameter | Value |
|-----------|-------|
| hidden_size | 4096 |
| intermediate_size | 11008 |
| num_layers | 32 |
| num_heads | 32 |
| head_dim | 128 |
| vocab_size | 32000 |
| max_seq_len | 4096 |

## Next Steps

- **Phase 2:** Multi-GPU support within single machine
- **Phase 3:** libp2p networking layer
- **Phase 4:** Pipeline parallelism protocol
- **Phase 5:** Full model inference
- **Phase 6:** Distributed KV cache
- **Phase 7:** Production API

## License

Apache 2.0

## References

- [Llama 2 Paper](https://arxiv.org/abs/2307.09288)
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
- [CGO Documentation](https://pkg.go.dev/cmd/cgo)
