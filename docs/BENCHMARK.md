# NeuroGrid Distributed Inference Benchmark

## Overview

This document tracks performance benchmarks for the NeuroGrid distributed inference engine across different model sizes and hardware configurations.

## Test Environment

### Hardware Configuration

| Node | GPU | VRAM | Role |
|------|-----|------|------|
| Coordinator (192.168.50.25) | NVIDIA RTX 4090 | 24GB | Orchestrates inference, executes local layers |
| Worker (192.168.50.130) | NVIDIA RTX 2080 Ti | 11GB | Executes assigned layers |

### Network
- Local network via Ethernet
- libp2p P2P communication
- mDNS peer discovery

---

## Benchmark Results

### TinyLlama 1.1B (2024-01-23)

**Model Configuration:**
- Parameters: 1.1B
- Layers: 22 transformer layers
- Hidden Size: 2048
- Vocab Size: 32,000
- Precision: BF16 → FP16

**Layer Distribution:**
- Coordinator (RTX 4090): 11 layers (odd)
- Worker (RTX 2080 Ti): 11 layers (even)

#### Performance Metrics

| Tokens | Total Time | Avg ms/token | ms/token (after 1st) | Throughput |
|--------|------------|--------------|----------------------|------------|
| 1      | 1.56s      | 1560ms       | -                    | 0.64 tps   |
| 10     | 2.50s      | 250ms        | 104ms                | 4.0 tps    |
| 20     | 3.50s      | 175ms        | 102ms                | 5.7 tps    |
| 50     | 6.03s      | 121ms        | 91ms                 | 8.3 tps    |

**Key Findings:**
- First token latency: ~1.5s (includes embedding lookup, initial forward pass)
- Sustained throughput: **90-100ms/token** (~10-11 tps)
- Network overhead per layer: ~2-3ms (estimated)

#### Comparison: Debug Logging Impact

| Metric | With Debug | Without Debug | Improvement |
|--------|------------|---------------|-------------|
| ms/token (20 tok) | ~120ms | ~102ms | 15% faster |
| Throughput | 5-8 tps | 10-11 tps | +40% |

---

### Llama 2 7B (Pending)

**Model Configuration:**
- Parameters: 7B
- Layers: 32 transformer layers
- Hidden Size: 4096
- Vocab Size: 32,000
- Precision: BF16 → FP16

**Expected Layer Distribution:**
- Coordinator (RTX 4090): ~20 layers
- Worker (RTX 2080 Ti): ~12 layers

*Results pending...*

---

## Performance Analysis

### Bottlenecks Identified

1. **Network Transfer**: ~4KB activation data per layer transfer
2. **Synchronous Execution**: Each layer waits for previous to complete
3. **First Token Latency**: Setup overhead dominates small requests

### Optimizations Applied

1. **PrefetchCoordinator**: Queues layer prefetch requests (fixed blocking issue)
2. **Removed Debug Logging**: 15-40% throughput improvement
3. **FP16 Weights**: Reduced memory bandwidth requirements

### Future Optimizations

1. **Pipeline Parallelism**: Overlap computation with network transfer
2. **Batched Inference**: Process multiple sequences simultaneously
3. **Tensor Parallelism**: Split large layers across GPUs
4. **Quantization**: INT8/INT4 for faster compute and lower memory

---

## How to Run Benchmarks

### Prerequisites
```bash
# Build with CUDA support
make build-coordinator

# Ensure worker is running on remote machine
# On worker (192.168.50.130):
./build/neurogrid --role performer --model ./models/tinyllama --model-name tinyllama
```

### Run Coordinator
```bash
export LD_LIBRARY_PATH=./build:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

./build/neurogrid \
  --role coordinator \
  --model ./models/tinyllama \
  --model-name tinyllama \
  --min-peers 1 \
  --http-port 8090
```

### Run Benchmark
```bash
# Single token (first token latency)
time curl -s http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "tinyllama", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 1}'

# Multi-token throughput
time curl -s http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "tinyllama", "messages": [{"role": "user", "content": "Tell me a story"}], "max_tokens": 50}'
```

---

## Version History

| Date | Version | Changes |
|------|---------|---------|
| 2024-01-23 | 0.1.0 | Initial benchmark with TinyLlama 1.1B |
| 2024-01-23 | 0.1.1 | Fixed PrefetchCoordinator blocking, removed debug logging |

---

## Notes

- All benchmarks performed on local network (low latency)
- Results may vary based on network conditions and GPU utilization
- First token latency includes tokenization and embedding lookup
