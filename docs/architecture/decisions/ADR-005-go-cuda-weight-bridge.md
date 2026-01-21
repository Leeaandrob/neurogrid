# ADR-005: Go → CUDA Inference Bridge Implementation

## Status

Accepted

## Date

2026-01-21

## Context

The NeuroGrid inference engine needs to execute transformer layers on GPU with weights managed by the Go runtime. This requires a bridge between Go's SafeTensors weight loading and CUDA's GPU memory management for efficient inference.

### Requirements

1. Transfer FP16 weights from Go to GPU with INT8 quantization
2. Implement GPU-accelerated layer execution with KV caching
3. Provide GPU embedding lookup for token-to-hidden conversion
4. Implement LM head matmul for logits computation
5. Support full inference pipeline from Chat Completions API to coherent text

### Decision Drivers

- Go's memory management differs from CUDA's device memory model
- INT8 quantization reduces memory footprint and increases throughput
- CGO provides the interface layer but requires careful memory management
- Performance requires minimizing host-device memory transfers

### Constraints

- Must work with existing Go weight loading from SafeTensors
- Must integrate with existing Engine and LayerExecutor interface
- Must support multi-GPU context from existing bindings

## Decision

We implement a layered architecture with clear separation between Go data structures and CUDA device memory, using CGO bindings for weight transfer and kernel execution.

### Architecture

```
pkg/inference/
├── engine.go           # Core inference engine (existing)
├── engine_gpu.go       # GPU initialization and pipeline
├── cuda_executor.go    # CUDALayerExecutor implementing LayerExecutor
├── gpu_embeddings.go   # GPU embedding table with lookup
├── gpu_lmhead.go       # GPU LM head matrix multiplication

gpu/bindings/
├── gpu.go              # CUDA CGO bindings
├── gpu_stub.go         # Non-CUDA stubs for testing

gpu/cuda/
├── cuda_layer_forward.cu    # Layer forward kernel
├── cuda_weights.cu          # Weight transfer with quantization
├── cuda_quantize.cu         # FP16→INT8 per-column quantization
```

### Key Components

1. **CUDALayerExecutor**: Implements `LayerExecutor` interface
   - Manages per-layer GPU weights via `LayerWeights` struct
   - Creates and manages KV caches per layer
   - Executes forward pass via CUDA kernels
   - Thread-safe with RWMutex protection

2. **CreateLayerWeightsFromHost**: Weight transfer function
   - Takes FP16 byte slices from Go (loaded from SafeTensors)
   - Quantizes projection matrices to INT8 with per-column scales
   - Keeps norm weights in FP16 (no quantization)
   - Returns opaque `*LayerWeights` pointer

3. **GPUEmbeddings**: Token embedding lookup
   - Uploads full embedding table to GPU memory
   - Provides `Lookup(tokenID)` returning GPU pointer
   - Provides `LookupToHost(tokenID)` copying to host

4. **GPULMHead**: Language model head
   - Uploads [hidden_size, vocab_size] weight matrix to GPU
   - `Forward(hidden)` computes logits via cuBLAS GEMM
   - Returns FP32 logits on host for sampling

### Weight Format

**Projection Matrices (INT8 quantized)**:
```
Q/K/V/O Proj: [out_dim, in_dim] INT8 + [out_dim] FP32 scales
Gate/Up Proj: [intermediate, hidden] INT8 + scales
Down Proj:    [hidden, intermediate] INT8 + scales
```

**Norm Weights (FP16, no quantization)**:
```
AttnNorm: [hidden_size] FP16
FFNNorm:  [hidden_size] FP16
```

### Stub Implementation

For builds without CUDA (`!cuda` build tag), `gpu_stub.go` provides matching function signatures that:
- Return placeholder pointers for allocations
- Simulate memory operations with host-side byte slices
- Allow full test suite execution without GPU hardware

## Alternatives Considered

### Option 1: Keep All Weights in FP16

**Description**: Transfer FP16 weights directly without quantization

**Pros**:
- Simpler implementation
- No quantization error

**Cons**:
- 2x memory usage compared to INT8
- Lower throughput on tensor cores

**Why Not Chosen**: Memory efficiency critical for large models

### Option 2: Python-Based Weight Loading

**Description**: Use Python/PyTorch for weight loading, pass to CUDA

**Pros**:
- Mature tooling for SafeTensors
- Easy FP16/INT8 conversion

**Cons**:
- Requires Python runtime
- Complex interprocess communication
- Additional dependency

**Why Not Chosen**: Want pure Go+CUDA stack

### Option 3: Direct Memory Mapping

**Description**: Memory-map weights directly to GPU (Unified Memory)

**Pros**:
- Zero-copy for large models
- Simpler programming model

**Cons**:
- Unified Memory has performance overhead
- Not all GPUs support full UVM features
- Less control over transfer timing

**Why Not Chosen**: Explicit memory management gives better performance control

## Consequences

### Positive

- Clean separation between Go weight management and GPU execution
- INT8 quantization reduces memory footprint by ~2x
- Stub implementation enables testing without GPU
- LayerExecutor interface allows swapping CPU/GPU execution
- Full pipeline from API to text output

### Negative

- CGO overhead for each binding call
- Quantization introduces small accuracy loss
- Complexity of maintaining stub/CUDA parity
- Requires CUDA toolkit for production builds

### Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Quantization accuracy loss | Medium | Medium | Per-column scaling, benchmark against FP16 |
| CGO memory leaks | Low | High | Explicit Free calls, defer cleanup patterns |
| Stub/CUDA signature drift | Medium | Medium | Single interface definition, build both targets in CI |
| KV cache memory exhaustion | Low | High | Track cache size, implement eviction |

## Implementation Notes

### Files Added

- `pkg/inference/cuda_executor.go`: CUDALayerExecutor (224 lines)
- `pkg/inference/gpu_embeddings.go`: GPUEmbeddings (102 lines)
- `pkg/inference/gpu_lmhead.go`: GPULMHead (242 lines)
- `tests/e2e/real_inference_test.go`: E2E tests (600+ lines)

### Tests Added

Coverage for all acceptance criteria:
1. `TestCUDALayerExecutor_Interface` - AC1
2. `TestCUDALayerExecutor_LoadLayer` - AC1
3. `TestCUDALayerExecutor_Forward` - AC1
4. `TestCreateLayerWeightsFromHost` - AC2
5. `TestCreateLayerWeightsFromHost_Mock` - AC2 (mock data)
6. `TestGPUEmbeddings_Lookup` - AC3
7. `TestGPUEmbeddings_OutOfRange` - AC3
8. `TestGPULMHead_Forward` - AC4
9. `TestEngine_InitializeGPU` - AC5
10. `TestRealInference_*` - AC5/AC6 (requires model)

### Build Commands

```bash
# With CUDA
CGO_ENABLED=1 go build -tags cuda ./...

# Without CUDA (stub mode)
go build ./...

# Run tests with CUDA
go test -tags cuda ./tests/e2e/
```

### Future Work

1. **Batch Processing**: Support batch size > 1 for throughput
2. **Streaming Generation**: Token-by-token callback for SSE
3. **Flash Attention**: Replace basic attention with FlashAttention-2
4. **Multi-GPU Inference**: Tensor parallelism across devices

## References

- [NVIDIA INT8 Inference](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-601/tensorrt-developer-guide/index.html#int8-inference)
- [CGO Documentation](https://pkg.go.dev/cmd/cgo)
- [PRP-05: Go→CUDA Weight Bridge](../prps/05-go-cuda-weight-bridge.md)
