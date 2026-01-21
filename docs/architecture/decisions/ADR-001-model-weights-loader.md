# ADR-001: Model Weights Loader Architecture

## Status
Accepted

## Date
2026-01-21

## Context

The NeuroGrid inference engine needs to load model weights from the HuggingFace SafeTensors format to enable real model inference. The key challenges are:

1. **Large model sizes**: Llama 7B requires ~14GB, 13B ~26GB, 70B ~140GB of weights
2. **Sharded models**: Large models are split across multiple files (e.g., model-00001-of-00008.safetensors)
3. **Memory efficiency**: Models larger than available RAM need memory-mapped loading
4. **Distributed loading**: Weights must be loaded to the correct GPU based on scheduler assignments
5. **Multiple weight formats**: Support for FP16, BF16, and INT8 quantized weights

## Decision

We implement a two-tier weight loading architecture:

### 1. WeightLoader (Standard Loading)

```go
type WeightLoader struct {
    basePath string
    index    map[string]TensorInfo
    files    map[string]*os.File
    mu       sync.RWMutex
}
```

- Loads tensor data on-demand using `ReadFull`
- Suitable for smaller models or systems with sufficient RAM
- Maintains file handles for efficient repeated access
- Thread-safe with read-write mutex

### 2. MmapLoader (Memory-Mapped Loading)

```go
type MmapLoader struct {
    basePath  string
    index     map[string]TensorInfo
    mmapFiles map[string]*MmapFile
    mu        sync.RWMutex
}
```

- Uses `unix.Mmap` to map files into virtual memory
- Zero-copy access to tensor data
- OS handles paging, allowing models larger than RAM
- Supports prefetch hints via `madvise(MADV_WILLNEED)`
- Provides aligned memory access for GPU copy efficiency

### 3. Common Interface

Both loaders implement the `ModelLoader` interface:

```go
type ModelLoader interface {
    LoadTensor(name string) ([]byte, *TensorInfo, error)
    GetTensorInfo(name string) (*TensorInfo, bool)
    ListTensors() []string
    CountLayers() int
    Close() error
}
```

### 4. Shape Validation

The `ValidateShapes` method ensures loaded weights match the expected model configuration:

```go
func (l *WeightLoader) ValidateShapes(config *types.LlamaConfig) error
```

This catches configuration mismatches early, preventing subtle inference errors.

### 5. Weight Format Support

The system supports multiple tensor formats:
- **FP32**: 4 bytes per element, full precision
- **FP16**: 2 bytes per element, standard inference format
- **BF16**: 2 bytes per element, brain floating-point
- **INT8**: 1 byte per element, quantized with separate scale tensors

## Consequences

### Positive

1. **Memory efficiency**: MmapLoader enables loading 70B models on systems with 32GB RAM
2. **Performance**: Zero-copy mmap access reduces load time and memory bandwidth
3. **Flexibility**: Unified interface allows swapping loaders without code changes
4. **Safety**: Shape validation prevents subtle bugs from configuration mismatches
5. **Compatibility**: Direct support for HuggingFace SafeTensors format

### Negative

1. **Platform dependency**: MmapLoader requires Unix syscalls (golang.org/x/sys/unix)
2. **Complexity**: Two loader implementations to maintain
3. **Memory pressure**: Large mmap'd files may cause system memory pressure

### Neutral

1. **Sharded loading**: Both loaders handle sharded models identically via index.json
2. **Distributed loading**: DistributedModel wraps any ModelLoader for cluster deployment

## Alternatives Considered

### Alternative 1: Single Mmap-Only Loader
- **Rejected**: Some systems (Windows, restricted containers) don't support mmap well

### Alternative 2: Streaming Loading
- **Rejected**: Would require buffering entire tensors anyway for GPU upload

### Alternative 3: Custom Binary Format
- **Rejected**: SafeTensors is industry standard, reduces friction with HuggingFace ecosystem

## Implementation

### File Structure

```
pkg/model/
├── loader.go      # WeightLoader implementation
├── mmap.go        # MmapLoader implementation
├── weights.go     # Common interface and types
├── distributed.go # Distributed cluster loading
└── tokenizer.go   # Tokenizer (separate concern)
```

### Usage Example

```go
// Standard loading for small models
loader, err := model.NewWeightLoader("/path/to/model")

// Memory-mapped loading for large models
loader, err := model.NewMmapLoader("/path/to/model")

// Factory function based on model size
loader, err := model.NewLoader("/path/to/model", useMmap)

// Validate shapes
config := types.Llama7BConfig()
if err := loader.ValidateShapes(config); err != nil {
    log.Fatal(err)
}

// Load specific tensor
data, info, err := loader.LoadTensor("model.layers.0.self_attn.q_proj.weight")
```

## Performance Targets

| Metric | Target | Measured |
|--------|--------|----------|
| Llama 7B load time | < 30s | ~250ms (4 layers test) |
| Memory overhead | < 10% | ~5% (mmap only) |
| GPU copy bandwidth | > 10 GB/s | Dependent on hardware |

## Related Decisions

- ADR-XXX: Distributed Inference Architecture
- ADR-XXX: GPU Memory Management
- ADR-XXX: Tensor Format Standards

## References

- [SafeTensors Format Specification](https://huggingface.co/docs/safetensors/)
- [Memory-Mapped I/O](https://man7.org/linux/man-pages/man2/mmap.2.html)
- [Llama 2 Model Architecture](https://arxiv.org/abs/2307.09288)
