# PRP: Model Weights Loader (Safetensors/Llama)

## Document Information

| Field | Value |
|-------|-------|
| **Feature Name** | Model Weights Loader |
| **PRP Version** | 1.0 |
| **Created** | 2026-01-21 |
| **Priority** | Short-Term (Critical Path) |
| **Status** | Ready for Implementation |
| **Confidence Score** | 9/10 |

---

## Discovery Summary

### Initial Task Analysis

Implement weight loading from safetensors format (HuggingFace standard) to enable real model inference. Currently the engine has all CUDA kernels and inference pipeline ready, but no way to load actual model weights.

### User Clarifications Received

- **Question**: Which model formats to support?
- **Answer**: Safetensors (primary), GGUF (future)
- **Impact**: Focus on safetensors implementation first

### Missing Requirements Identified

- Memory-mapped loading for large models
- Sharded weight support (model split across multiple files)
- Weight distribution to assigned layers/peers

---

## Goal

Implement `pkg/model/loader.go` that can load Llama 7B/13B/70B weights from safetensors format and distribute them to the appropriate GPU devices based on scheduler assignments.

## Why

- **Critical path**: Cannot run real inference without weights
- **Memory efficiency**: Need memory-mapped loading for 70B models
- **Distribution**: Weights must be loaded to correct peer/GPU per layer assignment

## What

### Success Criteria

- [ ] Load Llama 7B safetensors weights in < 30s
- [ ] Support sharded models (model-00001-of-00002.safetensors pattern)
- [ ] Memory-map weights for models > available RAM
- [ ] Distribute weights to correct GPU per scheduler assignment
- [ ] Support INT8 and FP16 weight formats
- [ ] Validate weight shapes against model config

---

## All Needed Context

### Research Phase Summary

- **Codebase patterns found**: `pkg/types/tensor.go`, `pkg/scheduler/scheduler.go`
- **External research needed**: Yes - safetensors format specification
- **Knowledge gaps identified**: Sharded loading, memory mapping in Go

### Documentation & References

```yaml
- url: https://huggingface.co/docs/safetensors/
  why: Safetensors format specification and Go library

- file: pkg/types/tensor.go
  why: Tensor structure to populate with loaded weights

- file: pkg/types/config.go
  why: Model configuration (layer sizes, hidden dim)

- file: pkg/scheduler/scheduler.go
  why: Layer assignments to know where to load weights

- file: gpu/bindings/gpu.go
  why: CGO interface to allocate GPU memory
```

### Current Codebase tree

```
pkg/
├── types/
│   ├── tensor.go      # Tensor type with Device field
│   └── config.go      # LlamaConfig with layer dimensions
├── scheduler/
│   └── scheduler.go   # LayerAssignment with peer/device mapping
├── inference/
│   └── engine.go      # Needs weights to run forward pass
└── model/             # NEW - to be created
    ├── loader.go      # Main loading logic
    ├── safetensors.go # Safetensors format parser
    └── distributed.go # Distributed weight loading
```

### Desired Codebase tree

```
pkg/model/
├── loader.go          # ModelLoader interface and factory
├── safetensors.go     # Safetensors format implementation
├── distributed.go     # DistributedLoader for multi-peer loading
├── mmap.go            # Memory-mapped file support
└── weights.go         # Weight tensor management
```

### Known Gotchas

```go
// CRITICAL: Safetensors uses little-endian byte order
// CRITICAL: Sharded models have index.json with tensor→file mapping
// CRITICAL: Memory-map requires proper alignment for GPU copy
// CRITICAL: INT8 weights have separate scale tensors per layer
```

---

## Implementation Blueprint

### Data Models

```go
// pkg/model/loader.go

type ModelLoader interface {
    // Load loads all model weights
    Load(ctx context.Context, path string) (*LoadedModel, error)

    // LoadLayer loads weights for a specific layer
    LoadLayer(ctx context.Context, path string, layerID int) (*LayerWeights, error)

    // Close releases resources
    Close() error
}

type LoadedModel struct {
    Config       *types.LlamaConfig
    Embedding    *types.Tensor  // token embeddings
    Layers       []*LayerWeights
    OutputNorm   *types.Tensor  // final RMSNorm
    LMHead       *types.Tensor  // output projection
}

type LayerWeights struct {
    LayerID      int
    AttnNorm     *types.Tensor  // attention RMSNorm
    QProj        *types.Tensor  // query projection
    KProj        *types.Tensor  // key projection
    VProj        *types.Tensor  // value projection
    OProj        *types.Tensor  // output projection
    QScale       *types.Tensor  // INT8 scale for Q (optional)
    KScale       *types.Tensor  // INT8 scale for K (optional)
    VScale       *types.Tensor  // INT8 scale for V (optional)
    OScale       *types.Tensor  // INT8 scale for O (optional)
    FFNNorm      *types.Tensor  // FFN RMSNorm
    GateProj     *types.Tensor  // gate projection
    UpProj       *types.Tensor  // up projection
    DownProj     *types.Tensor  // down projection
    GateScale    *types.Tensor  // INT8 scale (optional)
    UpScale      *types.Tensor  // INT8 scale (optional)
    DownScale    *types.Tensor  // INT8 scale (optional)
}

// pkg/model/safetensors.go

type SafetensorsHeader struct {
    Tensors map[string]TensorInfo `json:"__metadata__,omitempty"`
}

type TensorInfo struct {
    Dtype       string  `json:"dtype"`
    Shape       []int64 `json:"shape"`
    DataOffsets [2]int64 `json:"data_offsets"`
}

type SafetensorsLoader struct {
    basePath    string
    indexPath   string
    shards      []string
    tensorIndex map[string]ShardInfo
    mmapFiles   []*os.File
    mmapData    [][]byte
}

type ShardInfo struct {
    ShardFile string
    Offset    int64
    Size      int64
    Dtype     string
    Shape     []int64
}
```

### Task List

```yaml
Task 1: Create safetensors parser
  CREATE pkg/model/safetensors.go:
    - MIRROR pattern from: pkg/types/tensor.go (struct definitions)
    - Parse safetensors header (JSON at file start)
    - Support single file and sharded loading
    - Memory-map large files

Task 2: Create weight tensor types
  CREATE pkg/model/weights.go:
    - LayerWeights struct with all weight tensors
    - LoadedModel struct for complete model
    - Weight name mapping (HuggingFace → internal)

Task 3: Create memory-mapped file support
  CREATE pkg/model/mmap.go:
    - MmapFile function for memory-mapped access
    - Aligned read for GPU copy efficiency
    - Close/cleanup functions

Task 4: Create main loader interface
  CREATE pkg/model/loader.go:
    - ModelLoader interface
    - SafetensorsLoader implementation
    - Factory function NewLoader(format, path)

Task 5: Create distributed loader
  CREATE pkg/model/distributed.go:
    - DistributedLoader wraps base loader
    - Uses scheduler assignments
    - Loads only assigned layers per peer
    - Copies weights to correct GPU device

Task 6: Integrate with inference engine
  MODIFY pkg/inference/engine.go:
    - Add LoadModel method
    - Store loaded weights
    - Use weights in forward pass

Task 7: Add tests
  CREATE tests/model/loader_test.go:
    - Test safetensors parsing
    - Test sharded loading
    - Test distributed loading
    - Test memory mapping

Task 8: Add CLI support
  MODIFY cmd/neurogrid/main.go:
    - Add --model-path flag
    - Load model on startup
```

### Per-Task Pseudocode

```go
// Task 1: Safetensors parser
func parseSafetensorsHeader(file *os.File) (*SafetensorsHeader, error) {
    // PATTERN: Read 8-byte little-endian header size
    // Read header_size bytes as JSON
    // Parse tensor info map
    // Return header with tensor offsets
}

func (l *SafetensorsLoader) loadTensor(name string) (*types.Tensor, error) {
    // PATTERN: Look up tensor in index
    // Memory-map shard file if not already mapped
    // Slice mmap data at offset:offset+size
    // Convert to types.Tensor with correct dtype
    // GOTCHA: Ensure alignment for GPU copy
}

// Task 5: Distributed loader
func (d *DistributedLoader) LoadForPeer(ctx context.Context, peerID string) error {
    // PATTERN: Get layer assignments from scheduler
    // For each assigned layer:
    //   Load layer weights from base loader
    //   Allocate GPU memory on assigned device
    //   Copy weights to GPU
    // CRITICAL: Only load layers assigned to this peer
}
```

### Integration Points

```yaml
CONFIG:
  - add to: cmd/neurogrid/main.go
  - pattern: 'flag.String("model-path", "", "Path to model weights")'
  - required: true for inference mode

INFERENCE:
  - add to: pkg/inference/engine.go
  - method: LoadModel(path string) error
  - usage: Called before Generate() can be used

SCHEDULER:
  - add to: pkg/scheduler/scheduler.go
  - method: GetLayersForPeer(peerID string) []int
  - usage: DistributedLoader uses this for partial loading
```

---

## Validation Loop

### Level 1: Syntax & Style

```bash
go fmt ./pkg/model/...
go vet ./pkg/model/...
golangci-lint run ./pkg/model/...

# Expected: No errors
```

### Level 2: Unit Tests

```bash
go test -v ./tests/model/...

# Expected: All tests pass
```

### Level 3: Integration Test

```bash
# Download small test model (TinyLlama 1.1B)
./scripts/download_model.sh tinyllama

# Load and verify
go test -v -run TestLoadTinyLlama ./tests/model/...

# Expected: Model loads, weights match expected shapes
```

---

## Final Validation Checklist

- [ ] All tests pass: `go test ./pkg/model/... ./tests/model/...`
- [ ] No linting errors: `golangci-lint run ./pkg/model/...`
- [ ] Memory-mapped loading works for large files
- [ ] Sharded model loading works
- [ ] Weights distributed to correct GPU per assignment
- [ ] INT8 and FP16 formats supported
- [ ] Weight shapes validated against config

---

## Anti-Patterns to Avoid

- ❌ Don't load entire model into RAM for large models
- ❌ Don't ignore byte order (safetensors is little-endian)
- ❌ Don't skip shape validation
- ❌ Don't allocate GPU memory before knowing device assignment
- ❌ Don't forget to close memory-mapped files

---

## Performance Targets

| Metric | Target |
|--------|--------|
| Llama 7B load time | < 30s |
| Llama 13B load time | < 60s |
| Memory overhead | < 10% above weight size |
| GPU copy bandwidth | > 10 GB/s |

---

## Dependencies

```go
// go.mod additions
require (
    golang.org/x/sys v0.20.0  // For mmap syscalls
)
```

---

**PRP Confidence Score: 9/10**

**Rationale**:
- +3: Clear safetensors format specification available
- +2: Existing tensor types to build on
- +2: Scheduler integration well-defined
- +2: Memory mapping is standard Go/syscall
- -1: Sharded loading complexity needs testing
