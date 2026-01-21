# PRP: Tensor Parallelism (libp2p + CUDA Network)

## Document Information

| Field | Value |
|-------|-------|
| **Feature Name** | Tensor Parallelism over P2P Network |
| **PRP Version** | 1.0 |
| **Created** | 2026-01-21 |
| **Priority** | Long-Term (Scale) |
| **Status** | Ready for Implementation |
| **Confidence Score** | 6/10 |
| **Dependencies** | All previous PRPs |

---

## Discovery Summary

### Initial Task Analysis

Implement tensor parallelism across remote GPUs connected via libp2p, splitting individual layers across multiple GPUs to reduce single-request latency and enable models larger than any single GPU's VRAM.

### User Clarifications Received

- **Question**: Tensor parallel vs pipeline parallel - when to use which?
- **Answer**: Tensor parallel for latency, pipeline for throughput
- **Impact**: Hybrid approach with configurable parallelism strategy

### Missing Requirements Identified

- All-reduce implementation over P2P
- High-bandwidth transport optimization
- Synchronization barriers
- Partial tensor transfer

---

## Goal

Implement tensor parallelism that splits attention and FFN computations across multiple remote GPUs, using libp2p for tensor shard communication with optimized all-reduce operations.

## Why

- **Latency**: Reduce single-request latency (critical for interactive use)
- **Model size**: Enable models larger than single GPU VRAM
- **Flexibility**: Hybrid parallelism strategies
- **Utilization**: Better GPU utilization for large batches

## What

### Success Criteria

- [ ] Split attention heads across N GPUs (N ≤ 8)
- [ ] Split FFN columns across N GPUs
- [ ] All-reduce over libp2p with < 100ms overhead
- [ ] 2x latency improvement with 2 GPUs (vs pipeline)
- [ ] Support hybrid tensor + pipeline parallelism
- [ ] Graceful fallback on peer failure

---

## All Needed Context

### Research Phase Summary

- **Codebase patterns found**: `pkg/transport/p2p.go` for communication
- **External research needed**: Yes - Megatron-LM, all-reduce algorithms
- **Knowledge gaps identified**: NCCL-equivalent over TCP, all-reduce efficiency

### Documentation & References

```yaml
- url: https://arxiv.org/abs/1909.08053
  why: Megatron-LM tensor parallelism paper

- url: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html
  why: NCCL all-reduce patterns (to replicate over P2P)

- url: https://github.com/NVIDIA/Megatron-LM
  why: Reference implementation

- file: pkg/transport/p2p.go
  why: Existing P2P transport to extend

- file: gpu/cuda/attention.cu
  why: Attention kernel to split
```

### Current Codebase tree

```
pkg/
├── transport/
│   ├── transport.go    # Transport interface
│   ├── local.go        # CUDA P2P transport
│   └── p2p.go          # libp2p transport
├── scheduler/
│   └── scheduler.go    # Layer-level assignment
└── inference/
    └── engine.go       # Pipeline parallel inference

gpu/cuda/
├── attention.cu        # Full attention kernel
├── matmul.cu          # GEMM operations
└── kernels.cu         # Other kernels
```

### Desired Codebase tree

```
pkg/
├── transport/
│   ├── transport.go
│   ├── local.go
│   ├── p2p.go
│   └── collective/     # NEW: Collective operations
│       ├── allreduce.go
│       ├── allgather.go
│       └── barrier.go
├── scheduler/
│   ├── scheduler.go
│   └── tensor_parallel.go  # NEW: TP scheduling
├── inference/
│   ├── engine.go
│   └── tensor_parallel.go  # NEW: TP inference
└── parallel/           # NEW: Parallelism strategies
    ├── strategy.go
    ├── pipeline.go
    └── tensor.go

gpu/cuda/
├── attention.cu        # MODIFY: Split attention
├── attention_tp.cu     # NEW: Tensor-parallel attention
├── matmul.cu
├── matmul_tp.cu        # NEW: Column/row parallel GEMM
└── kernels.cu
```

### Known Gotchas

```go
// CRITICAL: Network bandwidth limits all-reduce efficiency
// CRITICAL: Synchronization overhead can dominate for small tensors
// CRITICAL: Head count must be divisible by TP degree
// CRITICAL: FFN intermediate size must be divisible by TP degree
// CRITICAL: All-reduce is O(n) data transfer, not O(1)
```

---

## Implementation Blueprint

### Data Models

```go
// pkg/parallel/strategy.go

type ParallelismStrategy struct {
    TensorParallelDegree   int  // GPUs per layer (TP)
    PipelineParallelDegree int  // Layers per group (PP)
    DataParallelDegree     int  // Batch splitting (DP)
}

type ParallelGroup struct {
    GroupID     int
    Members     []peer.ID
    Role        GroupRole
    LocalRank   int
    WorldSize   int
}

type GroupRole int

const (
    RoleTensorParallel GroupRole = iota
    RolePipelineParallel
)

// pkg/transport/collective/allreduce.go

type AllReduceOp int

const (
    OpSum AllReduceOp = iota
    OpMax
    OpMin
)

type AllReducer interface {
    AllReduce(ctx context.Context, data []byte, op AllReduceOp) ([]byte, error)
    AllGather(ctx context.Context, data []byte) ([][]byte, error)
    Barrier(ctx context.Context) error
}

type RingAllReducer struct {
    group      *ParallelGroup
    transports map[peer.ID]transport.Transport
    localRank  int
    worldSize  int
}

// pkg/scheduler/tensor_parallel.go

type TensorParallelScheduler struct {
    baseScheduler *Scheduler
    tpDegree      int
    tpGroups      map[int]*ParallelGroup  // layerID -> TP group
}

type TensorParallelAssignment struct {
    LayerID     int
    TPGroup     *ParallelGroup
    HeadRange   [2]int  // Start and end head indices
    FFNRange    [2]int  // Start and end FFN column indices
}

// pkg/inference/tensor_parallel.go

type TensorParallelEngine struct {
    baseEngine *Engine
    tpScheduler *TensorParallelScheduler
    allReducer  AllReducer
    localRank   int
    tpDegree    int
}
```

### Task List

```yaml
Task 1: Create parallel strategy types
  CREATE pkg/parallel/strategy.go:
    - ParallelismStrategy struct
    - ParallelGroup management
    - Role definitions

Task 2: Implement ring all-reduce
  CREATE pkg/transport/collective/allreduce.go:
    - RingAllReducer for efficient all-reduce
    - AllGather for collecting shards
    - Barrier for synchronization

Task 3: Create tensor-parallel scheduler
  CREATE pkg/scheduler/tensor_parallel.go:
    - TensorParallelScheduler
    - TP group formation
    - Head/FFN range assignment

Task 4: Create column-parallel GEMM
  CREATE gpu/cuda/matmul_tp.cu:
    - ColumnParallelLinear kernel
    - RowParallelLinear kernel
    - Partial output handling

Task 5: Create tensor-parallel attention
  CREATE gpu/cuda/attention_tp.cu:
    - Split heads across GPUs
    - Local attention computation
    - Output gathering

Task 6: Create tensor-parallel inference engine
  CREATE pkg/inference/tensor_parallel.go:
    - TensorParallelEngine
    - Forward with all-reduce
    - Hybrid PP+TP support

Task 7: Add Go bindings for TP kernels
  MODIFY gpu/bindings/gpu.go:
    - ColumnParallelGEMM binding
    - RowParallelGEMM binding
    - SplitAttention binding

Task 8: Optimize all-reduce for large tensors
  MODIFY pkg/transport/collective/allreduce.go:
    - Chunked transfer for large tensors
    - Overlap compute and communication
    - Compression for bandwidth

Task 9: Add hybrid parallelism support
  MODIFY pkg/inference/engine.go:
    - Detect parallelism strategy
    - Route to appropriate engine
    - Handle mixed PP+TP

Task 10: Add metrics and monitoring
  MODIFY pkg/metrics/registry.go:
    - all_reduce_duration histogram
    - tensor_parallel_overhead
    - communication_bandwidth

Task 11: Add tests
  CREATE tests/parallel/tensor_parallel_test.go:
    - Test all-reduce correctness
    - Test attention splitting
    - Test FFN splitting
    - Benchmark communication overhead
```

### Per-Task Pseudocode

```go
// Task 2: Ring all-reduce implementation
func (r *RingAllReducer) AllReduce(ctx context.Context, data []byte, op AllReduceOp) ([]byte, error) {
    // Ring all-reduce in 2 phases:
    // 1. Reduce-scatter: Each rank accumulates a chunk from all ranks
    // 2. All-gather: Each rank broadcasts its chunk to all ranks

    chunkSize := len(data) / r.worldSize
    chunks := splitIntoChunks(data, chunkSize)

    // Phase 1: Reduce-scatter
    // Each iteration: send chunk[i] to rank+1, receive from rank-1, accumulate
    for i := 0; i < r.worldSize-1; i++ {
        sendIdx := (r.localRank - i + r.worldSize) % r.worldSize
        recvIdx := (r.localRank - i - 1 + r.worldSize) % r.worldSize

        // Send to next rank
        nextRank := (r.localRank + 1) % r.worldSize
        r.transports[r.group.Members[nextRank]].SendActivation(ctx, 0, 0, chunks[sendIdx])

        // Receive from previous rank
        prevRank := (r.localRank - 1 + r.worldSize) % r.worldSize
        _, received, _ := r.transports[r.group.Members[prevRank]].RecvActivation(ctx, 0)

        // Accumulate
        chunks[recvIdx] = reduce(chunks[recvIdx], received, op)
    }

    // Phase 2: All-gather
    // Each iteration: send accumulated chunk to next, receive from prev
    for i := 0; i < r.worldSize-1; i++ {
        sendIdx := (r.localRank - i + 1 + r.worldSize) % r.worldSize
        recvIdx := (r.localRank - i + r.worldSize) % r.worldSize

        nextRank := (r.localRank + 1) % r.worldSize
        r.transports[r.group.Members[nextRank]].SendActivation(ctx, 0, 0, chunks[sendIdx])

        prevRank := (r.localRank - 1 + r.worldSize) % r.worldSize
        _, received, _ := r.transports[r.group.Members[prevRank]].RecvActivation(ctx, 0)

        chunks[recvIdx] = received
    }

    return mergeChunks(chunks), nil
}

// Task 3: TP scheduler
func (s *TensorParallelScheduler) FormTPGroups() error {
    // Group peers into TP groups of size tpDegree
    peers := s.baseScheduler.GetRegisteredPeers()
    if len(peers) < s.tpDegree {
        return ErrInsufficientPeers
    }

    numGroups := len(peers) / s.tpDegree

    for i := 0; i < numGroups; i++ {
        groupMembers := peers[i*s.tpDegree : (i+1)*s.tpDegree]

        group := &ParallelGroup{
            GroupID:   i,
            Members:   groupMembers,
            Role:      RoleTensorParallel,
            WorldSize: s.tpDegree,
        }

        // Assign layers to this group
        layersPerGroup := s.baseScheduler.config.NumLayers / numGroups
        for layerID := i * layersPerGroup; layerID < (i+1)*layersPerGroup; layerID++ {
            s.tpGroups[layerID] = group
        }
    }

    return nil
}

func (s *TensorParallelScheduler) GetAssignment(layerID, localRank int) *TensorParallelAssignment {
    group := s.tpGroups[layerID]

    numHeads := s.baseScheduler.config.NumHeads
    headsPerRank := numHeads / s.tpDegree
    headStart := localRank * headsPerRank
    headEnd := headStart + headsPerRank

    ffnSize := s.baseScheduler.config.IntermediateSize
    ffnPerRank := ffnSize / s.tpDegree
    ffnStart := localRank * ffnPerRank
    ffnEnd := ffnStart + ffnPerRank

    return &TensorParallelAssignment{
        LayerID:   layerID,
        TPGroup:   group,
        HeadRange: [2]int{headStart, headEnd},
        FFNRange:  [2]int{ffnStart, ffnEnd},
    }
}

// Task 4: Column-parallel GEMM (CUDA pseudocode)
// gpu/cuda/matmul_tp.cu
__global__ void column_parallel_gemm(
    const half* __restrict__ input,      // [batch, hidden]
    const half* __restrict__ weight,     // [hidden, local_out] - column shard
    half* __restrict__ output,           // [batch, local_out]
    int batch_size,
    int hidden_size,
    int local_output_size
) {
    // Standard GEMM but output is partial (columns)
    // No all-reduce needed after column-parallel
    // ...cuBLAS GEMM call...
}

__global__ void row_parallel_gemm(
    const half* __restrict__ input,      // [batch, local_in] - row shard
    const half* __restrict__ weight,     // [local_in, hidden] - row shard
    half* __restrict__ output,           // [batch, hidden] - partial
    int batch_size,
    int local_input_size,
    int hidden_size
) {
    // Standard GEMM but input is partial (rows)
    // ALL-REDUCE NEEDED after row-parallel to sum partials
    // ...cuBLAS GEMM call...
}

// Task 5: Tensor-parallel attention
// gpu/cuda/attention_tp.cu
// Each GPU handles a subset of attention heads
void tensor_parallel_attention(
    const Tensor& hidden,
    const Tensor& q_weight,  // [hidden, local_head_dim] column shard
    const Tensor& k_weight,  // [hidden, local_head_dim] column shard
    const Tensor& v_weight,  // [hidden, local_head_dim] column shard
    const Tensor& o_weight,  // [local_head_dim, hidden] row shard
    KVCache& local_cache,    // Local portion of KV cache
    int local_num_heads,
    int head_dim
) {
    // 1. Column-parallel Q, K, V projections (no all-reduce)
    auto q = column_parallel_gemm(hidden, q_weight);
    auto k = column_parallel_gemm(hidden, k_weight);
    auto v = column_parallel_gemm(hidden, v_weight);

    // 2. Local attention (only on assigned heads)
    auto attn_output = local_multi_head_attention(q, k, v, local_cache, local_num_heads, head_dim);

    // 3. Row-parallel output projection
    auto partial_output = row_parallel_gemm(attn_output, o_weight);

    // 4. ALL-REDUCE to sum partial outputs from all TP ranks
    // This is done in Go layer after CUDA kernel returns
}

// Task 6: Tensor-parallel inference engine
func (e *TensorParallelEngine) ForwardLayer(ctx context.Context, layerID int, hidden []byte, position int) ([]byte, error) {
    assignment := e.tpScheduler.GetAssignment(layerID, e.localRank)

    // 1. Attention with tensor parallelism
    // Column-parallel Q, K, V (no communication)
    q := e.columnParallelGEMM(hidden, e.weights.QProj[assignment.HeadRange[0]:assignment.HeadRange[1]])
    k := e.columnParallelGEMM(hidden, e.weights.KProj[assignment.HeadRange[0]:assignment.HeadRange[1]])
    v := e.columnParallelGEMM(hidden, e.weights.VProj[assignment.HeadRange[0]:assignment.HeadRange[1]])

    // Local attention on assigned heads
    attnOut := e.localAttention(q, k, v, position, assignment.HeadRange)

    // Row-parallel output projection
    attnPartial := e.rowParallelGEMM(attnOut, e.weights.OProj)

    // ALL-REDUCE to sum attention outputs
    attnFull, err := e.allReducer.AllReduce(ctx, attnPartial, OpSum)
    if err != nil {
        return nil, fmt.Errorf("attention all-reduce failed: %w", err)
    }

    // Residual connection
    attnFull = add(hidden, attnFull)

    // 2. FFN with tensor parallelism
    // Column-parallel gate and up projections
    gate := e.columnParallelGEMM(attnFull, e.weights.GateProj[assignment.FFNRange[0]:assignment.FFNRange[1]])
    up := e.columnParallelGEMM(attnFull, e.weights.UpProj[assignment.FFNRange[0]:assignment.FFNRange[1]])

    // Local SiLU and multiply
    ffnInter := silu(gate) * up

    // Row-parallel down projection
    ffnPartial := e.rowParallelGEMM(ffnInter, e.weights.DownProj)

    // ALL-REDUCE to sum FFN outputs
    ffnFull, err := e.allReducer.AllReduce(ctx, ffnPartial, OpSum)
    if err != nil {
        return nil, fmt.Errorf("FFN all-reduce failed: %w", err)
    }

    // Residual connection
    output := add(attnFull, ffnFull)

    return output, nil
}
```

### Integration Points

```yaml
TRANSPORT:
  - add: pkg/transport/collective/ for all-reduce
  - modify: p2p.go for collective operations

SCHEDULER:
  - add: pkg/scheduler/tensor_parallel.go
  - modify: scheduler.go for hybrid support

INFERENCE:
  - add: pkg/inference/tensor_parallel.go
  - modify: engine.go to detect and use TP

CUDA:
  - add: matmul_tp.cu, attention_tp.cu
  - modify: bindings/gpu.go for TP kernels

CONFIG:
  - add: tensor_parallel_degree setting
  - add: parallelism_strategy setting
```

---

## Validation Loop

### Level 1: Syntax & Style

```bash
go fmt ./pkg/...
go vet ./pkg/...
golangci-lint run ./pkg/...

# Expected: No errors
```

### Level 2: Unit Tests

```bash
go test -v ./tests/parallel/...

# Expected: All tests pass
```

### Level 3: Correctness Test

```bash
# Compare TP output with single-GPU output
go test -v -run TestTensorParallelCorrectness ./tests/parallel/...

# Expected: Output matches within tolerance (1e-3)
```

### Level 4: Performance Test

```bash
# Benchmark TP vs PP latency
go test -bench=BenchmarkTensorParallel -benchtime=10s ./tests/parallel/...

# Expected: TP-2 shows ~1.8x latency improvement over PP-2
```

---

## Final Validation Checklist

- [ ] All tests pass: `go test ./pkg/parallel/... ./tests/parallel/...`
- [ ] No linting errors: `golangci-lint run ./pkg/...`
- [ ] All-reduce produces correct results
- [ ] Attention splitting is correct
- [ ] FFN splitting is correct
- [ ] Latency improvement with TP-2 (>1.5x)
- [ ] Hybrid PP+TP works

---

## Anti-Patterns to Avoid

- ❌ Don't ignore synchronization overhead
- ❌ Don't transfer full tensors (use shards)
- ❌ Don't forget all-reduce after row-parallel
- ❌ Don't assume network bandwidth is unlimited
- ❌ Don't mix TP degrees within a layer

---

## Performance Targets

| Metric | Target |
|--------|--------|
| All-reduce overhead (1MB) | < 50ms |
| TP-2 latency improvement | > 1.5x vs PP |
| TP-4 latency improvement | > 2.5x vs PP |
| Communication/compute ratio | < 20% |

---

## Communication Analysis

For Llama 7B with batch=1, seq=1:

| Operation | Tensor Size | Frequency | Data/Layer |
|-----------|-------------|-----------|------------|
| Attention all-reduce | 4096 × 2 bytes | 1 per layer | 8 KB |
| FFN all-reduce | 4096 × 2 bytes | 1 per layer | 8 KB |
| **Total per layer** | | | **16 KB** |
| **Total per forward** | | 32 layers | **512 KB** |

At 10 Gbps network: 512 KB / 10 Gbps ≈ 0.4ms
With overhead: ~5-10ms realistic

---

## Limitations

1. **Bandwidth bound**: TP over network is slower than NVLink
2. **Latency sensitive**: All-reduce adds synchronization points
3. **Best for**: Large batches where compute dominates
4. **Not ideal for**: Single-token generation with small batch

---

**PRP Confidence Score: 6/10**

**Rationale**:
- +2: Megatron-LM provides proven patterns
- +1: Existing P2P infrastructure
- +1: CUDA kernels can be adapted
- -1: Network bandwidth limits efficiency
- -2: All-reduce over TCP/libp2p is novel
- -2: Synchronization overhead significant
- -1: Correctness verification is complex
