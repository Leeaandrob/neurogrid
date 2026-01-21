# PRP: Continuous Batching (Throughput Optimization)

## Document Information

| Field | Value |
|-------|-------|
| **Feature Name** | Continuous Batching |
| **PRP Version** | 1.0 |
| **Created** | 2026-01-21 |
| **Priority** | Medium-Term (Production) |
| **Status** | Ready for Implementation |
| **Confidence Score** | 8/10 |
| **Dependencies** | PRP-01, PRP-02, PRP-04 |

---

## Discovery Summary

### Initial Task Analysis

Implement continuous batching (also known as in-flight batching or iteration-level scheduling) to maximize GPU utilization by dynamically adding new requests to running batches.

### User Clarifications Received

- **Question**: Static batching vs continuous batching?
- **Answer**: Continuous batching for better throughput
- **Impact**: More complex implementation but significantly better efficiency

### Missing Requirements Identified

- Request queue management
- Batch size limits
- Priority scheduling
- Memory management for variable-length sequences

---

## Goal

Implement continuous batching in the inference engine to process multiple concurrent requests efficiently, dynamically adding new requests as existing ones complete.

## Why

- **Throughput**: 2-10x improvement over static batching
- **Latency**: Lower average latency per request
- **GPU utilization**: Keep GPU busy even with variable-length requests
- **Cost efficiency**: Serve more users per GPU

## What

### Success Criteria

- [ ] Support batch sizes up to 32 concurrent requests
- [ ] Dynamic request insertion without interrupting running requests
- [ ] Automatic eviction of completed sequences
- [ ] Maintain low latency for new requests (< 100ms queue time)
- [ ] Throughput > 100 tokens/sec aggregate (batch=8)
- [ ] Memory-efficient KV cache management

---

## All Needed Context

### Research Phase Summary

- **Codebase patterns found**: `pkg/inference/engine.go` with single-request Generate
- **External research needed**: Yes - vLLM and Orca batching papers
- **Knowledge gaps identified**: PagedAttention for memory efficiency

### Documentation & References

```yaml
- url: https://arxiv.org/abs/2309.06180
  why: vLLM paper on PagedAttention and continuous batching

- url: https://www.usenix.org/system/files/osdi22-yu.pdf
  why: Orca paper on iteration-level scheduling

- file: pkg/inference/engine.go
  why: Current single-request implementation to extend

- file: pkg/inference/kvcache.go
  why: KV cache management for batching
```

### Current Codebase tree

```
pkg/inference/
├── engine.go         # Single-request Generate
├── sampler.go        # Token sampling
└── kvcache.go        # KV cache (single sequence)
```

### Desired Codebase tree

```
pkg/inference/
├── engine.go         # MODIFY: Add batch processing
├── sampler.go        # MODIFY: Batch sampling
├── kvcache.go        # MODIFY: Multi-sequence cache
├── batch.go          # NEW: Batch management
├── scheduler.go      # NEW: Request scheduler
└── memory.go         # NEW: Memory management
```

### Known Gotchas

```go
// CRITICAL: Different sequences have different lengths - need padding or packing
// CRITICAL: KV cache must track per-sequence positions
// CRITICAL: New requests need prefill while others are generating
// CRITICAL: Memory fragmentation with variable lengths
// CRITICAL: Batch size affects GPU memory requirements
```

---

## Implementation Blueprint

### Data Models

```go
// pkg/inference/batch.go

type Batch struct {
    Sequences    []*Sequence
    MaxBatchSize int
    ActiveCount  int
    Generation   int64  // Monotonic counter for ordering
}

type Sequence struct {
    ID           string
    RequestID    string
    Tokens       []int
    Position     int              // Current position in generation
    MaxTokens    int
    Temperature  float32
    TopP         float32
    State        SequenceState
    Callback     StreamCallback   // For streaming
    CreatedAt    time.Time
    KVCacheSlot  int              // Slot in batched KV cache
    Hidden       []byte           // Current hidden state
}

type SequenceState int

const (
    SequencePending SequenceState = iota
    SequencePrefill
    SequenceGenerate
    SequenceComplete
)

// pkg/inference/scheduler.go

type RequestScheduler struct {
    pendingQueue chan *Request
    runningBatch *Batch
    maxBatchSize int
    maxQueueSize int
    mu           sync.Mutex
}

type Request struct {
    ID          string
    Prompt      string
    MaxTokens   int
    Temperature float32
    TopP        float32
    Stream      bool
    Callback    StreamCallback
    ResultChan  chan *GenerateResponse
    CreatedAt   time.Time
}

// pkg/inference/memory.go

type KVCacheManager struct {
    cache       map[int]*BatchedKVCache  // layerID -> batched cache
    slotPool    *SlotPool
    maxSlots    int
    slotSize    int  // Max sequence length per slot
}

type BatchedKVCache struct {
    Keys    *types.Tensor  // [batch, heads, seq_len, head_dim]
    Values  *types.Tensor  // [batch, heads, seq_len, head_dim]
    Lengths []int          // Actual length per slot
}

type SlotPool struct {
    available []int
    inUse     map[string]int  // sequenceID -> slot
    mu        sync.Mutex
}
```

### Task List

```yaml
Task 1: Create batch data structures
  CREATE pkg/inference/batch.go:
    - Batch struct for managing sequences
    - Sequence struct with state machine
    - Batch operations (add, remove, compact)

Task 2: Create request scheduler
  CREATE pkg/inference/scheduler.go:
    - RequestScheduler with priority queue
    - Admission control
    - Dynamic batch adjustment

Task 3: Create batched KV cache manager
  CREATE pkg/inference/memory.go:
    - KVCacheManager with slot allocation
    - BatchedKVCache for multi-sequence
    - Slot pool for memory reuse

Task 4: Modify engine for batched inference
  MODIFY pkg/inference/engine.go:
    - Add BatchedForward method
    - Integrate scheduler
    - Handle mixed prefill/generate

Task 5: Modify sampler for batch
  MODIFY pkg/inference/sampler.go:
    - BatchSample for multiple sequences
    - Per-sequence temperature/top_p

Task 6: Update CUDA kernels for batching
  MODIFY gpu/cuda/attention.cu:
    - Batched attention kernel
    - Variable sequence length handling

Task 7: Add batch processing loop
  MODIFY pkg/inference/engine.go:
    - Main batch processing goroutine
    - Continuous iteration loop
    - Request insertion/completion

Task 8: Add metrics and monitoring
  CREATE pkg/inference/metrics.go:
    - Batch size metrics
    - Queue depth
    - Throughput counters

Task 9: Add tests
  CREATE tests/inference/batch_test.go:
    - Test batch operations
    - Test scheduler
    - Test concurrent requests
    - Throughput benchmarks
```

### Per-Task Pseudocode

```go
// Task 2: Request Scheduler
func NewRequestScheduler(maxBatchSize, maxQueueSize int) *RequestScheduler {
    s := &RequestScheduler{
        pendingQueue: make(chan *Request, maxQueueSize),
        maxBatchSize: maxBatchSize,
        maxQueueSize: maxQueueSize,
    }
    return s
}

func (s *RequestScheduler) Submit(req *Request) error {
    select {
    case s.pendingQueue <- req:
        return nil
    default:
        return ErrQueueFull
    }
}

func (s *RequestScheduler) FillBatch(batch *Batch) {
    // Add pending requests to batch if space available
    for batch.ActiveCount < s.maxBatchSize {
        select {
        case req := <-s.pendingQueue:
            seq := s.requestToSequence(req)
            batch.AddSequence(seq)
        default:
            return  // No more pending requests
        }
    }
}

// Task 4: Batched Forward
func (e *Engine) BatchedForward(ctx context.Context, batch *Batch) error {
    // Separate sequences by state
    var prefillSeqs, generateSeqs []*Sequence
    for _, seq := range batch.Sequences {
        if seq.State == SequencePrefill {
            prefillSeqs = append(prefillSeqs, seq)
        } else if seq.State == SequenceGenerate {
            generateSeqs = append(generateSeqs, seq)
        }
    }

    // Process prefill sequences (full prompt)
    if len(prefillSeqs) > 0 {
        if err := e.batchPrefill(ctx, prefillSeqs); err != nil {
            return err
        }
    }

    // Process generate sequences (single token each)
    if len(generateSeqs) > 0 {
        if err := e.batchGenerate(ctx, generateSeqs); err != nil {
            return err
        }
    }

    return nil
}

func (e *Engine) batchGenerate(ctx context.Context, seqs []*Sequence) error {
    // Batch all hidden states
    batchHidden := e.packHiddenStates(seqs)
    positions := make([]int, len(seqs))
    for i, seq := range seqs {
        positions[i] = seq.Position
    }

    // Forward through all layers with batched attention
    for layerID := 0; layerID < e.config.NumLayers; layerID++ {
        batchHidden = e.batchedLayerForward(ctx, layerID, batchHidden, positions)
    }

    // Batch sample next tokens
    logits := e.batchComputeLogits(batchHidden)
    nextTokens := e.sampler.BatchSample(logits, seqs)

    // Update sequences
    for i, seq := range seqs {
        token := nextTokens[i]

        // Stream callback
        if seq.Callback != nil {
            tokenText := e.tokenizer.DecodeSingle(token)
            seq.Callback(tokenText, false, "")
        }

        // Check completion
        if token == e.tokenizer.EOSToken() || seq.Position >= seq.MaxTokens {
            seq.State = SequenceComplete
            if seq.Callback != nil {
                reason := "stop"
                if seq.Position >= seq.MaxTokens {
                    reason = "length"
                }
                seq.Callback("", true, reason)
            }
        } else {
            seq.Tokens = append(seq.Tokens, token)
            seq.Position++
        }
    }

    return nil
}

// Task 7: Main batch loop
func (e *Engine) RunBatchLoop(ctx context.Context) {
    batch := NewBatch(e.maxBatchSize)

    for {
        select {
        case <-ctx.Done():
            return
        default:
        }

        // Fill batch with pending requests
        e.scheduler.FillBatch(batch)

        if batch.ActiveCount == 0 {
            // No work, sleep briefly
            time.Sleep(1 * time.Millisecond)
            continue
        }

        // Process one iteration
        if err := e.BatchedForward(ctx, batch); err != nil {
            log.Printf("Batch forward error: %v", err)
        }

        // Remove completed sequences
        completed := batch.RemoveCompleted()

        // Release KV cache slots
        for _, seq := range completed {
            e.kvManager.ReleaseSlot(seq.KVCacheSlot)
        }

        // Compact batch to fill gaps
        batch.Compact()
    }
}
```

### Integration Points

```yaml
ENGINE:
  - modify: pkg/inference/engine.go
  - add: RunBatchLoop goroutine
  - add: BatchedForward method

API:
  - modify: api/handlers.go
  - call: scheduler.Submit instead of engine.Generate

CUDA:
  - modify: gpu/cuda/attention.cu
  - add: Batched attention kernel
  - handle: Variable sequence lengths

CONFIG:
  - add: max_batch_size setting
  - add: max_queue_size setting
```

---

## Validation Loop

### Level 1: Syntax & Style

```bash
go fmt ./pkg/inference/...
go vet ./pkg/inference/...
golangci-lint run ./pkg/inference/...

# Expected: No errors
```

### Level 2: Unit Tests

```bash
go test -v ./tests/inference/batch_test.go

# Expected: All tests pass
```

### Level 3: Throughput Benchmark

```bash
# Run throughput benchmark
go test -bench=BenchmarkBatchedThroughput -benchtime=30s ./tests/inference/...

# Expected: > 100 tokens/sec aggregate at batch=8
```

---

## Final Validation Checklist

- [ ] All tests pass: `go test ./pkg/inference/...`
- [ ] No linting errors: `golangci-lint run ./pkg/inference/...`
- [ ] Batch size 32 works without OOM
- [ ] Dynamic request insertion works
- [ ] Completed sequences removed properly
- [ ] Throughput > 100 tok/s at batch=8
- [ ] Latency acceptable (< 2x single request)

---

## Anti-Patterns to Avoid

- ❌ Don't block on single slow sequence
- ❌ Don't copy KV cache on every iteration
- ❌ Don't ignore memory fragmentation
- ❌ Don't use static batch waiting
- ❌ Don't forget to release slots on completion

---

## Performance Targets

| Metric | Target |
|--------|--------|
| Max batch size | 32 sequences |
| Throughput (batch=8) | > 100 tok/s |
| Queue wait time | < 100ms |
| Memory efficiency | > 80% KV cache utilization |

---

**PRP Confidence Score: 8/10**

**Rationale**:
- +2: vLLM and Orca papers provide clear algorithms
- +2: Existing CUDA infrastructure
- +2: Clear performance targets
- +1: Scheduler patterns well-known
- -1: Batched attention kernel complexity
- -1: Memory management challenges
- -1: Variable sequence length handling
