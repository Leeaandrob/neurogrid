# Task Breakdown: LFM2.5-1.2B-Thinking Model Support with BF16 Native Compute

**PRP Reference**: `docs/prps/lfm2-support.md`
**Created**: 2026-03-12
**Estimated Total Effort**: Large (3-4 weeks, 2-3 developers)
**Overall Complexity**: Complex -- first non-Llama architecture, new CUDA kernels, full BF16 compute pipeline

---

## PRP Analysis Summary

- **Feature**: Add LFM2.5-1.2B-Thinking as the second model architecture in NeuroGrid Engine
- **Scope**: BF16 CUDA kernels, depthwise causal conv1d, hybrid conv+attention layer loop, hybrid cache system, BPE tokenizer with ChatML, weight loader for LFM2 naming convention
- **Architecture Impact**: Transforms the engine from Llama-only to multi-architecture. 16 layers: 10 conv blocks + 6 GQA attention blocks sharing identical SwiGLU FFN
- **Key Technical Requirements**: BF16 GEMM via cuBLAS (CUDA_R_16BF + CUBLAS_COMPUTE_32F), depthwise causal conv1d with FP32 state, QK LayerNorm in attention, tied embeddings, eos_token_id=7
- **Backward Compatibility**: All existing Llama/Mistral/TinyLlama models must continue to work unchanged

## Task Complexity Assessment

| Dimension | Rating | Notes |
|-----------|--------|-------|
| CUDA kernel work | High | 3 new kernel files, BF16 variants of all element-wise ops |
| Integration points | High | Config propagation across 5 structs, 3 files per GPU function |
| Architectural change | High | Uniform layer loop must become heterogeneous |
| Risk of regression | Medium | Backward compat with zero-value defaults mitigates |
| Testing complexity | High | Golden data needed from HuggingFace reference |

---

## Phase Organization

### Phase 1: Foundation (T-001 through T-003)
**Objective**: Establish BF16 data types, CUDA building blocks, and config extensions. No behavior change to existing models.

**Deliverables**:
- BFloat16 Go type with conversion functions
- Extended LlamaConfig with LFM2 fields (zero-value backward compat)
- BF16 cuBLAS GEMM kernel
- BF16 element-wise kernels (RMSNorm, SiLU, add, mul)

**Milestone**: `make cuda` compiles with new .cu files; `make test` passes; existing models unaffected.

### Phase 2: New CUDA Kernels (T-004 through T-006)
**Objective**: Implement the two novel CUDA components: depthwise causal conv1d and the full LIV conv layer forward, plus BF16 attention with QK LayerNorm.

**Deliverables**:
- Causal conv1d prefill + decode kernels with FP32 conv state
- Complete LIV conv layer forward (RMSNorm, in_proj, gate, conv, out_proj, FFN)
- BF16 attention with per-head QK RMSNorm before RoPE

**Milestone**: CUDA kernel unit tests pass against known reference values.

### Phase 3: Go Integration Layer (T-007 through T-010)
**Objective**: Wire CUDA kernels into Go via CGO, load weights with LFM2 naming, manage hybrid cache, and modify the inference loop.

**Deliverables**:
- CGO bindings for all new CUDA functions (gpu.go + gpu_stub.go + gpu.h)
- Weight loader supporting LFM2 tensor paths with conv weight reshape
- Hybrid cache manager (conv state + KV cache per layer type)
- Inference engine branching by layer type in forwardAllLayersHidden

**Milestone**: Engine can load LFM2 weights, instantiate hybrid caches, and execute forward pass through all 16 layers.

### Phase 4: Tokenizer, Scheduler, and End-to-End (T-011 through T-013)
**Objective**: Complete the user-visible feature: tokenizer, chat template, scheduler, and comprehensive testing.

**Deliverables**:
- ChatML template with thinking token support
- BPE tokenizer verified for 65K vocab
- Scheduler with per-layer-type memory estimation
- Golden test suite validated against HuggingFace reference

**Milestone**: `make run-lfm2-thinking` produces coherent inference output; all existing tests still pass.

---

## Critical Path

```
T-001 (Config+Dtype) --> T-002 (BF16 CUDA) --> T-003 (Conv1d Kernel) --> T-004 (Conv Layer)
                     \                      \-> T-005 (BF16 Attention) --> T-007 (CGO Bindings)
                      \-> T-008 (Weight Loader) --> T-010 (Inference Engine)
                       \-> T-009 (Hybrid Cache) --> T-010 (Inference Engine)
                                                       \-> T-013 (Tests)
T-006 (Model Config) --> T-008 (Weight Loader) --> T-010
T-011 (Tokenizer) --> T-013 (Tests)
T-012 (Scheduler) --> T-013 (Tests)
```

**Bottleneck**: T-002 (BF16 CUDA kernels) gates both the conv and attention kernel work.
**Parallelization**: T-006, T-008, T-009, T-011, T-012 can proceed in parallel once T-001 is done.

---

## Detailed Task Breakdown

---

### T-001: BF16 Data Type Foundation and Config Extension

**Task ID**: T-001
**Task Name**: Add BFloat16 Go type and extend LlamaConfig for LFM2 fields
**Priority**: Critical
**Effort**: Short (4-6 hours)

#### Context & Background

**Source PRP Document**: `docs/prps/lfm2-support.md` - Tasks 1 and 7 (partial)

**Feature Overview**: LFM2 requires new config fields (LayerTypes[], ConvKernelSize, ConvDim, ConvBias, TieEmbeddings, Dtype, ModelType) and a BFloat16 Go type for weight loading. All new fields must use zero-value defaults that preserve existing Llama behavior.

**Task Purpose**:
- **As a** model loader
- **I need** config fields describing hybrid layer types and a BFloat16 data type
- **So that** the engine can distinguish conv vs attention layers and handle BF16 weights natively

#### Dependencies
- **Prerequisite Tasks**: None (foundation task)
- **Parallel Tasks**: None
- **Blocked By**: Nothing

#### Technical Requirements

- **REQ-1**: When LayerTypes is nil, all layers are treated as attention (backward compat)
- **REQ-2**: When Dtype is empty string, FP16 behavior is used (backward compat)
- **REQ-3**: When TieEmbeddings is false, separate lm_head weight is loaded (backward compat)
- **REQ-4**: BFloat16 round-trip conversion (FP32 -> BF16 -> FP32) must be accurate within 1 ULP

#### Implementation Details

**Files to Modify/Create**:
```
pkg/types/config.go       - ADD: LayerTypes, ConvKernelSize, ConvDim, ConvBias, TieEmbeddings, Dtype, ModelType fields
                           - ADD: IsConvLayer(), IsAttentionLayer(), NumConvLayers(), NumAttentionLayers() methods
                           - ADD: LFM2_1_2BThinkingConfig() preset
pkg/types/dtype.go        - CREATE: BFloat16 type as uint16
                           - ADD: Float32ToBFloat16(), BFloat16.Float32(), ReadBFloat16Slice()
pkg/inference/config_transfer.go - ADD: new fields to TransferableConfig with JSON tags
                                 - MODIFY: ToLlamaConfig(), FromLlamaConfig() to include new fields
pkg/scheduler/scheduler.go      - ADD: new fields to scheduler.ModelConfig (LayerTypes, ConvKernelSize, etc.)
```

**Code Patterns to Follow**:
- `pkg/types/config.go:5-16` - Existing LlamaConfig struct pattern
- `pkg/types/config.go:19-32` - Existing preset function pattern (Llama7BConfig)
- `pkg/inference/config_transfer.go:25-37` - TransferableConfig JSON tags

**Key Implementation Steps**:
1. Add new fields to LlamaConfig with zero-value defaults -> All existing presets and tests unaffected
2. Add helper methods (IsConvLayer, etc.) with nil-safe checks -> Existing code calling these on nil LayerTypes gets false
3. Create dtype.go with BFloat16 type -> Standard bit manipulation, no external deps
4. Add LFM2_1_2BThinkingConfig() preset with exact values from HuggingFace config.json
5. Propagate new fields to TransferableConfig and scheduler.ModelConfig

#### Acceptance Criteria

```gherkin
Scenario 1: Backward compatibility preserved
  Given an existing LlamaConfig with no LFM2 fields set
  When IsConvLayer(0) is called
  Then it returns false (nil LayerTypes = all attention)
  And Dtype defaults to "" (existing FP16 behavior)
  And TieEmbeddings defaults to false (separate lm_head)

Scenario 2: LFM2 config correctly identifies layer types
  Given LFM2_1_2BThinkingConfig() is created
  When checking layer types
  Then layers 0-9 are IsConvLayer() == true
  And layers 10-15 are IsAttentionLayer() == true
  And NumConvLayers() == 10
  And NumAttentionLayers() == 6

Scenario 3: BFloat16 conversion accuracy
  Given a float32 value of 1.5
  When converted to BFloat16 and back to float32
  Then the result is exactly 1.5

Scenario 4: Config serialization round-trip
  Given an LFM2 config with all new fields populated
  When serialized and deserialized via config_transfer.go
  Then all fields including LayerTypes match the original
```

**Rule-Based Criteria**:
- [ ] All existing config presets (Llama7B, 13B, 70B, TinyLlama) unchanged
- [ ] `go vet ./pkg/types/...` passes
- [ ] `go test ./pkg/types/...` passes (add unit tests for new methods)
- [ ] `go test ./pkg/inference/...` passes (config_transfer round-trip)
- [ ] Zero-value LlamaConfig behaves identically to pre-change behavior

#### Validation

```bash
go vet ./pkg/types/... ./pkg/inference/...
go test ./pkg/types/... -v -run TestLFM2Config
go test ./pkg/inference/... -v -run TestConfigTransfer
make test-go  # All existing Go tests pass
```

---

### T-002: BF16 CUDA Element-wise Kernels and cuBLAS GEMM

**Task ID**: T-002
**Task Name**: Implement BF16 CUDA kernels (RMSNorm, SiLU, add, mul) and BF16 cuBLAS GEMM
**Priority**: Critical
**Effort**: Medium (1-2 days)

#### Context & Background

**Source PRP Document**: `docs/prps/lfm2-support.md` - Task 2

**Feature Overview**: The entire LFM2 compute pipeline uses BF16 natively. This requires BF16 variants of all element-wise operations (RMSNorm, SiLU, add, mul) and a BF16 GEMM wrapper using cuBLAS cublasGemmEx with CUDA_R_16BF input type and CUBLAS_COMPUTE_32F compute type.

**Task Purpose**:
- **As a** CUDA compute backend
- **I need** BF16-native kernels for all building-block operations
- **So that** the LFM2 forward pass can execute entirely in BF16 without conversion overhead

#### Dependencies
- **Prerequisite Tasks**: T-001 (BFloat16 type definition)
- **Parallel Tasks**: T-006 (Model Config) can proceed independently
- **Blocked By**: Nothing

#### Technical Requirements

- **REQ-1**: BF16 GEMM must use cublasGemmEx with CUDA_R_16BF + CUBLAS_COMPUTE_32F (no CUBLAS_COMPUTE_16BF exists)
- **REQ-2**: All BF16 kernels must accumulate in FP32 for numerical stability
- **REQ-3**: BF16 requires compute capability >= 8.0 (Ampere: RTX 3090/A100+)
- **REQ-4**: BF16 GEMM output matches FP16 GEMM reference within 1e-2 tolerance

#### Implementation Details

**Files to Modify/Create**:
```
gpu/cuda/bf16_utils.cu    - CREATE: BF16 element-wise kernels
gpu/cuda/bf16_utils.h     - CREATE: BF16 kernel declarations
gpu/cuda/matmul.cu        - MODIFY: ADD cuda_gemm_bf16() using cublasGemmEx
gpu/cuda/matmul.h         - MODIFY: ADD cuda_gemm_bf16 declaration
```

**Code Patterns to Follow**:
- `gpu/cuda/kernels.cu` - Existing FP16 kernel pattern (cuda_rmsnorm, cuda_silu, cuda_add, cuda_mul)
- `gpu/cuda/matmul.cu:64-143` - Existing cuda_gemm_fp16 row-major/column-major conversion pattern

**Key Implementation Steps**:
1. Create bf16_utils.cu with `#include <cuda_bf16.h>` for `__nv_bfloat16` type
2. Implement cuda_bf16_rmsnorm: BF16 input/output, FP32 accumulation for variance
3. Implement cuda_bf16_silu, cuda_bf16_add, cuda_bf16_mul: BF16 input/output, FP32 intermediate
4. Implement cuda_bf16_to_fp32 and cuda_fp32_to_bf16 conversion kernels
5. Add cuda_gemm_bf16 to matmul.cu: same row-major logic as cuda_gemm_fp16 but using cublasGemmEx
6. Add cuda_gemm_bf16_bf16out variant (BF16 accumulation output)

**CRITICAL GOTCHA**: `cuda_bf16.h` requires `-arch=sm_80` or higher. The Makefile uses `-arch=native` which handles this on Ampere+ GPUs. On pre-Ampere GPUs, compilation will fail -- this is expected and documented.

#### Acceptance Criteria

```gherkin
Scenario 1: BF16 GEMM correctness
  Given two BF16 matrices A[128,2048] and B[2048,2048]
  When cuda_gemm_bf16 computes C = A @ B
  Then C matches cuda_gemm_fp16 reference within 1e-2 tolerance

Scenario 2: BF16 RMSNorm correctness
  Given a BF16 input vector of size 2048 and FP16 norm weights
  When cuda_bf16_rmsnorm is applied
  Then output matches FP16 cuda_rmsnorm within 1e-2 tolerance

Scenario 3: BF16 element-wise operations
  Given BF16 vectors a and b of size 2048
  When cuda_bf16_add(c, a, b) is called
  Then c[i] = a[i] + b[i] for all i (within BF16 precision)

Scenario 4: Graceful failure on pre-Ampere
  Given a GPU with compute capability < 8.0
  When BF16 operations are attempted
  Then a clear error message indicates BF16 requires Ampere+
```

**Rule-Based Criteria**:
- [ ] `make cuda` compiles without errors
- [ ] BF16 GEMM uses CUBLAS_COMPUTE_32F (never CUBLAS_COMPUTE_16BF)
- [ ] All BF16 kernels accumulate in FP32
- [ ] bf16_utils.h has proper extern "C" guards
- [ ] No changes to existing FP16/INT8 kernel code paths

#### Validation

```bash
make cuda                                    # Compiles new .cu files
cd tests && go test -tags cuda -run TestBF16GEMM -v
cd tests && go test -tags cuda -run TestBF16RMSNorm -v
make test                                    # All existing CUDA tests pass
```

---

### T-003: Depthwise Causal Conv1d CUDA Kernel

**Task ID**: T-003
**Task Name**: Implement depthwise causal conv1d CUDA kernel with prefill and decode paths
**Priority**: Critical
**Effort**: Medium (1-2 days)

#### Context & Background

**Source PRP Document**: `docs/prps/lfm2-support.md` - Task 3

**Feature Overview**: The LFM2 conv blocks use depthwise causal conv1d with kernel width 3. This requires two kernel variants: a prefill kernel that processes the full input sequence with causal padding, and a decode kernel that updates a single token using a circular buffer state. The conv state is maintained in FP32 for numerical stability.

**Task Purpose**:
- **As a** conv layer compute kernel
- **I need** efficient causal conv1d with state management
- **So that** conv blocks can execute correctly during both prefill and autoregressive decode

#### Dependencies
- **Prerequisite Tasks**: T-002 (BF16 utils for __nv_bfloat16 type)
- **Parallel Tasks**: T-005 (BF16 Attention) can proceed independently after T-002
- **Blocked By**: Nothing

#### Technical Requirements

- **REQ-1**: Prefill kernel applies causal padding (pad left by kernel_size-1, truncate to seq_len)
- **REQ-2**: Decode kernel uses FIFO state update (shift left, insert new value, 3-tap FIR)
- **REQ-3**: Conv state stored in FP32 on GPU, sized [batch, dim, width]
- **REQ-4**: State must be explicitly resetable (cuda_conv_state_reset) for sequence boundaries

#### Implementation Details

**Files to Modify/Create**:
```
gpu/cuda/conv.cu   - CREATE: Prefill + Decode kernels, state management
gpu/cuda/conv.h    - CREATE: Function declarations with extern "C"
```

**Code Patterns to Follow**:
- `docs/prps/lfm2-support.md:666-699` - Exact decode kernel pseudocode from PRP
- `docs/prps/lfm2-support.md:375-408` - Header file structure from PRP

**Key Implementation Steps**:
1. Implement cuda_causal_conv1d_fwd_bf16 (prefill): Grid=(batch,dim), Block=128. For each channel, apply causal padding then sliding window dot product
2. Implement cuda_causal_conv1d_update_bf16 (decode): Grid=(batch,ceil(dim/64)), Block=64. FIFO shift + 3-tap FIR
3. Implement cuda_conv_state_create: cudaMalloc FP32 buffer [batch, dim, width]
4. Implement cuda_conv_state_reset: cudaMemset to zero
5. Implement cuda_conv_state_free: cudaFree

**CRITICAL GOTCHA**: Conv weight in SafeTensors is [2048, 1, 3] but the kernel expects [2048, 3]. The reshape happens in the weight loader (T-008), not here. The kernel always receives [dim, width].

#### Acceptance Criteria

```gherkin
Scenario 1: Prefill produces correct output
  Given a known input sequence [1, 2048, 8] and conv weight [2048, 3]
  When cuda_causal_conv1d_fwd_bf16 processes the full sequence
  Then output matches HuggingFace causal_conv1d reference within 1e-3

Scenario 2: Decode matches prefill output
  Given the same input processed token-by-token via update kernel
  When comparing the last-token output of decode vs prefill
  Then they match within 1e-3

Scenario 3: State reset prevents cross-sequence leakage
  Given a conv state with non-zero values from a previous sequence
  When cuda_conv_state_reset is called and a new sequence begins
  Then the first token output is identical to a fresh-state computation

Scenario 4: State management lifecycle
  Given cuda_conv_state_create allocates memory
  When cuda_conv_state_free is called
  Then GPU memory is released without leaks
```

**Rule-Based Criteria**:
- [ ] `make cuda` compiles conv.cu without errors
- [ ] Prefill kernel handles seq_len=1 edge case (single token prefill)
- [ ] Decode kernel unrolls for width=3 (fits in registers)
- [ ] FP32 accumulation used in both prefill and decode
- [ ] State buffer sized correctly: batch * dim * width * sizeof(float)

#### Validation

```bash
make cuda
cd tests && go test -tags cuda -run TestCausalConv1dPrefill -v
cd tests && go test -tags cuda -run TestCausalConv1dDecode -v
cd tests && go test -tags cuda -run TestConvStateReset -v
```

---

### T-004: LIV Conv Layer Forward Pass

**Task ID**: T-004
**Task Name**: Implement complete LIV conv block forward pass (conv_layer.cu)
**Priority**: Critical
**Effort**: Medium (1-2 days)

#### Context & Background

**Source PRP Document**: `docs/prps/lfm2-support.md` - Task 4

**Feature Overview**: The LIV conv block is the core operator for the 10 conv layers. Flow: RMSNorm -> in_proj(2048->6144) -> chunk(B,C,x) -> Bx=B*x -> conv1d(Bx) -> y=C*conv_out -> out_proj -> residual -> RMSNorm -> SwiGLU FFN -> residual.

**Task Purpose**:
- **As a** conv layer executor
- **I need** a complete forward pass combining all BF16 kernels
- **So that** a single CUDA call can process an entire conv layer

#### Dependencies
- **Prerequisite Tasks**: T-002 (BF16 kernels), T-003 (Conv1d kernel)
- **Parallel Tasks**: T-005 (BF16 Attention) can proceed in parallel
- **Blocked By**: Nothing

#### Technical Requirements

- **REQ-1**: ConvLayerWeights struct holds all weights (in_proj, conv, out_proj, norms, FFN) on GPU
- **REQ-2**: Forward pass branches on seq_len==1 (decode via update kernel) vs seq_len>1 (prefill)
- **REQ-3**: Shared FFN code (SwiGLU) with attention layers -- identical computation
- **REQ-4**: All temporary buffers allocated per-call (or preallocated in future optimization)

#### Implementation Details

**Files to Modify/Create**:
```
gpu/engine/conv_layer.cu  - CREATE: ConvLayerWeights struct, weight upload, forward pass
gpu/engine/conv_layer.h   - CREATE: C declarations for conv layer functions
```

**Code Patterns to Follow**:
- `gpu/engine/layer.cu:34-69` - LayerWeights struct pattern
- `gpu/engine/layer.cu:378-603` - cuda_layer_forward pattern (buffer alloc, computation, cleanup)
- `docs/prps/lfm2-support.md:704-743` - Exact forward pass pseudocode from PRP

**Key Implementation Steps**:
1. Define ConvLayerWeights struct with all BF16 weight pointers
2. Implement cuda_create_conv_layer_weights_bf16: upload host BF16 data to GPU (no INT8 quantization -- BF16 native)
3. Implement cuda_conv_layer_forward_bf16 following the 12-step pseudocode from PRP
4. Handle seq=1 vs seq>1 dispatch for conv1d
5. Share FFN code via the existing SwiGLU pattern (RMSNorm -> gate/up -> SiLU -> mul -> down -> residual)
6. Implement cuda_free_conv_layer_weights for cleanup

#### Acceptance Criteria

```gherkin
Scenario 1: Conv layer forward produces correct output
  Given ConvLayerWeights loaded with known BF16 data
  When cuda_conv_layer_forward_bf16 is called with position=0
  Then output matches step-by-step manual computation within 1e-2

Scenario 2: Conv layer with FFN produces correct residual
  Given a conv layer forward pass completes
  When the output is checked
  Then it includes both the conv block residual and FFN residual

Scenario 3: Decode mode (seq=1) uses update kernel
  Given position > 0 and seq_len = 1
  When cuda_conv_layer_forward_bf16 is called
  Then the decode conv1d update path is used (not prefill)

Scenario 4: Weight lifecycle management
  Given cuda_create_conv_layer_weights_bf16 allocates GPU memory
  When cuda_free_conv_layer_weights is called
  Then all GPU allocations (8 weight buffers) are freed
```

**Rule-Based Criteria**:
- [ ] `make cuda` compiles conv_layer.cu without errors
- [ ] ConvLayerWeights struct mirrors the C header from PRP
- [ ] Forward pass includes 12 computation steps as specified
- [ ] Temporary buffers freed on both success and error paths (goto cleanup pattern)
- [ ] No INT8 quantization -- BF16 weights used directly

#### Validation

```bash
make cuda
cd tests && go test -tags cuda -run TestConvLayerForward -v
cd tests && go test -tags cuda -run TestConvLayerWithFFN -v
```

---

### T-005: BF16 Attention Layer with QK LayerNorm

**Task ID**: T-005
**Task Name**: Add BF16 attention forward pass with per-head QK RMSNorm
**Priority**: Critical
**Effort**: Medium (1-2 days)

#### Context & Background

**Source PRP Document**: `docs/prps/lfm2-support.md` - Task 5

**Feature Overview**: The 6 attention layers in LFM2 use standard GQA attention but with two additions: (1) BF16 compute throughout, and (2) per-head RMSNorm applied to Q and K AFTER projection but BEFORE RoPE. Weight shape for Q/K norms is [head_dim] = [64].

**Task Purpose**:
- **As a** BF16 attention executor
- **I need** QK LayerNorm + BF16 GQA attention
- **So that** LFM2 attention layers produce correct output

#### Dependencies
- **Prerequisite Tasks**: T-002 (BF16 kernels, BF16 GEMM)
- **Parallel Tasks**: T-003, T-004 (Conv kernels)
- **Blocked By**: Nothing

#### Technical Requirements

- **REQ-1**: QK LayerNorm is RMSNorm applied per-head AFTER Q/K projection, BEFORE RoPE
- **REQ-2**: Q norm weight shape [head_dim], K norm weight shape [head_dim] (shared across heads)
- **REQ-3**: RoPE must use interleaved style for LFM2 with theta=1000000
- **REQ-4**: BF16 attention with FP32 softmax accumulation

#### Implementation Details

**Files to Modify/Create**:
```
gpu/engine/layer.cu       - MODIFY: ADD q_layernorm, k_layernorm to LayerWeights struct
                           - ADD: cuda_attn_layer_forward_bf16() function
gpu/engine/layer.h        - MODIFY: ADD new struct fields and function declaration
gpu/cuda/attention.cu     - MODIFY: ADD cuda_attention_with_kvcache_bf16()
                           - ADD: cuda_basic_attention_gqa_bf16()
gpu/cuda/attention.h      - MODIFY: ADD BF16 function declarations
```

**Code Patterns to Follow**:
- `gpu/engine/layer.cu:378-603` - Existing cuda_layer_forward (replicate for BF16 variant)
- `gpu/cuda/attention.cu` - Existing attention kernels (add BF16 variants)

**Key Implementation Steps**:
1. Add q_layernorm and k_layernorm (half*/bfloat16*) to LayerWeights struct
2. Implement cuda_attn_layer_forward_bf16: same as cuda_layer_forward but:
   a. BF16 GEMM for Q/K/V/O projections
   b. After Q projection: per-head RMSNorm (reshape Q to [batch, seq, num_heads, head_dim], norm each head)
   c. After K projection: per-head RMSNorm (reshape K similarly)
   d. RoPE with theta=1000000
   e. BF16 attention (FP32 softmax)
   f. BF16 FFN
3. Implement BF16 attention kernels (cuda_attention_with_kvcache_bf16, cuda_basic_attention_gqa_bf16)

**CRITICAL GOTCHA**: QK LayerNorm BEFORE RoPE, not after. The ordering is: Projection -> LayerNorm -> RoPE -> Attention.

#### Acceptance Criteria

```gherkin
Scenario 1: QK LayerNorm applied correctly
  Given Q projection output of shape [1, 1, 32, 64]
  When per-head RMSNorm is applied with weight [64]
  Then each of the 32 heads is independently normalized

Scenario 2: Ordering is correct (Proj -> Norm -> RoPE)
  Given an attention layer forward pass
  When examining the computation order
  Then QK LayerNorm occurs AFTER projection but BEFORE RoPE application

Scenario 3: BF16 attention matches FP16 reference
  Given identical inputs in BF16 and FP16
  When both attention variants process the same data
  Then outputs match within 1e-2 tolerance

Scenario 4: KV cache works with BF16 attention
  Given a BF16 attention layer during decode (seq=1)
  When attention uses KV cache
  Then cached K/V values are correctly utilized
```

**Rule-Based Criteria**:
- [ ] Existing cuda_layer_forward (FP16/INT8) unchanged
- [ ] QK LayerNorm weight shape is [head_dim], not [hidden_size]
- [ ] Per-head RMSNorm loops over num_heads, applying to each [head_dim] slice
- [ ] BF16 attention uses FP32 for softmax computation
- [ ] RoPE theta configurable (1000000 for LFM2)

#### Validation

```bash
make cuda
cd tests && go test -tags cuda -run TestBF16Attention -v
cd tests && go test -tags cuda -run TestQKLayerNorm -v
```

---

### T-006: Model Config Detection and Registration

**Task ID**: T-006
**Task Name**: Add LFM2 model family detection, config parsing, and model alias
**Priority**: High
**Effort**: Short (4-6 hours)

#### Context & Background

**Source PRP Document**: `docs/prps/lfm2-support.md` - Task 7

**Feature Overview**: The engine needs to detect LFM2 from config.json, parse its unique fields (layer_types array, conv_L_cache, etc.), and register it as a known model family with the correct RoPE style and chat template.

**Task Purpose**:
- **As a** model configuration system
- **I need** LFM2 config parsing and family detection
- **So that** the engine correctly identifies and configures LFM2 models

#### Dependencies
- **Prerequisite Tasks**: T-001 (Config fields exist)
- **Parallel Tasks**: All Phase 2 tasks
- **Blocked By**: Nothing

#### Technical Requirements

- **REQ-1**: config.json fields: layer_types (array), conv_L_cache (int), conv_dim (int), conv_bias (bool), tie_word_embeddings (bool), torch_dtype (string)
- **REQ-2**: LFM2 uses interleaved RoPE with theta=1000000
- **REQ-3**: LFM2 uses ChatML template
- **REQ-4**: eos_token_id=7 (im_end), NOT 2 (endoftext)

#### Implementation Details

**Files to Modify/Create**:
```
pkg/model/model_config.go  - MODIFY: ADD layer_types, conv_L_cache, conv_dim, conv_bias, tie_word_embeddings, torch_dtype JSON fields to ModelConfig
                            - MODIFY: ADD "lfm2" to KnownModelFamilies map
                            - MODIFY: ADD LFM2 pattern to DetectModelFamily()
                            - MODIFY: ADD LFM2 to DetectRoPEStyle() -> interleaved
                            - MODIFY: ADD LFM2/ChatML to DetectChatTemplate()
configs/models.yaml         - MODIFY: ADD lfm2-1.2b-thinking alias
Makefile                    - ADD: download-lfm2-thinking and run-lfm2-thinking targets
```

**Code Patterns to Follow**:
- `pkg/model/model_config.go:210-236` - KnownModelFamilies map pattern
- `pkg/model/model_config.go:239-260` - DetectModelFamily pattern

**Key Implementation Steps**:
1. Add new JSON-tagged fields to ModelConfig struct
2. Add "lfm2" entry to KnownModelFamilies with RoPEStyleInterleaved and ChatML template
3. Update DetectModelFamily to match "lfm" in model_type or architectures array
4. Update DetectRoPEStyle to return interleaved for LFM2
5. Update DetectChatTemplate to return ChatML for LFM2
6. Add model alias to configs/models.yaml
7. Add Makefile targets

#### Acceptance Criteria

```gherkin
Scenario 1: LFM2 config.json parses correctly
  Given a config.json with model_type="lfm2" and layer_types array
  When LoadModelConfig parses it
  Then all layer_types, conv_L_cache, conv_dim fields are populated

Scenario 2: Model family detection works
  Given model name containing "lfm"
  When DetectModelFamily is called
  Then it returns the lfm2 ModelFamily with correct RoPE and template

Scenario 3: Existing model detection unchanged
  Given model name "tinyllama"
  When DetectModelFamily is called
  Then it still returns the tinyllama family (no regression)
```

**Rule-Based Criteria**:
- [ ] ModelConfig JSON tags match HuggingFace config.json field names
- [ ] KnownModelFamilies["lfm2"] has RoPEStyle=Interleaved
- [ ] All existing model families still detected correctly
- [ ] configs/models.yaml has proper HuggingFace repo reference

#### Validation

```bash
go test ./pkg/model/... -v -run TestLoadModelConfig
go test ./pkg/model/... -v -run TestDetectModelFamily
make test-go
```

---

### T-007: Go-CUDA CGO Bindings for All New Functions

**Task ID**: T-007
**Task Name**: Add CGO bindings for conv layer, BF16 GEMM, conv state, and BF16 attention
**Priority**: Critical
**Effort**: Medium (1-2 days)

#### Context & Background

**Source PRP Document**: `docs/prps/lfm2-support.md` - Task 6

**Feature Overview**: Every new CUDA function needs three files updated: gpu.go (real implementation), gpu_stub.go (no-GPU fallback), and gpu.h (C header). This task creates Go wrappers for all CUDA functions from T-002 through T-005.

**Task Purpose**:
- **As a** Go inference layer
- **I need** CGO bindings for all new CUDA operations
- **So that** the Go inference engine can invoke BF16 and conv CUDA kernels

#### Dependencies
- **Prerequisite Tasks**: T-002 (BF16 kernels), T-003 (Conv1d), T-004 (Conv layer), T-005 (BF16 attention)
- **Parallel Tasks**: T-008 (Weight loader) can start early
- **Blocked By**: All CUDA kernels must have stable C API

#### Technical Requirements

- **REQ-1**: Every new C function must appear in all three files (gpu.go, gpu_stub.go, gpu.h)
- **REQ-2**: Stub functions must return ErrNotImplemented (not nil) so non-CUDA builds fail clearly
- **REQ-3**: Go wrapper must handle unsafe.Pointer conversions safely
- **REQ-4**: CheckBF16Support() must query compute capability and return true only for >= 8.0

#### Implementation Details

**Files to Modify/Create**:
```
gpu/bindings/gpu.go       - ADD: ConvLayerWeights Go struct, CreateConvLayerWeightsBF16(), ConvLayerForwardBF16(),
                             FreeConvLayerWeights(), GEMM_BF16(), ConvStateCreate/Reset/Free(),
                             AttnLayerForwardBF16(), CheckBF16Support()
gpu/bindings/gpu_stub.go  - ADD: stubs for ALL new functions returning ErrNotImplemented
gpu/bindings/gpu.h        - ADD: C declarations for all new functions
```

**Code Patterns to Follow**:
- `gpu/bindings/gpu.go:659-697` - CreateLayerWeightsFromHost pattern (many parameters)
- `gpu/bindings/gpu.go:700-747` - LayerForward pattern
- `gpu/bindings/gpu_stub.go:574-581` - CreateLayerWeightsFromHost stub pattern

**Key Implementation Steps**:
1. Add C header declarations in gpu.h for all new functions
2. Add Go wrapper structs (ConvLayerWeights) and methods in gpu.go
3. Implement each wrapper following the existing CGO pattern (C.function_name with C type casts)
4. Add matching stubs in gpu_stub.go returning ErrNotImplemented
5. Add CheckBF16Support using cudaGetDeviceProperties to check compute capability

**New functions to bind (complete list)**:
- `cuda_gemm_bf16` / `cuda_gemm_bf16_bf16out`
- `cuda_bf16_rmsnorm` / `cuda_bf16_silu` / `cuda_bf16_add` / `cuda_bf16_mul`
- `cuda_bf16_to_fp32` / `cuda_fp32_to_bf16`
- `cuda_causal_conv1d_fwd_bf16` / `cuda_causal_conv1d_update_bf16`
- `cuda_conv_state_create` / `cuda_conv_state_reset` / `cuda_conv_state_free`
- `cuda_create_conv_layer_weights_bf16` / `cuda_conv_layer_forward_bf16` / `cuda_free_conv_layer_weights`
- `cuda_attn_layer_forward_bf16`

#### Acceptance Criteria

```gherkin
Scenario 1: CUDA build compiles
  Given all CUDA .cu files compiled into libgpu_engine.so
  When go build -tags cuda ./gpu/bindings/...
  Then compilation succeeds with no undefined symbols

Scenario 2: Non-CUDA build compiles
  Given the !cuda build tag
  When go build ./gpu/bindings/...
  Then compilation succeeds with all stubs

Scenario 3: Stub functions return ErrNotImplemented
  Given a non-CUDA build
  When ConvLayerForwardBF16() is called
  Then ErrNotImplemented is returned

Scenario 4: BF16 support check
  Given an Ampere GPU (compute 8.0+)
  When CheckBF16Support() is called
  Then it returns true
```

**Rule-Based Criteria**:
- [ ] Every function in gpu.go has a matching stub in gpu_stub.go
- [ ] Every C function called from gpu.go is declared in gpu.h
- [ ] `go build -tags cuda ./gpu/bindings/...` succeeds
- [ ] `go build ./gpu/bindings/...` succeeds (stub build)
- [ ] No unsafe.Pointer usage without nil checks

#### Validation

```bash
go build -tags cuda ./gpu/bindings/...    # CUDA build
go build ./gpu/bindings/...               # Stub build
go vet -tags cuda ./gpu/bindings/...
make build-coordinator                     # Full binary build
```

---

### T-008: LFM2 Weight Loading

**Task ID**: T-008
**Task Name**: Extend weight loader for LFM2 tensor paths, conv weight reshape, and tied embeddings
**Priority**: High
**Effort**: Medium (1 day)

#### Context & Background

**Source PRP Document**: `docs/prps/lfm2-support.md` - Task 8

**Feature Overview**: LFM2 uses different tensor naming than Llama (e.g., `model.layers.{i}.conv.in_proj.weight` for conv layers, `model.layers.{i}.self_attn.q_layernorm.weight` for QK norms). Conv weights need reshaping from [hidden, 1, kernel] to [hidden, kernel]. BF16 weights should be kept as BF16 bytes when Dtype="bf16".

**Task Purpose**:
- **As a** weight loader
- **I need** LFM2 tensor path resolution and BF16 native loading
- **So that** all conv and attention layer weights load correctly from SafeTensors

#### Dependencies
- **Prerequisite Tasks**: T-001 (Config with LayerTypes), T-006 (Model config detection)
- **Parallel Tasks**: T-007 (CGO bindings)
- **Blocked By**: Nothing

#### Technical Requirements

- **REQ-1**: Conv layer tensor paths: `model.layers.{i}.conv.in_proj.weight`, `.conv.conv.weight`, `.conv.out_proj.weight`
- **REQ-2**: Conv weight reshape: squeeze dim 1 from [hidden, 1, kernel] to [hidden, kernel]
- **REQ-3**: Tied embeddings: when TieEmbeddings=true, lm_head.weight == embed_tokens.weight (no separate allocation)
- **REQ-4**: BF16 loading: when Dtype="bf16", skip BF16->FP16 conversion, keep raw BF16 bytes

#### Implementation Details

**Files to Modify/Create**:
```
pkg/model/loader.go   - ADD: LoadConvLayerWeights(layerID) function
                        - ADD: LoadAttentionLayerWeightsLFM2(layerID) function (with QK norms)
                        - MODIFY: LoadTensor BF16 path: skip conversion when Dtype="bf16"
                        - MODIFY: LoadLMHead to check TieEmbeddings flag
pkg/model/weights.go  - ADD: ConvLayerWeights struct
                        - MODIFY: LayerWeights to add QLayerNorm, KLayerNorm fields
```

**Code Patterns to Follow**:
- `pkg/model/loader.go:292-349` - Existing LoadLayerWeights pattern
- `pkg/model/loader.go:352-366` - LoadLMHead fallback pattern

**Key Implementation Steps**:
1. Add ConvLayerWeights struct with fields matching PRP data model
2. Implement LoadConvLayerWeights: load conv-specific tensors with correct paths
3. Implement reshape logic: if shape is [hidden, 1, kernel], produce [hidden, kernel] by dropping middle dim
4. Implement LoadAttentionLayerWeightsLFM2: like LoadLayerWeights but with QK norm loading and LFM2 tensor paths
5. Add BF16 native loading flag: when Dtype="bf16", LoadTensor skips ConvertBF16ToFP16
6. Modify LoadLMHead to accept TieEmbeddings parameter

**LFM2 Conv Layer Tensor Paths**:
```
model.layers.{i}.conv.in_proj.weight       [3*hidden, hidden]
model.layers.{i}.conv.conv.weight          [hidden, 1, kernel] -> reshape to [hidden, kernel]
model.layers.{i}.conv.out_proj.weight      [hidden, hidden]
model.layers.{i}.operator_norm.weight      [hidden]
model.layers.{i}.ffn_norm.weight           [hidden]
model.layers.{i}.feed_forward.w1.weight    [intermediate, hidden]
model.layers.{i}.feed_forward.w2.weight    [hidden, intermediate]
model.layers.{i}.feed_forward.w3.weight    [intermediate, hidden]
```

**LFM2 Attention Layer Tensor Paths**:
```
model.layers.{i}.self_attn.q_proj.weight       [hidden, hidden]
model.layers.{i}.self_attn.k_proj.weight       [kv_dim, hidden]
model.layers.{i}.self_attn.v_proj.weight       [kv_dim, hidden]
model.layers.{i}.self_attn.o_proj.weight       [hidden, hidden]
model.layers.{i}.self_attn.q_layernorm.weight  [head_dim]
model.layers.{i}.self_attn.k_layernorm.weight  [head_dim]
model.layers.{i}.operator_norm.weight          [hidden]
model.layers.{i}.ffn_norm.weight               [hidden]
model.layers.{i}.feed_forward.w1/w2/w3.weight  (same as conv FFN)
```

#### Acceptance Criteria

```gherkin
Scenario 1: Conv layer weights load correctly
  Given an LFM2 SafeTensors file with conv layer tensors
  When LoadConvLayerWeights(0) is called
  Then in_proj, conv, out_proj, norms, and FFN weights are loaded
  And conv weight is reshaped from [2048,1,3] to [2048,3]

Scenario 2: Attention layer weights with QK norms
  Given LFM2 attention layer tensors
  When LoadAttentionLayerWeightsLFM2(10) is called
  Then Q/K/V/O, QK norms, operator norm, ffn norm, and FFN weights are loaded
  And QK norm shapes are [64] (head_dim)

Scenario 3: Tied embeddings share allocation
  Given TieEmbeddings=true
  When LoadLMHead is called
  Then it returns the same data as LoadEmbeddings (no separate load)

Scenario 4: BF16 native loading preserves bytes
  Given Dtype="bf16" and a BF16 tensor in SafeTensors
  When LoadTensor is called
  Then raw BF16 bytes are returned (no FP16 conversion)

Scenario 5: Existing Llama loading unchanged
  Given a Llama model with Dtype="" and TieEmbeddings=false
  When LoadLayerWeights is called
  Then existing behavior (BF16->FP16 conversion) is preserved
```

**Rule-Based Criteria**:
- [ ] Conv weight reshape drops dim 1 only when shape is [N, 1, K]
- [ ] LoadTensor BF16 conversion skipped only when Dtype="bf16" explicitly
- [ ] Existing LoadLayerWeights function not modified (new functions added)
- [ ] All tensor path strings match HuggingFace model exactly

#### Validation

```bash
go test ./pkg/model/... -v -run TestLoadConvLayerWeights
go test ./pkg/model/... -v -run TestLoadLFM2AttentionWeights
go test ./pkg/model/... -v -run TestTiedEmbeddings
go test ./pkg/model/... -v -run TestBF16NativeLoading
make test-go
```

---

### T-009: Hybrid Cache Manager

**Task ID**: T-009
**Task Name**: Implement ConvStateCache and HybridCacheManager for mixed layer types
**Priority**: High
**Effort**: Short (4-6 hours)

#### Context & Background

**Source PRP Document**: `docs/prps/lfm2-support.md` - Task 9

**Feature Overview**: LFM2 needs a hybrid cache: conv layers use fixed-size state buffers [batch, hidden, kernel_size] in FP32, while attention layers use growing KV caches. The HybridCacheManager creates the right cache type per layer based on config.IsConvLayer().

**Task Purpose**:
- **As a** cache manager
- **I need** per-layer-type cache allocation and lifecycle management
- **So that** conv layers have state buffers and attention layers have KV caches

#### Dependencies
- **Prerequisite Tasks**: T-001 (Config with IsConvLayer), T-007 (CGO bindings for conv state)
- **Parallel Tasks**: T-008 (Weight loading)
- **Blocked By**: Nothing

#### Technical Requirements

- **REQ-1**: Conv state is fixed size: [batch, hidden_size, kernel_size] FP32 on GPU
- **REQ-2**: Conv state must be zero-initialized at start of each new sequence
- **REQ-3**: For models without LayerTypes (nil), all caches are KV caches (backward compat)
- **REQ-4**: ResetAll must reset both conv states and KV caches

#### Implementation Details

**Files to Modify/Create**:
```
pkg/inference/kvcache.go  - ADD: ConvStateCache struct and methods
                           - ADD: HybridCacheManager struct and methods
                           - MODIFY: Extend KVCacheManager or add parallel manager
```

**Code Patterns to Follow**:
- `pkg/inference/kvcache.go:10-16` - KVCacheConfig pattern
- `pkg/inference/kvcache.go:150-207` - KVCacheManager pattern

**Key Implementation Steps**:
1. Define ConvStateCache struct: layerID, hiddenSize, kernelSize, gpuPtr (unsafe.Pointer)
2. Implement Create(): calls bindings.ConvStateCreate
3. Implement Reset(): calls bindings.ConvStateReset (zero state)
4. Implement Free(): calls bindings.ConvStateFree
5. Define HybridCacheManager: kvCaches map[int], convCaches map[int], config
6. Implement InitForModel(): iterate layers, create right cache type per layer
7. Implement ResetAll(): reset both cache types
8. Implement Close(): free all caches

#### Acceptance Criteria

```gherkin
Scenario 1: Hybrid cache creates correct cache types
  Given LFM2 config with 10 conv + 6 attention layers
  When HybridCacheManager.InitForModel() is called
  Then 10 ConvStateCaches and 6 KV caches are created

Scenario 2: Conv state resets between sequences
  Given an active conv state from a previous request
  When ResetAll() is called
  Then conv states are zeroed and KV caches are cleared

Scenario 3: Backward compatible with Llama
  Given a Llama config (nil LayerTypes)
  When HybridCacheManager.InitForModel() is called
  Then all layers get KV caches (zero conv caches)

Scenario 4: Resource cleanup
  Given an active HybridCacheManager
  When Close() is called
  Then all GPU memory for conv states and KV caches is freed
```

**Rule-Based Criteria**:
- [ ] Conv state size calculation: batch * hidden * kernel * 4 (FP32)
- [ ] Existing KVCacheManager code unchanged
- [ ] Nil LayerTypes = all KV caches (backward compat)
- [ ] Conv state zeroed at creation and on reset (prevent cross-request leakage)

#### Validation

```bash
go test ./pkg/inference/... -v -run TestHybridCacheManager
go test ./pkg/inference/... -v -run TestConvStateCache
make test-go
```

---

### T-010: Inference Engine Hybrid Layer Loop

**Task ID**: T-010
**Task Name**: Modify inference engine to branch forwardAllLayersHidden by layer type
**Priority**: Critical
**Effort**: Medium (1-2 days)

#### Context & Background

**Source PRP Document**: `docs/prps/lfm2-support.md` - Task 10

**Feature Overview**: The core engine loop in `forwardAllLayersHidden()` currently assumes all layers are attention+FFN. For LFM2, it must dispatch to either forwardConvLayer() or forwardLayer() based on config.IsConvLayer(layerID). Conv layers produce no K,V output and do not update KV caches.

**Task Purpose**:
- **As a** inference engine
- **I need** to dispatch to the correct forward function per layer type
- **So that** the hybrid conv+attention architecture processes correctly

#### Dependencies
- **Prerequisite Tasks**: T-007 (CGO bindings), T-008 (Weight loader), T-009 (Hybrid cache)
- **Parallel Tasks**: T-011 (Tokenizer), T-012 (Scheduler)
- **Blocked By**: All CUDA + binding work complete

#### Technical Requirements

- **REQ-1**: Conv layers: no K,V output, no KV cache update
- **REQ-2**: Attention layers: standard forward with K,V and KV cache update
- **REQ-3**: When LayerTypes is nil, all layers use the existing attention path (backward compat)
- **REQ-4**: Conv state resets at sequence boundaries (handled by cache manager)

#### Implementation Details

**Files to Modify/Create**:
```
pkg/inference/engine.go         - MODIFY: forwardAllLayersHidden() to branch by layer type
                                 - ADD: forwardConvLayer() method
pkg/inference/cuda_executor.go  - ADD: convLayerWeights map, convStates map
                                 - ADD: LoadConvLayer(layerID, weights) method
                                 - ADD: ForwardConv(ctx, layerID, hidden, position) method
                                 - MODIFY: Close() to free conv resources
                                 - MODIFY: ResetKVCache() to reset conv states
pkg/inference/engine_gpu.go     - MODIFY: InitializeGPU layer loading loop to branch by type
```

**Code Patterns to Follow**:
- `pkg/inference/engine.go:628-667` - Existing forwardAllLayersHidden loop
- `pkg/inference/cuda_executor.go:106-144` - Existing LoadLayer pattern
- `pkg/inference/cuda_executor.go:150-273` - Existing Forward pattern
- `pkg/inference/engine_gpu.go:94-169` - Existing layer loading loop

**Key Implementation Steps**:
1. Modify forwardAllLayersHidden: add `if e.config.IsConvLayer(layerID)` branch
2. Implement forwardConvLayer: calls executor ForwardConv, returns only hidden (no K,V)
3. Add conv weight maps and state maps to CUDALayerExecutor
4. Implement LoadConvLayer: loads ConvLayerWeights to GPU via bindings
5. Implement ForwardConv: calls bindings.ConvLayerForwardBF16
6. Modify InitializeGPU: for each layer, check IsConvLayer to decide which weights to load
7. Ensure KV cache update only for attention layers

**Pseudocode for modified forwardAllLayersHidden**:
```go
for layerID := 0; layerID < e.config.NumLayers; layerID++ {
    if e.config.IsConvLayer(layerID) {
        output, err := e.forwardConvLayer(ctx, layerID, current, position)
        // No K,V, no KV cache update
        current = output
    } else {
        output, k, v, err := e.forwardLayer(ctx, layerID, current, position)
        // Standard KV cache update
        if cache, ok := e.kvCaches.GetCache(layerID); ok && cache.IsLocal() {
            cache.Update(ctx, k, v, position)
        }
        current = output
    }
}
```

#### Acceptance Criteria

```gherkin
Scenario 1: LFM2 processes all 16 layers correctly
  Given LFM2 config with 10 conv + 6 attention layers
  When forwardAllLayersHidden executes
  Then layers 0-9 go through forwardConvLayer
  And layers 10-15 go through forwardLayer
  And output is valid hidden state

Scenario 2: Conv layers skip KV cache update
  Given a conv layer (layerID < 10)
  When forwardConvLayer returns
  Then no KV cache update is attempted for this layer

Scenario 3: Llama model unchanged
  Given a Llama config (nil LayerTypes)
  When forwardAllLayersHidden executes
  Then all layers use forwardLayer (existing path)
  And all layers update KV cache

Scenario 4: GPU weight loading branches correctly
  Given InitializeGPU loading an LFM2 model
  When iterating layers
  Then conv layers load ConvLayerWeights
  And attention layers load AttentionLayerWeights (with QK norms)
```

**Rule-Based Criteria**:
- [ ] Nil LayerTypes = all layers use forwardLayer (zero changes to existing behavior)
- [ ] Conv layers return only hidden state (no K, V)
- [ ] forwardConvLayer passes conv state from HybridCacheManager
- [ ] All existing tests still pass
- [ ] GPU memory cleanup includes conv weights and states

#### Validation

```bash
go build -tags cuda ./pkg/inference/...
go test -tags cuda ./pkg/inference/... -v
make test                                  # All CUDA tests
make test-go                               # All Go tests
```

---

### T-011: ChatML Template and Tokenizer Verification

**Task ID**: T-011
**Task Name**: Add ChatML template with thinking token support and verify BPE tokenizer for 65K vocab
**Priority**: High
**Effort**: Short (4-6 hours)

#### Context & Background

**Source PRP Document**: `docs/prps/lfm2-support.md` - Task 11

**Feature Overview**: LFM2 uses ChatML format (`<|im_start|>`, `<|im_end|>`) and has thinking tokens `<think>`/`</think>` (IDs 64400/64401) that should be configurable. The BPE tokenizer must handle a 65K vocabulary with eos_token_id=7.

**Task Purpose**:
- **As a** chat interface
- **I need** correct ChatML formatting and thinking token handling
- **So that** LFM2 receives properly formatted prompts and generates with correct stopping behavior

#### Dependencies
- **Prerequisite Tasks**: T-006 (Model config detection for ChatML)
- **Parallel Tasks**: T-010 (Inference engine)
- **Blocked By**: Nothing

#### Technical Requirements

- **REQ-1**: ChatML format: `<|startoftext|><|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n`
- **REQ-2**: eos_token_id=7 (`<|im_end|>`), NOT 2
- **REQ-3**: Thinking tokens `<think>`(64400)/`</think>`(64401) configurable via show_thinking
- **REQ-4**: BPE tokenizer must correctly encode/decode 65K vocab

#### Implementation Details

**Files to Modify/Create**:
```
pkg/model/chat_template.go   - ADD: ChatMLTemplate struct implementing ChatTemplate
                               - ADD: thinking token strip logic
pkg/model/model_config.go    - MODIFY: DetectChatTemplate for ChatML / <|im_start|> detection
pkg/model/tokenizer.go       - VERIFY: BPE handles 65K vocab
                               - ADD: special token constants for LFM2
```

**Code Patterns to Follow**:
- `pkg/model/chat_template.go:114-157` - TinyLlamaChatTemplate pattern
- `pkg/model/model_config.go:97-134` - DetectChatTemplate pattern

**Key Implementation Steps**:
1. Create ChatMLTemplate struct implementing ChatTemplate interface
2. Implement Format(): produce `<|im_start|>role\ncontent<|im_end|>\n` per message
3. Add thinking token stripping: configurable removal of `<think>...</think>` blocks
4. Update DetectChatTemplate to detect `<|im_start|>` pattern in raw template
5. Verify BPE tokenizer loads tokenizer.json with 65K entries
6. Add LFM2 special token constants (eos=7, bos=1, pad=0, think=64400/64401)

#### Acceptance Criteria

```gherkin
Scenario 1: ChatML format is correct
  Given messages [{role:"system", content:"You are helpful"}, {role:"user", content:"Hello"}]
  When ChatMLTemplate.Format() is called
  Then output matches: "<|im_start|>system\nYou are helpful<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n"

Scenario 2: EOS token is 7
  Given the LFM2 tokenizer
  When EOSToken() is called
  Then it returns 7 (not 2)

Scenario 3: Thinking tokens can be stripped
  Given output text "<think>Let me reason...</think>The answer is 4"
  When show_thinking=false
  Then returned text is "The answer is 4"

Scenario 4: ChatML detection from raw template
  Given a tokenizer_config.json with "<|im_start|>" in chat_template
  When DetectChatTemplate is called
  Then ChatMLTemplate is returned
```

**Rule-Based Criteria**:
- [ ] ChatMLTemplate implements ChatTemplate interface fully
- [ ] eos_token_id=7 is hardcoded for LFM2 (not relying on default 2)
- [ ] Existing chat templates (Llama2, Llama3, TinyLlama, Mistral) unaffected
- [ ] Thinking token stripping handles edge cases (no think tags, nested, etc.)

#### Validation

```bash
go test ./pkg/model/... -v -run TestChatMLTemplate
go test ./pkg/model/... -v -run TestDetectChatTemplate
go test ./pkg/model/... -v -run TestLFM2Tokenizer
make test-go
```

---

### T-012: Scheduler Per-Layer Memory Estimation

**Task ID**: T-012
**Task Name**: Add per-layer-type VRAM estimation for heterogeneous layer scheduling
**Priority**: Medium
**Effort**: Short (3-4 hours)

#### Context & Background

**Source PRP Document**: `docs/prps/lfm2-support.md` - Task 12

**Feature Overview**: Conv and attention layers have different memory footprints. Conv layers have no KV cache but have conv state and in_proj weights (3x larger). The scheduler must estimate memory per layer type for correct VRAM assignment.

**Task Purpose**:
- **As a** scheduler
- **I need** accurate per-layer-type memory estimation
- **So that** layer-to-GPU assignments are feasible for heterogeneous models

#### Dependencies
- **Prerequisite Tasks**: T-001 (Config with layer type info)
- **Parallel Tasks**: All other tasks
- **Blocked By**: Nothing

#### Technical Requirements

- **REQ-1**: Conv layer estimate: in_proj(3*h*h*2B) + conv(h*k*2B) + out_proj(h*h*2B) + FFN + conv_state(h*k*4B)
- **REQ-2**: Attention layer estimate: existing formula + QK norms + KV cache
- **REQ-3**: When LayerTypes is nil, use uniform estimate (backward compat)
- **REQ-4**: Total model memory = sum of per-layer estimates (not NumLayers * uniform)

#### Implementation Details

**Files to Modify/Create**:
```
pkg/scheduler/scheduler.go  - ADD: LayerTypes, ConvKernelSize fields to ModelConfig
                              - ADD: EstimateConvLayerMemory() method
                              - ADD: EstimateAttentionLayerMemoryLFM2() method
                              - MODIFY: ComputeAssignments to use per-layer estimates
                              - MODIFY: TotalModelMemory for heterogeneous layers
```

**Code Patterns to Follow**:
- `pkg/scheduler/scheduler.go:172-205` - Existing EstimateLayerMemory

**Key Implementation Steps**:
1. Add LayerTypes and ConvKernelSize to scheduler.ModelConfig
2. Implement EstimateConvLayerMemory: no KV cache, has in_proj(6144*2048*2) + conv + out_proj + FFN + conv_state
3. Implement EstimateAttentionLayerMemoryLFM2: existing + QK norms (2 * head_dim * 2 bytes)
4. Modify ComputeAssignments: for each layer, use correct estimate based on layer type
5. Modify TotalModelMemory: sum per-layer estimates instead of uniform multiplication

#### Acceptance Criteria

```gherkin
Scenario 1: Conv layer memory estimate is reasonable
  Given LFM2 config (hidden=2048, intermediate=5632, kernel=3)
  When EstimateConvLayerMemory() is called
  Then result is approximately 132MB (within 20% of manual calculation)

Scenario 2: Heterogeneous assignment works
  Given 10 conv + 6 attention layers with different memory estimates
  When ComputeAssignments is called
  Then each layer gets the correct per-type memory estimate

Scenario 3: Uniform model unchanged
  Given Llama config (nil LayerTypes)
  When ComputeAssignments is called
  Then all layers use EstimateLayerMemory() (existing behavior)
```

**Rule-Based Criteria**:
- [ ] Nil LayerTypes = existing uniform estimation (backward compat)
- [ ] Conv layer estimate includes conv state (FP32, not FP16)
- [ ] TotalModelMemory sums heterogeneous estimates correctly
- [ ] All existing scheduler tests pass

#### Validation

```bash
go test ./pkg/scheduler/... -v -run TestEstimateConvLayerMemory
go test ./pkg/scheduler/... -v -run TestHeterogeneousAssignment
make test-go
```

---

### T-013: Comprehensive Test Suite and Golden Data

**Task ID**: T-013
**Task Name**: Create test suite with golden data from HuggingFace reference implementation
**Priority**: High
**Effort**: Medium (1-2 days)

#### Context & Background

**Source PRP Document**: `docs/prps/lfm2-support.md` - Task 13

**Feature Overview**: The final validation requires golden test data generated by the HuggingFace reference implementation. A Python script generates per-layer hidden states, and Go tests compare the NeuroGrid output against these references.

**Task Purpose**:
- **As a** quality assurance process
- **I need** automated tests with ground truth from the reference implementation
- **So that** we have confidence the implementation is numerically correct

#### Dependencies
- **Prerequisite Tasks**: All other tasks (T-001 through T-012)
- **Parallel Tasks**: None (integration test)
- **Blocked By**: Full model download required

#### Technical Requirements

- **REQ-1**: BF16 GEMM: tolerance 1e-2 vs FP16 reference
- **REQ-2**: Conv1d kernel: tolerance 1e-3 vs HuggingFace reference
- **REQ-3**: Full inference: coherent output matching HuggingFace generation
- **REQ-4**: Backward compat: TinyLlama/Llama still work unchanged

#### Implementation Details

**Files to Modify/Create**:
```
tests/bf16_test.go             - CREATE: BF16 GEMM and conversion tests
tests/conv_test.go             - CREATE: Causal conv1d kernel tests
tests/conv_layer_test.go       - CREATE: Full conv layer forward tests
scripts/generate_golden_lfm2.py - CREATE: Golden data generator using HuggingFace
tests/golden/lfm2/              - CREATE: Golden test data directory
```

**Code Patterns to Follow**:
- Existing test patterns in `tests/` directory

**Key Implementation Steps**:
1. Create generate_golden_lfm2.py: load model via transformers, run inference, save per-layer hidden states as binary FP32
2. Create bf16_test.go: TestBF16GEMM (compare vs FP16), TestBF16Conversion (round-trip)
3. Create conv_test.go: TestCausalConv1dPrefill (known input/output), TestCausalConv1dDecode (matches prefill), TestConvStateReset
4. Create conv_layer_test.go: TestConvLayerForward (full block), TestConvLayerWithFFN
5. Add golden comparison test: TestGoldenLFM2 (per-layer output comparison)
6. Add end-to-end test: full inference produces coherent output
7. Add backward compat test: TinyLlama still works

#### Acceptance Criteria

```gherkin
Scenario 1: BF16 GEMM accuracy
  Given random BF16 matrices
  When GEMM result is compared with FP16 GEMM
  Then max absolute difference < 1e-2

Scenario 2: Conv1d correctness
  Given known input and weight values
  When causal conv1d processes them
  Then output matches hand-calculated reference within 1e-3

Scenario 3: Golden test passes
  Given per-layer hidden states from HuggingFace reference
  When NeuroGrid processes the same input
  Then per-layer outputs match within 1e-2 tolerance

Scenario 4: Backward compatibility
  Given an existing TinyLlama model
  When inference is run after all LFM2 changes
  Then output is identical to before the changes
```

**Rule-Based Criteria**:
- [ ] All CUDA tests pass: `make test`
- [ ] All Go tests pass: `make test-go`
- [ ] BF16 GEMM validated against FP16 reference
- [ ] Conv1d validated against HuggingFace reference
- [ ] Full inference matches golden data
- [ ] TinyLlama still works unchanged
- [ ] No linting errors: `make lint`

#### Validation

```bash
# Generate golden data
python scripts/generate_golden_lfm2.py

# Run all tests
make test                                         # CUDA tests
make test-go                                      # Go tests
cd tests && go test -tags cuda -run TestBF16 -v
cd tests && go test -tags cuda -run TestCausalConv1d -v
cd tests && go test -tags cuda -run TestConvLayer -v
cd tests && go test -tags cuda -run TestGoldenLFM2 -v

# End-to-end
make run-lfm2-thinking
curl -X POST http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"lfm2-1.2b-thinking","messages":[{"role":"user","content":"What is 2+2?"}],"temperature":0.05,"max_tokens":256}'

# Backward compat
make run-tinyllama
curl -X POST http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"tinyllama","messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'
```

---

## Implementation Recommendations

### Suggested Team Structure

| Developer | Focus | Tasks |
|-----------|-------|-------|
| Dev A (CUDA specialist) | GPU kernels | T-002, T-003, T-004, T-005 |
| Dev B (Go backend) | Integration layer | T-001, T-006, T-007, T-008, T-009, T-010 |
| Dev C (Testing/QA) | Validation | T-011, T-012, T-013 |

### Optimal Task Sequencing

**Week 1** (Foundation + Kernels):
- Day 1-2: T-001 (Config/Dtype) + T-006 (Model Config) -- Dev B
- Day 1-3: T-002 (BF16 CUDA) -- Dev A
- Day 3-5: T-003 (Conv1d) + T-004 (Conv Layer) -- Dev A
- Day 3-5: T-008 (Weight Loader) + T-009 (Hybrid Cache) -- Dev B

**Week 2** (Integration):
- Day 1-2: T-005 (BF16 Attention) -- Dev A
- Day 1-3: T-007 (CGO Bindings) -- Dev B
- Day 3-5: T-010 (Inference Engine) -- Dev A + Dev B
- Day 1-2: T-011 (Tokenizer) + T-012 (Scheduler) -- Dev C

**Week 3** (Testing + Polish):
- Day 1-5: T-013 (Tests + Golden Data) -- Dev C
- Day 1-3: Bug fixes from integration testing -- All

### Parallelization Opportunities

| Time Slot | Dev A | Dev B | Dev C |
|-----------|-------|-------|-------|
| W1 D1-2 | T-002 (BF16 CUDA) | T-001 (Config) | T-011 (Tokenizer) |
| W1 D3-5 | T-003+T-004 (Conv) | T-006+T-008 (Config+Weights) | T-012 (Scheduler) |
| W2 D1-2 | T-005 (BF16 Attn) | T-007 (CGO Bindings) | T-013 start (golden script) |
| W2 D3-5 | T-010 (Engine) | T-009 (Cache) + T-010 | T-013 (unit tests) |
| W3 | Integration testing | Integration testing | T-013 (golden tests) |

### Resource Allocation Notes

- **Ampere+ GPU required**: BF16 CUDA kernels need compute capability 8.0+. CI must have RTX 3090/A100 or newer.
- **HuggingFace model access**: T-013 requires downloading LFM2.5-1.2B-Thinking (~2.4 GB). CI must have network access or pre-cached weights.
- **Python environment**: T-013 golden data script requires `transformers`, `torch`, `safetensors` packages.

---

## Final Validation Checklist (from PRP)

- [ ] All tests pass: `make test-all`
- [ ] No linting errors: `make lint`
- [ ] Build succeeds: `make build-all`
- [ ] BF16 GEMM validated against FP16 reference (tolerance 1e-2)
- [ ] Conv1d kernel validated against HuggingFace reference (tolerance 1e-3)
- [ ] Full LFM2 inference matches golden data
- [ ] TinyLlama/Mistral/Llama still work (backward compat)
- [ ] ChatML template produces correct format
- [ ] Thinking tokens configurable (show_thinking param)
- [ ] Tied embeddings correctly share single weight
- [ ] Conv state resets between requests (no cross-request leakage)
- [ ] GPU compute capability check fails gracefully on pre-Ampere
- [ ] EOS token = 7 (`<|im_end|>`) terminates generation correctly
- [ ] Memory usage within expectations (~3GB VRAM for LFM2.5-1.2B)
