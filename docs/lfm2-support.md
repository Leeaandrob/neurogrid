name: "PRP: LFM2.5-1.2B-Thinking Model Support with BF16 Native Compute"
description: |

## Purpose

Implement full support for the Liquid AI LFM2.5-1.2B-Thinking model in NeuroGrid Engine, including the hybrid conv+attention architecture, new CUDA kernels for depthwise causal convolution, BF16 native compute, and hybrid cache management. This is the first non-Llama architecture supported by the engine.

## Core Principles

1. Include ALL necessary documentation, examples, and caveats
2. Provide executable tests/lints the AI can run and fix
3. Use keywords and patterns from the codebase
4. Start simple, validate, then enhance
5. Maintain backward compatibility with existing Llama/Mistral/TinyLlama models
6. Follow all rules in CLAUDE.md

---

## Discovery Summary

### Initial Task Analysis

User requested full implementation of LFM2.5-1.2B-Thinking model support in NeuroGrid Engine. The engine is currently hardcoded for Llama-family transformer architectures (uniform attention+FFN layers). LFM2.5 uses a hybrid architecture with 10 convolution blocks and 6 GQA attention blocks, requiring significant architectural changes across config, CUDA kernels, weight loading, cache management, and inference pipeline.

### User Clarifications Received

- **Question**: What does "BF16 native support" mean?
- **Answer**: Full BF16 compute pipeline — BF16 GEMM via cuBLAS, BF16 kernels, BF16 storage. Not just BF16→FP16 conversion at load.
- **Impact**: Requires new BF16 GEMM kernels, BF16 element-wise kernels (RMSNorm, SiLU, add, mul), and BF16 attention. Minimum GPU: Ampere (compute capability 8.0+).

- **Question**: Conv block gating computation?
- **Answer**: `in_proj(2048→6144)` → chunk into B, C, x (each 2048) → `Bx = B * x` → `conv1d(Bx, k=3)` → `y = C * conv_out` → `out_proj(y)`. No activation between steps.
- **Impact**: Straightforward CUDA kernel — element-wise gates sandwich a depthwise conv1d.

- **Question**: Do both layer types share the same FFN?
- **Answer**: Yes. Both conv and attention layers have identical SwiGLU FFN (w1/w2/w3) after their operator block.
- **Impact**: FFN CUDA code can be shared between layer types. Only the operator block differs.

- **Question**: Distributed support?
- **Answer**: Single-node first. Distributed as follow-up.
- **Impact**: Simplifies initial scope — no conv cache serialization over P2P needed.

### Missing Requirements Identified

1. `conv_L_cache=3` means conv state buffer holds 3 past values. The depthwise conv kernel width is effectively 3 (not 4 — the research initially suggested 4 but the HuggingFace implementation uses `kernel_size=conv_L_cache=3` with `padding=conv_L_cache-1=2` for causal padding).
2. QK LayerNorm: RMSNorm applied per-head to Q and K AFTER projection, BEFORE RoPE. Weight shape: `[head_dim]` = `[64]`.
3. `eos_token_id=7` (`<|im_end|>`), NOT 2 (`<|endoftext|>`). Critical for stopping generation.
4. Tied embeddings: `lm_head.weight == model.embed_tokens.weight`. Must not allocate separate storage.
5. Conv weight stored as `[2048, 1, 3]` in SafeTensors but used as `[2048, 3]` — must reshape on load.
6. Thinking tokens `<think>`/`</think>` (IDs 64400/64401) should be configurable: strip from output or include.

## Goal

Add LFM2.5-1.2B-Thinking as the second model architecture supported by NeuroGrid Engine, alongside the existing Llama family. The engine must:

1. Load and parse the LFM2 config.json with `layer_types[]` array
2. Load SafeTensors weights with LFM2 naming convention (conv + attention + FFN)
3. Execute forward pass through hybrid conv+attention layers using BF16 CUDA kernels
4. Manage a hybrid cache: fixed conv state buffers + growing KV caches
5. Tokenize with BPE (65K vocab) and apply ChatML template with thinking token support
6. Produce correct inference output validated against HuggingFace reference implementation

## Why

- **First non-transformer model**: Opens NeuroGrid to hybrid architectures (Mamba, RWKV, etc.)
- **Edge AI reasoning**: 1.2B model with reasoning capability, fits in <3GB VRAM
- **BF16 foundation**: BF16 compute pipeline benefits all future models on Ampere+ GPUs
- **Memory efficiency**: 62.5% less KV cache vs pure transformer (only 6/16 layers need it)
- **Architecture generalization**: Refactoring from Llama-only to multi-architecture makes the engine production-grade

## What

### User-Visible Behavior

- `make download-lfm2-thinking` downloads the model from HuggingFace
- `make run-lfm2-thinking` starts inference with LFM2.5-1.2B-Thinking
- `POST /v1/chat/completions` with `"model": "lfm2-1.2b-thinking"` produces reasoning output
- Thinking tokens `<think>...</think>` appear in streaming output (configurable via `show_thinking` param)
- `GET /v1/models` lists the new model

### Success Criteria

- [ ] LFM2.5-1.2B-Thinking config.json parses correctly with all layer_types
- [ ] All 16 layers load: 10 conv layers + 6 attention layers + shared FFN
- [ ] BF16 GEMM produces correct results (validated against cuBLAS FP16 reference within 1e-2 tolerance)
- [ ] Depthwise causal conv1d kernel matches HuggingFace reference within 1e-3 tolerance
- [ ] QK LayerNorm applied correctly in attention layers
- [ ] Hybrid cache: conv state [batch, 2048, 3] fixed + KV cache growing for attn layers
- [ ] ChatML template produces correct prompt format
- [ ] Thinking tokens parsed and optionally stripped
- [ ] Tied embeddings use single weight allocation
- [ ] Backward compatibility: Llama/Mistral/TinyLlama models still work unchanged
- [ ] All existing tests pass (`make test`, `make test-e2e`)
- [ ] New CUDA kernel tests pass for conv1d, BF16 GEMM, QK LayerNorm
- [ ] Golden test: full inference output matches HuggingFace reference for a test prompt

## All Needed Context

### Research Phase Summary

- **Codebase patterns found**: BF16→FP16 conversion already in loader.go, ChatTemplate interface, LayerExecutor interface, CGO binding pattern (gpu.go/gpu_stub.go), Makefile auto-discovers .cu files
- **External research needed**: Yes — BF16 cuBLAS API, causal conv1d CUDA kernels, LFM2 config schema, ChatML template format
- **Knowledge gaps identified**: BF16 GEMM via cublasGemmEx, depthwise causal conv1d implementation, QK LayerNorm pattern, conv state cache management

### Documentation & References

```yaml
# MUST READ - Include these in your context window
- url: https://huggingface.co/LiquidAI/LFM2.5-1.2B-Thinking
  why: "Model card with exact config.json, tokenizer_config.json, chat_template"
  critical: "layer_types array, conv_L_cache=3, eos_token_id=7, rope_theta=1000000"

- url: https://github.com/huggingface/transformers/blob/main/src/transformers/models/lfm2/modeling_lfm2.py
  why: "Complete reference implementation of all block types"
  critical: "Lfm2ShortConv forward pass, Lfm2Attention with QK LayerNorm, Lfm2HybridConvCache"

- url: https://github.com/Dao-AILab/causal-conv1d
  why: "Reference CUDA kernels for depthwise causal conv1d"
  critical: "Separate prefill and decode kernels. Weight shape [dim, width] not [dim, 1, width]"

- url: https://docs.nvidia.com/cuda/cublas/
  why: "BF16 GEMM via cublasGemmEx"
  critical: "Use CUDA_R_16BF + CUBLAS_COMPUTE_32F. No CUBLAS_COMPUTE_16BF exists."

- url: https://arxiv.org/abs/2511.23404
  why: "LFM2 technical report with GSC block formulation"
  critical: "(B,C,h_tilde)=Linear(h), y=B*h_tilde, z=Conv_k(y), o=Linear_out(C*z)"

- file: gpu/engine/layer.cu
  why: "Current transformer forward pass — pattern to extend"

- file: gpu/bindings/gpu.go
  why: "CGO binding pattern — must add parallel functions"

- file: pkg/types/config.go
  why: "LlamaConfig struct — must extend for LFM2 fields"

- file: pkg/model/loader.go
  why: "Weight loading — must add conv layer tensor paths"

- file: pkg/inference/engine.go
  why: "Forward loop — must branch by layer type"

- file: pkg/inference/cuda_executor.go
  why: "CUDA layer executor — must add conv layer support"

- file: pkg/model/model_config.go
  why: "Config parsing, model detection — must add LFM2 entries"
```

### Current Codebase Tree (relevant files)

```
neurogrid-engine/
├── gpu/
│   ├── cuda/
│   │   ├── attention.cu/.h      # GQA attention + KV cache
│   │   ├── matmul.cu/.h         # cuBLAS FP16/INT8 GEMM
│   │   ├── kernels.cu/.h        # RMSNorm, SiLU, RoPE, add, mul
│   │   ├── quantize.cu/.h       # FP16↔INT8 quantization
│   │   └── memory.cu/.h         # GPU memory management
│   ├── engine/
│   │   └── layer.cu/.h          # Transformer layer forward (HARDCODED attn+FFN)
│   └── bindings/
│       ├── gpu.go               # CGO bridge (//go:build cuda)
│       ├── gpu.h                # C header declarations
│       ├── gpu_stub.go          # No-GPU fallback (//go:build !cuda)
│       └── pinned_memory.go/.h  # Pinned memory for DMA
├── pkg/
│   ├── types/
│   │   └── config.go            # LlamaConfig struct (LLAMA-ONLY)
│   ├── model/
│   │   ├── loader.go            # SafeTensors weight loading (LLAMA PATHS)
│   │   ├── weights.go           # Weight structs (LLAMA FIELDS)
│   │   ├── model_config.go      # Config parsing, model detection
│   │   ├── tokenizer.go         # BPE tokenizer
│   │   ├── chat_template.go     # Chat template interface
│   │   └── sentencepiece.go     # SentencePiece tokenizer
│   ├── inference/
│   │   ├── engine.go            # Core inference loop (UNIFORM LAYERS)
│   │   ├── engine_gpu.go        # GPU initialization & weight loading
│   │   ├── engine_nogpu.go      # No-GPU fallback
│   │   ├── cuda_executor.go     # CUDA layer executor (ATTENTION ONLY)
│   │   ├── kvcache.go           # KV cache (UNIFORM PER LAYER)
│   │   ├── sampler.go           # Token sampling
│   │   └── config_transfer.go   # P2P config serialization
│   └── scheduler/
│       └── scheduler.go         # VRAM-aware scheduling (UNIFORM LAYERS)
├── configs/
│   └── models.yaml              # Model aliases
└── Makefile                     # Build system (auto-discovers .cu files)
```

### Desired Codebase Tree (new/modified files)

```
neurogrid-engine/
├── gpu/
│   ├── cuda/
│   │   ├── attention.cu/.h      # MODIFY: add QK LayerNorm support
│   │   ├── matmul.cu/.h         # MODIFY: add BF16 GEMM (cublasGemmEx with CUDA_R_16BF)
│   │   ├── kernels.cu/.h        # MODIFY: add BF16 variants of RMSNorm, SiLU, add, mul
│   │   ├── conv.cu/.h           # NEW: depthwise causal conv1d (prefill + decode kernels)
│   │   ├── bf16_utils.cu/.h     # NEW: BF16↔FP32 conversion, BF16 element-wise ops
│   │   ├── quantize.cu/.h       # (unchanged)
│   │   └── memory.cu/.h         # (unchanged)
│   ├── engine/
│   │   ├── layer.cu/.h          # MODIFY: add cuda_attn_layer_forward_bf16() with QK LayerNorm
│   │   └── conv_layer.cu/.h     # NEW: cuda_conv_layer_forward_bf16() — LIV conv block
│   └── bindings/
│       ├── gpu.go               # MODIFY: add conv layer, BF16, QK LayerNorm bindings
│       ├── gpu.h                # MODIFY: add new C declarations
│       └── gpu_stub.go          # MODIFY: add stubs for all new functions
├── pkg/
│   ├── types/
│   │   ├── config.go            # MODIFY: add LayerTypes, ConvKernelSize, TieEmbeddings, Dtype
│   │   └── dtype.go             # NEW: BFloat16 type, conversion functions
│   ├── model/
│   │   ├── loader.go            # MODIFY: add conv weight loading, BF16 support, tied embeddings
│   │   ├── weights.go           # MODIFY: add ConvLayerWeights struct
│   │   ├── model_config.go      # MODIFY: add LFM2 detection, config fields, ChatML template
│   │   ├── chat_template.go     # MODIFY: add ChatMLTemplate with thinking token support
│   │   └── tokenizer.go         # MODIFY: handle 65K vocab BPE with special tokens
│   ├── inference/
│   │   ├── engine.go            # MODIFY: branch forwardAllLayers by layer type
│   │   ├── engine_gpu.go        # MODIFY: branch weight loading by layer type
│   │   ├── cuda_executor.go     # MODIFY: add ForwardConv, LoadConvLayer, conv weights map
│   │   ├── kvcache.go           # MODIFY: add ConvStateCache, HybridCacheManager
│   │   └── config_transfer.go   # MODIFY: add LFM2 config fields
│   └── scheduler/
│       └── scheduler.go         # MODIFY: per-layer memory estimation (conv vs attn)
├── configs/
│   └── models.yaml              # MODIFY: add LFM2 model alias
├── scripts/
│   └── generate_golden_lfm2.py  # NEW: golden data generator using HuggingFace
├── tests/
│   ├── conv_test.go             # NEW: depthwise conv1d kernel tests
│   ├── bf16_test.go             # NEW: BF16 GEMM and conversion tests
│   ├── conv_layer_test.go       # NEW: full conv layer forward tests
│   └── golden/lfm2/             # NEW: golden test data for LFM2
└── Makefile                     # (auto-discovers new .cu files, no change needed)
```

### Known Gotchas of our Codebase & Library Quirks

```go
// CRITICAL: Config propagation chain — new fields must be added to ALL 5 structs:
// 1. pkg/types/config.go: LlamaConfig
// 2. pkg/model/model_config.go: ModelConfig (with JSON tags)
// 3. pkg/scheduler/scheduler.go: scheduler.ModelConfig
// 4. pkg/inference/config_transfer.go: TransferableConfig (with JSON tags)
// 5. Conversion functions: ToLlamaConfig(), FromLlamaConfig()

// CRITICAL: Every new GPU function needs THREE files updated:
// 1. gpu/bindings/gpu.go (//go:build cuda) — real implementation
// 2. gpu/bindings/gpu_stub.go (//go:build !cuda) — stub returning error
// 3. gpu/bindings/gpu.h — C header declaration

// CRITICAL: cuBLAS BF16 — There is NO CUBLAS_COMPUTE_16BF enum.
// Must use CUBLAS_COMPUTE_32F with CUDA_R_16BF inputs.
// BF16 requires compute capability >= 8.0 (Ampere: RTX 3090/A100+)

// CRITICAL: Conv weight in SafeTensors is [2048, 1, 3] but kernel expects [2048, 3]
// Must reshape during weight loading (squeeze dim 1)

// CRITICAL: eos_token_id=7 (<|im_end|>), NOT 2 (<|endoftext|>)
// Stopping on wrong token = infinite generation

// GOTCHA: Zero-value defaults must preserve Llama behavior:
// LayerTypes=nil means "all attention layers" (backward compat)
// Dtype="" means "fp16" (existing behavior)
// TieEmbeddings=false means separate lm_head (existing behavior)

// GOTCHA: Existing INT8 quantization in layer.cu uses FP16 norms.
// For BF16 path, norms should stay BF16. The quantization path
// (FP16→INT8 or BF16→INT8) may need adjustment.

// GOTCHA: Conv state must be zeroed at start of each new sequence.
// Failure to reset between requests causes cross-request state leakage.

// GOTCHA: Padding tokens must be zeroed before conv blocks.
// Without apply_mask_to_padding_states, padding corrupts conv state.
```

## Implementation Blueprint

### Data Models and Structure

```go
// pkg/types/config.go — Extended LlamaConfig
type LlamaConfig struct {
    // Existing fields (unchanged)
    HiddenSize       int
    IntermediateSize int
    NumLayers        int
    NumHeads         int
    NumKVHeads       int
    HeadDim          int
    VocabSize        int
    MaxSeqLen        int
    RMSNormEps       float32
    RopeTheta        float32

    // NEW: LFM2 support
    LayerTypes      []string // "conv" or "full_attention" per layer. nil = all attention (backward compat)
    ConvKernelSize  int      // conv_L_cache from config.json (default 0 = no conv)
    ConvDim         int      // conv_dim (typically == HiddenSize)
    ConvBias        bool     // conv_bias (default false)
    TieEmbeddings   bool     // tie_embedding (lm_head == embed_tokens)
    Dtype           string   // "fp16", "bf16", "int8" (default "fp16")
    ModelType       string   // "llama", "lfm2" etc.
}

func (c *LlamaConfig) IsConvLayer(layerID int) bool {
    if c.LayerTypes == nil { return false }
    if layerID >= len(c.LayerTypes) { return false }
    return c.LayerTypes[layerID] == "conv"
}

func (c *LlamaConfig) IsAttentionLayer(layerID int) bool {
    return !c.IsConvLayer(layerID)
}

func (c *LlamaConfig) NumConvLayers() int {
    count := 0
    for _, lt := range c.LayerTypes {
        if lt == "conv" { count++ }
    }
    return count
}

func (c *LlamaConfig) NumAttentionLayers() int {
    return c.NumLayers - c.NumConvLayers()
}
```

```go
// pkg/types/dtype.go — NEW: BFloat16 type
type BFloat16 uint16

func Float32ToBFloat16(f float32) BFloat16 {
    bits := math.Float32bits(f)
    bits += (bits >> 16) & 1 // rounding
    bits += 0x7FFF
    return BFloat16(bits >> 16)
}

func (b BFloat16) Float32() float32 {
    return math.Float32frombits(uint32(b) << 16)
}
```

```go
// pkg/model/weights.go — Extended weight structs
type ConvLayerWeights struct {
    InProjWeight []byte // [3*hidden, hidden] BF16/FP16
    ConvWeight   []byte // [hidden, kernel_size] BF16/FP16 (reshaped from [hidden, 1, k])
    OutProjWeight []byte // [hidden, hidden] BF16/FP16
    OperatorNorm []byte // [hidden] RMSNorm weight
    FFNNorm      []byte // [hidden] RMSNorm weight
    GateWeight   []byte // [intermediate, hidden] SwiGLU gate
    UpWeight     []byte // [intermediate, hidden] SwiGLU up
    DownWeight   []byte // [hidden, intermediate] SwiGLU down
}

type AttentionLayerWeights struct {
    // Existing fields plus:
    QLayerNorm []byte // [head_dim] per-head RMSNorm for Q (NEW for LFM2)
    KLayerNorm []byte // [head_dim] per-head RMSNorm for K (NEW for LFM2)
}
```

```c
// gpu/cuda/conv.h — NEW: Causal Conv1d declarations
#ifndef CONV_H
#define CONV_H

#ifdef __cplusplus
extern "C" {
#endif

// Prefill: process full sequence through depthwise causal conv1d
int cuda_causal_conv1d_fwd_bf16(
    const void* x,       // [batch, dim, seqlen] BF16
    const void* weight,  // [dim, width] BF16
    void* out,           // [batch, dim, seqlen] BF16
    void* conv_state,    // [batch, dim, width] FP32 (updated in-place)
    int batch, int dim, int seqlen, int width
);

// Decode: single token update with state management
int cuda_causal_conv1d_update_bf16(
    const void* x,       // [batch, dim] BF16 single token
    void* out,           // [batch, dim] BF16
    void* conv_state,    // [batch, dim, width] FP32 (updated in-place)
    const void* weight,  // [dim, width] BF16
    int batch, int dim, int width
);

// Conv state management
void* cuda_conv_state_create(int batch, int dim, int width);
int cuda_conv_state_reset(void* state, int batch, int dim, int width);
void cuda_conv_state_free(void* state);

#ifdef __cplusplus
}
#endif
#endif // CONV_H
```

```c
// gpu/engine/conv_layer.h — NEW: LIV Conv layer forward
#ifndef CONV_LAYER_H
#define CONV_LAYER_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    void* in_proj_weight;   // [3*hidden, hidden] BF16
    void* conv_weight;      // [hidden, kernel_size] BF16
    void* out_proj_weight;  // [hidden, hidden] BF16
    void* operator_norm;    // [hidden] BF16
    void* ffn_norm;         // [hidden] BF16
    void* gate_weight;      // [intermediate, hidden] BF16
    void* up_weight;        // [intermediate, hidden] BF16
    void* down_weight;      // [hidden, intermediate] BF16
    int hidden_size;
    int intermediate_size;
    int conv_kernel_size;
    float norm_eps;
} ConvLayerWeights;

// Create conv layer weights from host BF16 data
ConvLayerWeights* cuda_create_conv_layer_weights_bf16(
    const void* in_proj, const void* conv_w, const void* out_proj,
    const void* op_norm, const void* ffn_norm,
    const void* gate, const void* up, const void* down,
    int hidden, int intermediate, int kernel_size, float norm_eps
);

// Forward pass: RMSNorm → LIV Conv → Residual → RMSNorm → SwiGLU FFN → Residual
int cuda_conv_layer_forward_bf16(
    void* output,              // [batch, seq, hidden] BF16
    const void* input,         // [batch, seq, hidden] BF16
    const ConvLayerWeights* w,
    void* conv_state,          // [batch, hidden, kernel_size] FP32
    int batch, int seq_len, int position
);

void cuda_free_conv_layer_weights(ConvLayerWeights* w);

#ifdef __cplusplus
}
#endif
#endif // CONV_LAYER_H
```

### List of Tasks

```yaml
Task 1: BF16 Data Type Foundation
MODIFY pkg/types/config.go:
  - ADD fields: LayerTypes, ConvKernelSize, ConvDim, ConvBias, TieEmbeddings, Dtype, ModelType
  - ADD methods: IsConvLayer(), IsAttentionLayer(), NumConvLayers(), NumAttentionLayers()
  - ADD preset: LFM2_1_2BThinkingConfig()
  - PRESERVE all existing presets and methods (backward compatibility)
CREATE pkg/types/dtype.go:
  - BFloat16 type as uint16
  - Float32ToBFloat16(), BFloat16.Float32() conversion functions
  - ReadBFloat16Slice() for SafeTensors loading

Task 2: BF16 CUDA Kernels
CREATE gpu/cuda/bf16_utils.cu/.h:
  - cuda_bf16_rmsnorm(): RMSNorm with BF16 in/out, FP32 accumulation
  - cuda_bf16_silu(): SiLU activation for BF16
  - cuda_bf16_add(): Element-wise BF16 addition
  - cuda_bf16_mul(): Element-wise BF16 multiplication
  - cuda_bf16_to_fp32(): BF16→FP32 conversion kernel
  - cuda_fp32_to_bf16(): FP32→BF16 conversion kernel
MODIFY gpu/cuda/matmul.cu/.h:
  - ADD cuda_gemm_bf16(): cublasGemmEx with CUDA_R_16BF inputs, CUBLAS_COMPUTE_32F
  - ADD cuda_gemm_bf16_bf16out(): BF16 in, BF16 out variant

Task 3: Depthwise Causal Conv1d Kernel
CREATE gpu/cuda/conv.cu/.h:
  - cuda_causal_conv1d_fwd_bf16(): Prefill kernel (full sequence)
    Grid: (batch, dim), Block: 128 threads
    Causal padding: pad left by kernel_size-1, truncate to seq_len
    FP32 accumulation, BF16 in/out
  - cuda_causal_conv1d_update_bf16(): Decode kernel (single token)
    Grid: (batch, ceil(dim/64)), Block: 64 threads
    Circular buffer state update: shift left, insert new value
    3 multiply-adds per channel
  - cuda_conv_state_create/reset/free(): Conv state management

Task 4: LIV Conv Layer Forward
CREATE gpu/engine/conv_layer.cu/.h:
  - ConvLayerWeights struct: in_proj, conv, out_proj, norms, FFN weights
  - cuda_create_conv_layer_weights_bf16(): Upload BF16 weights to GPU
  - cuda_conv_layer_forward_bf16():
    1. RMSNorm(input, operator_norm)
    2. in_proj GEMM: [batch, seq, hidden] → [batch, seq, 3*hidden] BF16
    3. Transpose to [batch, 3*hidden, seq], chunk into B, C, x each [batch, hidden, seq]
    4. Element-wise multiply: Bx = B * x
    5. Causal conv1d(Bx, kernel=3) → conv_out
    6. Element-wise multiply: y = C * conv_out
    7. Transpose back to [batch, seq, hidden]
    8. out_proj GEMM: [batch, seq, hidden] → [batch, seq, hidden] BF16
    9. Residual add
    10. RMSNorm(residual, ffn_norm)
    11. SwiGLU FFN: down(silu(gate(x)) * up(x))
    12. Final residual add
  - cuda_free_conv_layer_weights()

Task 5: Attention Layer with QK LayerNorm (BF16)
MODIFY gpu/engine/layer.cu/.h:
  - ADD to LayerWeights struct: q_layernorm, k_layernorm (BF16 weights, [head_dim])
  - ADD cuda_attn_layer_forward_bf16(): BF16 variant of cuda_layer_forward
    Same as existing but:
    - BF16 GEMM for projections
    - After Q projection: apply per-head RMSNorm to Q
    - After K projection: apply per-head RMSNorm to K
    - Then RoPE as normal
    - BF16 attention
    - BF16 FFN
MODIFY gpu/cuda/attention.cu/.h:
  - ADD cuda_attention_with_kvcache_bf16(): BF16 attention with KV cache
  - ADD cuda_basic_attention_gqa_bf16(): BF16 full attention

Task 6: Go↔CUDA Bindings
MODIFY gpu/bindings/gpu.go:
  - ADD ConvLayerWeights Go struct wrapping C pointer
  - ADD CreateConvLayerWeightsBF16(): bridge to cuda_create_conv_layer_weights_bf16
  - ADD ConvLayerForwardBF16(): bridge to cuda_conv_layer_forward_bf16
  - ADD FreeConvLayerWeights(): bridge to cuda_free_conv_layer_weights
  - ADD GEMM_BF16(): bridge to cuda_gemm_bf16
  - ADD ConvStateCreate/Reset/Free(): conv state management
  - ADD AttnLayerForwardBF16(): bridge to cuda_attn_layer_forward_bf16
  - ADD CheckBF16Support(): query compute capability >= 8.0
MODIFY gpu/bindings/gpu_stub.go:
  - ADD stubs returning ErrNotImplemented for ALL new functions
MODIFY gpu/bindings/gpu.h:
  - ADD declarations for all new C functions

Task 7: Model Config & Detection
MODIFY pkg/model/model_config.go:
  - ADD JSON fields to ModelConfig: layer_types, conv_L_cache, conv_dim, conv_bias, tie_embedding, dtype, model_type
  - ADD "lfm2" to KnownModelFamilies map
  - ADD LFM2 pattern to DetectModelFamily() (match "lfm" in model_type or architectures)
  - ADD LFM2 to DetectRoPEStyle() → interleaved, theta=1000000
  - ADD LFM2 to DetectChatTemplate() → ChatML
MODIFY pkg/model/model_config.go conversion functions:
  - Map new config fields from ModelConfig → LlamaConfig
MODIFY cmd/neurogrid/main.go:
  - ADD "lfm2", "lfm2-1.2b-thinking" cases in getModelConfig()
MODIFY configs/models.yaml:
  - ADD lfm2-1.2b-thinking alias: LiquidAI/LFM2.5-1.2B-Thinking

Task 8: Weight Loading
MODIFY pkg/model/weights.go:
  - ADD ConvLayerWeights struct with all conv block fields
  - ADD AttentionLayerWeights extended with QLayerNorm, KLayerNorm
MODIFY pkg/model/loader.go:
  - ADD LoadConvLayerWeights(layerID) function:
    Tensor paths: model.layers.{i}.conv.in_proj.weight, .conv.conv.weight, .conv.out_proj.weight
    + model.layers.{i}.operator_norm.weight, .ffn_norm.weight
    + model.layers.{i}.feed_forward.w1/w2/w3.weight
    Reshape conv.weight from [hidden, 1, kernel] to [hidden, kernel] (squeeze dim 1)
  - ADD LoadAttentionLayerWeightsLFM2(layerID) function:
    Tensor paths: model.layers.{i}.self_attn.{q,k,v}_proj.weight, .out_proj.weight
    + model.layers.{i}.self_attn.q_layernorm.weight, .k_layernorm.weight
    + model.layers.{i}.operator_norm.weight, .ffn_norm.weight
    + model.layers.{i}.feed_forward.w1/w2/w3.weight
  - MODIFY LoadEmbeddings: handle tied embeddings (TieEmbeddings=true → lm_head = embed_tokens)
  - ADD BF16 loading path: keep as BF16 bytes when Dtype="bf16" (no conversion to FP16)

Task 9: Hybrid Cache Manager
MODIFY pkg/inference/kvcache.go:
  - ADD ConvStateCache struct:
    layerID int, hiddenSize int, kernelSize int
    stateBuffer []byte — [batch, hidden, kernel_size] FP32 on GPU
    gpuPtr unsafe.Pointer
  - ADD ConvStateCache methods: Create(), Reset(), Free()
  - ADD HybridCacheManager struct:
    kvCaches map[int]*DistributedKVCache (only for attention layers)
    convCaches map[int]*ConvStateCache (only for conv layers)
    config *LlamaConfig
  - ADD HybridCacheManager methods: InitForModel(), ResetAll(), Close()
  - MODIFY or extend KVCacheManager to support hybrid mode

Task 10: Inference Engine Integration
MODIFY pkg/inference/engine.go:
  - MODIFY forwardAllLayersHidden():
    ```
    for layerID := 0; layerID < e.config.NumLayers; layerID++ {
        if e.config.IsConvLayer(layerID) {
            output, err = e.forwardConvLayer(ctx, layerID, current, position)
            // Conv layers: no K,V output, no KV cache update
        } else {
            output, k, v, err = e.forwardLayer(ctx, layerID, current, position)
            // Attention layers: standard KV cache update
        }
    }
    ```
  - ADD forwardConvLayer() method
MODIFY pkg/inference/cuda_executor.go:
  - ADD convLayerWeights map[int]*bindings.ConvLayerWeights
  - ADD convStates map[int]unsafe.Pointer (GPU conv state pointers)
  - ADD LoadConvLayer(layerID, weights) method
  - ADD ForwardConv(ctx, layerID, hidden, position) method
  - MODIFY Close(): free conv weights and states
  - MODIFY ResetCache(): reset conv states
MODIFY pkg/inference/engine_gpu.go:
  - MODIFY layer loading loop to branch by layer type:
    conv layers → LoadConvLayerWeights → LoadConvLayer
    attn layers → LoadAttentionLayerWeightsLFM2 → LoadLayer (extended)

Task 11: Tokenizer & Chat Template
MODIFY pkg/model/chat_template.go:
  - ADD ChatMLTemplate struct implementing ChatTemplate interface
  - Format: <|startoftext|><|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n
  - ADD ThinkingTokenHandler: strip <think>...</think> from non-final messages
  - ADD show_thinking parameter support
MODIFY pkg/model/model_config.go:
  - ADD ChatML template detection in DetectChatTemplate()
MODIFY pkg/model/tokenizer.go:
  - VERIFY BPE tokenizer handles 65K vocab from tokenizer.json
  - ADD special token mappings: eos=7, bos=1, pad=0
  - ADD tool call tokens: 10, 11, 12, 13
  - ADD thinking tokens: 64400, 64401

Task 12: Scheduler Updates
MODIFY pkg/scheduler/scheduler.go:
  - ADD new fields to scheduler.ModelConfig matching LlamaConfig additions
  - ADD EstimateConvLayerMemory(): different from attention (no KV cache, has conv state)
    Conv layer: in_proj(3*h*h*2B) + conv(h*k*2B) + out_proj(h*h*2B) + FFN + conv_state(h*k*4B)
    ≈ 132 MB per conv layer
  - ADD EstimateAttentionLayerMemory(): includes QK norms + KV cache
    ≈ 128 MB per attn layer + KV cache growth
  - MODIFY ComputeAssignments: use per-layer memory estimates

Task 13: Tests
CREATE tests/bf16_test.go:
  - TestBF16GEMM: compare BF16 GEMM against FP16 GEMM (tolerance 1e-2)
  - TestBF16Conversion: round-trip accuracy
CREATE tests/conv_test.go:
  - TestCausalConv1dPrefill: known input → expected output
  - TestCausalConv1dDecode: sequential single-token updates match prefill output
  - TestConvStateReset: verify state zeroing
CREATE tests/conv_layer_test.go:
  - TestConvLayerForward: full LIV conv block forward
  - TestConvLayerWithFFN: conv block + SwiGLU FFN
CREATE scripts/generate_golden_lfm2.py:
  - Load LFM2.5-1.2B-Thinking via HuggingFace transformers
  - Run inference on test prompt
  - Save per-layer hidden states as binary FP32 for golden comparison
ADD to tests/golden/lfm2/:
  - Golden input/output tensors for each layer type
```

### Per-Task Pseudocode

```cuda
// Task 3: Depthwise Causal Conv1d — Decode Kernel (critical path)
// This is the hottest kernel during autoregressive generation
__global__ void causal_conv1d_update_bf16(
    const __nv_bfloat16* x,     // [batch, dim] single token
    __nv_bfloat16* out,          // [batch, dim]
    float* conv_state,           // [batch, dim, width] circular buffer
    const __nv_bfloat16* weight, // [dim, width]
    int batch, int dim, int width)
{
    int b = blockIdx.x;
    int d = blockIdx.y * blockDim.x + threadIdx.x;
    if (d >= dim) return;

    // Load weight for this channel (width=3, fits in registers)
    float w[3]; // PATTERN: unroll for width=3
    for (int i = 0; i < width; i++)
        w[i] = __bfloat162float(weight[d * width + i]);

    // Read new input
    float x_val = __bfloat162float(x[b * dim + d]);

    // Shift state left, insert new value (FIFO)
    int base = b * dim * width + d * width;
    for (int i = 0; i < width - 1; i++)
        conv_state[base + i] = conv_state[base + i + 1];
    conv_state[base + width - 1] = x_val;

    // 3-tap FIR: out = sum(w[i] * state[i])
    float acc = 0.0f;
    for (int i = 0; i < width; i++)
        acc += w[i] * conv_state[base + i];

    out[b * dim + d] = __float2bfloat16(acc);
}
```

```cuda
// Task 4: LIV Conv Layer Forward (pseudocode)
int cuda_conv_layer_forward_bf16(output, input, weights, conv_state, batch, seq, pos) {
    // 1. RMSNorm
    cuda_bf16_rmsnorm(normed, input, weights->operator_norm, hidden, norm_eps);

    // 2. in_proj GEMM: [batch*seq, hidden] → [batch*seq, 3*hidden]
    cuda_gemm_bf16(batch*seq, 3*hidden, hidden, normed, weights->in_proj, projected);

    // 3. Chunk + transpose: [batch, seq, 3*hidden] → B,C,x each [batch, hidden, seq]
    // (done via pointer arithmetic, no copy needed)

    // 4. Bx = B * x (element-wise)
    cuda_bf16_mul(Bx, B, x, batch * hidden * seq);

    // 5. Causal conv1d
    if (seq == 1) {
        cuda_causal_conv1d_update_bf16(Bx, conv_out, conv_state, weights->conv, batch, hidden, kernel);
    } else {
        cuda_causal_conv1d_fwd_bf16(Bx, weights->conv, conv_out, conv_state, batch, hidden, seq, kernel);
    }

    // 6. y = C * conv_out
    cuda_bf16_mul(y, C, conv_out, batch * hidden * seq);

    // 7. Transpose back + out_proj GEMM
    cuda_gemm_bf16(batch*seq, hidden, hidden, y_transposed, weights->out_proj, block_out);

    // 8. Residual
    cuda_bf16_add(residual, input, block_out, batch * seq * hidden);

    // 9. FFN (identical to attention layer FFN — shared code)
    cuda_bf16_rmsnorm(ffn_normed, residual, weights->ffn_norm, hidden, norm_eps);
    cuda_gemm_bf16(batch*seq, intermediate, hidden, ffn_normed, weights->gate, gate_out);
    cuda_gemm_bf16(batch*seq, intermediate, hidden, ffn_normed, weights->up, up_out);
    cuda_bf16_silu(gate_out, batch*seq*intermediate);
    cuda_bf16_mul(ffn_mid, gate_out, up_out, batch*seq*intermediate);
    cuda_gemm_bf16(batch*seq, hidden, intermediate, ffn_mid, weights->down, ffn_out);
    cuda_bf16_add(output, residual, ffn_out, batch*seq*hidden);

    return 0;
}
```

### Integration Points

```yaml
CONFIG:
  - MODIFY: configs/models.yaml
  - ADD: lfm2-1.2b-thinking alias pointing to LiquidAI/LFM2.5-1.2B-Thinking
  - ADD: download target in Makefile (make download-lfm2-thinking)

API:
  - No route changes needed — existing /v1/chat/completions handles any model
  - ADD: show_thinking parameter to request body (api/types.go)
  - MODIFY: api/server.go to strip thinking tokens when show_thinking=false
  - MODIFY: api/stream.go to handle thinking token output in SSE

INFERENCE ENGINE:
  - MODIFY: pkg/inference/engine.go — forwardAllLayersHidden() dispatch
  - MODIFY: pkg/inference/cuda_executor.go — dual executor paths
  - MODIFY: pkg/inference/engine_gpu.go — dual weight loading paths

BUILD:
  - No Makefile changes needed — .cu files auto-discovered by wildcard
  - VERIFY: nvcc flags support BF16 (-arch=native covers Ampere+)
  - ADD: make run-lfm2-thinking target
```

## Validation Loop

### Level 1: Syntax & Style

```bash
# Run these FIRST - fix any errors before proceeding
make fmt                              # Go fmt + clang-format for CUDA
go vet -tags cuda ./...               # Static analysis
make lint                             # golangci-lint

# Expected: No errors. If errors, READ the error and fix.
```

### Level 2: Compilation

```bash
# Build CUDA library first
make cuda                             # Compiles all .cu into libgpu_engine.so

# Build Go binaries
make build-coordinator                # Build neurogrid binary
make build-worker                     # Build worker binary

# Expected: Clean compilation. BF16 requires -arch=native on Ampere+
```

### Level 3: Unit Tests

```bash
# Run CUDA kernel tests
make test                             # All CUDA tests

# Run specific new tests
cd tests && go test -tags cuda -run TestBF16 -v
cd tests && go test -tags cuda -run TestCausalConv1d -v
cd tests && go test -tags cuda -run TestConvLayer -v

# Run Go package tests
make test-go                          # All Go tests (no CUDA)

# Expected: All pass. BF16 tests may have higher tolerance (1e-2 vs 1e-4)
```

### Level 4: Integration Test

```bash
# Download model
make download-lfm2-thinking

# Generate golden data
python scripts/generate_golden_lfm2.py

# Run golden comparison test
cd tests && go test -tags cuda -run TestGoldenLFM2 -v

# Expected: Per-layer outputs match HuggingFace within 1e-2 tolerance
```

### Level 5: End-to-End

```bash
# Start inference server
make run-lfm2-thinking

# Test inference
curl -X POST http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"lfm2-1.2b-thinking","messages":[{"role":"user","content":"What is 2+2?"}],"temperature":0.05,"max_tokens":256}'

# Expected: Coherent response with thinking trace
# Verify: <think>...</think> content present in output

# Test backward compatibility with existing model
make run-tinyllama
curl -X POST http://localhost:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"tinyllama","messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'

# Expected: TinyLlama still works unchanged
```

## Final Validation Checklist

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
- [ ] EOS token = 7 (<|im_end|>) terminates generation correctly
- [ ] Memory usage within expectations (~3GB VRAM for LFM2.5-1.2B)

---

## Anti-Patterns to Avoid

- ❌ Don't break Llama/Mistral support — all existing models must work unchanged
- ❌ Don't use CUBLAS_COMPUTE_16BF — it doesn't exist. Always CUBLAS_COMPUTE_32F
- ❌ Don't assume all layers are attention — always check IsConvLayer()
- ❌ Don't allocate separate lm_head when TieEmbeddings=true
- ❌ Don't forget to reset conv state between sequences
- ❌ Don't zero-pad conv state lazily — explicit reset at sequence start
- ❌ Don't use eos_token_id=2 for LFM2 — it's 7 (<|im_end|>)
- ❌ Don't load conv weight as [hidden, 1, kernel] — reshape to [hidden, kernel]
- ❌ Don't apply RoPE before QK LayerNorm — norm first, then RoPE
- ❌ Don't skip updating gpu_stub.go when adding new GPU functions
- ❌ Don't forget config propagation chain (5 structs must stay in sync)

---

## Task Breakdown Reference

See: `docs/tasks/lfm2-support.md` for detailed task breakdown with dependencies and acceptance criteria.

---

**PRP Confidence Score: 8/10**

Rationale: High confidence due to comprehensive external research (HuggingFace ref impl, Dao-AILab causal-conv1d, cuBLAS BF16 docs), complete codebase analysis with exact line numbers, and clear separation of concerns. The 2-point deduction is for: (1) BF16 full pipeline is novel for this codebase and may surface unforeseen integration issues, and (2) the conv1d CUDA kernel implementation needs careful testing against the reference.
