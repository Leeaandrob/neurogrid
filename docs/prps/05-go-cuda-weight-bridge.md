# PRP-05: Go → CUDA Inference Bridge (Full Pipeline)

## Overview

**Goal:** Enable end-to-end inference from Chat Completions API to coherent text output using real model weights and CUDA execution.

**Problem Statement:**
O sistema atual tem todos os componentes, mas não estão conectados:
- Go carrega pesos de SafeTensors com sucesso (`pkg/model/loader.go`)
- CUDA layer forward existe (`gpu/engine/layer.cu`) e funciona
- Tokenizer SentencePiece funciona (`pkg/model/sentencepiece.go`)
- API HTTP/SSE funciona (`api/server.go`)

**Gaps Críticos Identificados:**
1. `LayerExecutor` interface não tem implementação
2. `applyLMHead()` retorna zeros/mock em vez de fazer matmul real
3. `embedToken()` retorna zeros quando embeddings não estão na GPU
4. Pesos carregados em Go nunca são transferidos para GPU
5. Não há conexão entre `WeightLoader` e `LayerForward` CUDA

**Dependencies:** PRPs 01-04 (completed)

---

## Arquitetura Atual vs Desejada

### Fluxo Atual (QUEBRADO)
```
API Request → Tokenizer → Engine.Generate()
                              │
                              ├─ embedToken() → ZEROS
                              ├─ forwardLayer() → layerExecutor==nil → RETORNA INPUT
                              └─ applyLMHead() → TOKEN 42 SEMPRE
                                        │
                                        ▼
                              Garbage Output ("<0x27><0x27>...")
```

### Fluxo Desejado (FUNCIONAL)
```
API Request → SentencePiece → Engine.Generate()
                                    │
                                    ├─ embedToken() → GPU Embedding Lookup
                                    │
                                    ├─ forwardLayer() → CUDALayerExecutor
                                    │       │
                                    │       └─ cuda_layer_forward() → RMSNorm → QKV → Attention → FFN
                                    │
                                    └─ applyLMHead() → GPU MatMul → Logits
                                              │
                                              ▼
                                    Sample → Decode → Coherent Text
```

---

## Acceptance Criteria

### AC1: CUDALayerExecutor Implementation
- [ ] Struct `CUDALayerExecutor` implementa interface `LayerExecutor`
- [ ] Método `Forward()` chama `cuda_layer_forward()` via CGO
- [ ] Mantém pesos de todas as camadas em GPU memory
- [ ] Gerencia lifecycle de memória (alloc/free)

### AC2: GPU Weight Bridge
- [ ] Função `CreateLayerWeightsFromHost()` transfere pesos Go → GPU
- [ ] Suporta formato FP16 (SafeTensors nativo)
- [ ] Conversão INT8 quantizado (opcional, para VRAM limitada)
- [ ] Tratamento de erro com cleanup em caso de falha

### AC3: GPU Embedding Lookup
- [ ] Embeddings carregados em GPU memory
- [ ] `embedToken()` faz lookup na GPU (não cópia para CPU)
- [ ] Retorna hidden state em GPU memory

### AC4: GPU LM Head
- [ ] LM Head carregado em GPU memory
- [ ] `applyLMHead()` executa matmul real: `logits = hidden @ lm_head`
- [ ] Retorna logits em CPU para sampling

### AC5: End-to-End Inference Test
- [ ] Teste com TinyLlama (1.1B params)
- [ ] Input: "Hello, my name is"
- [ ] Output: Texto coerente (não garbage)
- [ ] Validação: Output contém palavras reais do vocabulário

### AC6: Chat Completions API Functional
- [ ] `POST /v1/chat/completions` retorna resposta coerente
- [ ] Streaming SSE funciona com tokens reais
- [ ] Latência primeiro token < 500ms (TinyLlama)

---

## Technical Design

### 1. CUDALayerExecutor

```go
// pkg/inference/cuda_executor.go

// CUDALayerExecutor implements LayerExecutor using CUDA kernels.
type CUDALayerExecutor struct {
    layerWeights map[int]*bindings.LayerWeights // layerID -> GPU weights
    config       *types.LlamaConfig
    deviceID     int
    mu           sync.RWMutex
}

// NewCUDALayerExecutor creates an executor with weights loaded to GPU.
func NewCUDALayerExecutor(config *types.LlamaConfig, deviceID int) *CUDALayerExecutor {
    return &CUDALayerExecutor{
        layerWeights: make(map[int]*bindings.LayerWeights),
        config:       config,
        deviceID:     deviceID,
    }
}

// LoadLayer loads a single layer's weights to GPU.
func (e *CUDALayerExecutor) LoadLayer(layerID int, weights *model.TransformerLayerWeights) error {
    e.mu.Lock()
    defer e.mu.Unlock()

    gpuWeights, err := bindings.CreateLayerWeightsFromHost(
        weights.QProj, weights.KProj, weights.VProj, weights.OProj,
        weights.GateProj, weights.UpProj, weights.DownProj,
        weights.AttnNorm, weights.FFNNorm,
        e.config,
    )
    if err != nil {
        return fmt.Errorf("failed to upload layer %d to GPU: %w", layerID, err)
    }

    e.layerWeights[layerID] = gpuWeights
    return nil
}

// Forward executes layer forward pass on GPU.
func (e *CUDALayerExecutor) Forward(
    ctx context.Context,
    layerID int,
    hidden []byte,
    position int,
) (output []byte, k []byte, v []byte, err error) {
    e.mu.RLock()
    weights, ok := e.layerWeights[layerID]
    e.mu.RUnlock()

    if !ok {
        return nil, nil, nil, fmt.Errorf("layer %d not loaded", layerID)
    }

    // Allocate output buffer
    outputSize := len(hidden)
    output = make([]byte, outputSize)

    // KV sizes
    kvSize := e.config.NumKVHeads * e.config.HeadDim * 2 // FP16
    k = make([]byte, kvSize)
    v = make([]byte, kvSize)

    // Execute on GPU
    err = bindings.LayerForwardWithKV(
        output, hidden, weights,
        []int32{int32(position)},
        k, v,
        e.config,
    )
    if err != nil {
        return nil, nil, nil, fmt.Errorf("CUDA layer forward failed: %w", err)
    }

    return output, k, v, nil
}

// Close frees all GPU resources.
func (e *CUDALayerExecutor) Close() error {
    e.mu.Lock()
    defer e.mu.Unlock()

    for _, weights := range e.layerWeights {
        bindings.FreeLayerWeights(weights)
    }
    e.layerWeights = nil
    return nil
}
```

### 2. GPU Embeddings

```go
// pkg/inference/gpu_embeddings.go

// GPUEmbeddings holds token embeddings in GPU memory.
type GPUEmbeddings struct {
    ptr        unsafe.Pointer // GPU memory
    vocabSize  int
    hiddenSize int
}

// NewGPUEmbeddings uploads embeddings to GPU.
func NewGPUEmbeddings(data []byte, vocabSize, hiddenSize int) (*GPUEmbeddings, error) {
    ptr, err := bindings.AllocAndCopy(data)
    if err != nil {
        return nil, err
    }

    return &GPUEmbeddings{
        ptr:        ptr,
        vocabSize:  vocabSize,
        hiddenSize: hiddenSize,
    }, nil
}

// Lookup returns embedding for a token (stays on GPU).
func (e *GPUEmbeddings) Lookup(tokenID int) (unsafe.Pointer, error) {
    if tokenID < 0 || tokenID >= e.vocabSize {
        return nil, fmt.Errorf("token %d out of range [0, %d)", tokenID, e.vocabSize)
    }

    // Return pointer offset into GPU memory
    bytesPerEmbed := e.hiddenSize * 2 // FP16
    offset := uintptr(tokenID * bytesPerEmbed)
    return unsafe.Pointer(uintptr(e.ptr) + offset), nil
}

// LookupToHost copies embedding to host memory.
func (e *GPUEmbeddings) LookupToHost(tokenID int) ([]byte, error) {
    ptr, err := e.Lookup(tokenID)
    if err != nil {
        return nil, err
    }

    size := e.hiddenSize * 2
    data := make([]byte, size)
    if err := bindings.CopyToHostRaw(data, ptr, uint64(size)); err != nil {
        return nil, err
    }
    return data, nil
}
```

### 3. GPU LM Head

```go
// pkg/inference/gpu_lmhead.go

// GPULMHead holds the output projection in GPU memory.
type GPULMHead struct {
    ptr        unsafe.Pointer // [hiddenSize, vocabSize] in FP16
    hiddenSize int
    vocabSize  int
}

// NewGPULMHead uploads LM head to GPU.
func NewGPULMHead(data []byte, hiddenSize, vocabSize int) (*GPULMHead, error) {
    ptr, err := bindings.AllocAndCopy(data)
    if err != nil {
        return nil, err
    }

    return &GPULMHead{
        ptr:        ptr,
        hiddenSize: hiddenSize,
        vocabSize:  vocabSize,
    }, nil
}

// Forward computes logits = hidden @ lm_head
// hidden: [1, hiddenSize] in FP16 (GPU pointer)
// returns: [vocabSize] in FP32 (CPU memory for sampling)
func (h *GPULMHead) Forward(hidden unsafe.Pointer) ([]float32, error) {
    // Allocate GPU output buffer
    logitsGPU, err := bindings.Malloc(uint64(h.vocabSize * 4)) // FP32
    if err != nil {
        return nil, err
    }
    defer bindings.Free(logitsGPU)

    // Execute GEMM: [1, hidden] x [hidden, vocab] = [1, vocab]
    err = bindings.GEMMFP16ToFP32(
        logitsGPU,  // C: output
        hidden,     // A: input hidden state
        h.ptr,      // B: LM head weights
        1,          // M: batch size
        h.hiddenSize, // K: hidden dimension
        h.vocabSize,  // N: vocab size
    )
    if err != nil {
        return nil, fmt.Errorf("LM head GEMM failed: %w", err)
    }

    // Copy logits to CPU for sampling
    logits := make([]float32, h.vocabSize)
    if err := bindings.CopyFP32ToHost(logits, logitsGPU); err != nil {
        return nil, err
    }

    return logits, nil
}
```

### 4. Updated Engine Integration

```go
// pkg/inference/engine.go - Modifications

type Engine struct {
    // ... existing fields ...

    // GPU components (new)
    gpuEmbeddings   *GPUEmbeddings
    gpuLMHead       *GPULMHead
    cudaExecutor    *CUDALayerExecutor
    useGPU          bool
}

// InitializeGPU sets up GPU inference pipeline.
func (e *Engine) InitializeGPU(loader *model.WeightLoader, deviceID int) error {
    // 1. Load embeddings to GPU
    embData, _, err := loader.LoadEmbeddings()
    if err != nil {
        return fmt.Errorf("load embeddings: %w", err)
    }
    e.gpuEmbeddings, err = NewGPUEmbeddings(embData, e.config.VocabSize, e.config.HiddenSize)
    if err != nil {
        return fmt.Errorf("GPU embeddings: %w", err)
    }

    // 2. Load LM head to GPU
    lmData, _, err := loader.LoadLMHead()
    if err != nil {
        return fmt.Errorf("load lm head: %w", err)
    }
    e.gpuLMHead, err = NewGPULMHead(lmData, e.config.HiddenSize, e.config.VocabSize)
    if err != nil {
        return fmt.Errorf("GPU LM head: %w", err)
    }

    // 3. Create CUDA executor
    e.cudaExecutor = NewCUDALayerExecutor(e.config, deviceID)

    // 4. Load all layers
    for layerID := 0; layerID < e.config.NumLayers; layerID++ {
        weights, err := loader.LoadLayerWeights(layerID)
        if err != nil {
            return fmt.Errorf("load layer %d: %w", layerID, err)
        }
        if err := e.cudaExecutor.LoadLayer(layerID, weights); err != nil {
            return fmt.Errorf("GPU layer %d: %w", layerID, err)
        }
        log.Printf("Loaded layer %d/%d to GPU", layerID+1, e.config.NumLayers)
    }

    e.useGPU = true
    e.layerExecutor = e.cudaExecutor
    return nil
}

// embedToken - Updated to use GPU embeddings
func (e *Engine) embedToken(token int) ([]byte, error) {
    if e.useGPU && e.gpuEmbeddings != nil {
        return e.gpuEmbeddings.LookupToHost(token)
    }
    // Fallback to CPU (existing code)
    // ...
}

// applyLMHead - Updated to use GPU
func (e *Engine) applyLMHead(hidden []byte) ([]float32, error) {
    if e.useGPU && e.gpuLMHead != nil {
        // Upload hidden to GPU
        hiddenGPU, err := bindings.AllocAndCopy(hidden)
        if err != nil {
            return nil, err
        }
        defer bindings.Free(hiddenGPU)

        return e.gpuLMHead.Forward(hiddenGPU)
    }
    // Fallback (for testing only - should not happen in production)
    return nil, fmt.Errorf("GPU not initialized for LM head")
}
```

### 5. CUDA Bindings Additions

```go
// gpu/bindings/gpu.go - New functions needed

// CreateLayerWeightsFromHost transfers layer weights from Go memory to GPU.
func CreateLayerWeightsFromHost(
    qProj, kProj, vProj, oProj []byte,
    gateProj, upProj, downProj []byte,
    attnNorm, ffnNorm []byte,
    config *types.LlamaConfig,
) (*LayerWeights, error)

// LayerForwardWithKV executes layer forward and returns K/V for caching.
func LayerForwardWithKV(
    output, input []byte,
    weights *LayerWeights,
    positions []int32,
    kOut, vOut []byte, // K/V outputs for cache
    config *types.LlamaConfig,
) error

// AllocAndCopy allocates GPU memory and copies host data.
func AllocAndCopy(data []byte) (unsafe.Pointer, error)

// GEMMFP16ToFP32 performs FP16 GEMM with FP32 output (for logits).
func GEMMFP16ToFP32(c, a, b unsafe.Pointer, M, K, N int) error

// CopyFP32ToHost copies FP32 data from GPU to host slice.
func CopyFP32ToHost(dst []float32, src unsafe.Pointer) error
```

---

## Implementation Tasks

### Phase 1: GPU Weight Bridge (3-4 tasks)

**Task 1.1: CUDA Weight Upload Function**
- File: `gpu/engine/layer.cu`
- Implement `cuda_create_layer_weights_from_host()`
- Copy host pointers to device memory
- Return populated `LayerWeights` struct

**Task 1.2: Go Binding for Weight Upload**
- File: `gpu/bindings/weights.go`
- CGO wrapper for CUDA function
- Handle unsafe.Pointer conversion
- Error handling with cleanup

**Task 1.3: CUDALayerExecutor Implementation**
- File: `pkg/inference/cuda_executor.go`
- Implement `LayerExecutor` interface
- Manage layer weight lifecycle
- Thread-safe weight access

### Phase 2: GPU Embeddings & LM Head (2-3 tasks)

**Task 2.1: GPUEmbeddings Implementation**
- File: `pkg/inference/gpu_embeddings.go`
- Upload embeddings to GPU
- Token lookup on GPU
- Copy result to host

**Task 2.2: GPULMHead Implementation**
- File: `pkg/inference/gpu_lmhead.go`
- Upload LM head weights
- GEMM for logits computation
- FP16 input → FP32 output

**Task 2.3: CUDA GEMM FP16→FP32**
- File: `gpu/cuda/matmul.cu`
- Add variant that outputs FP32
- Use cuBLAS or custom kernel

### Phase 3: Engine Integration (2 tasks)

**Task 3.1: Engine GPU Initialization**
- File: `pkg/inference/engine.go`
- Add `InitializeGPU()` method
- Load all weights to GPU
- Set `useGPU` flag

**Task 3.2: Coordinator Integration**
- File: `cmd/neurogrid/main.go`
- Call `engine.InitializeGPU()` after loading
- Proper error handling
- Log GPU memory usage

### Phase 4: Validation (2 tasks)

**Task 4.1: E2E Inference Test with TinyLlama**
- File: `tests/e2e/real_inference_test.go`
- Load TinyLlama model
- Run inference
- Validate coherent output

**Task 4.2: API Integration Test**
- File: `tests/e2e/api_real_test.go`
- Start server with real model
- Call `/v1/chat/completions`
- Validate response quality

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Full Inference Pipeline                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐                                                            │
│  │ SafeTensors  │                                                            │
│  │    Files     │                                                            │
│  └──────┬───────┘                                                            │
│         │ LoadLayerWeights()                                                 │
│         ▼                                                                    │
│  ┌──────────────┐    CreateLayerWeightsFromHost()    ┌─────────────────┐    │
│  │  Go Memory   │ ─────────────────────────────────► │   GPU Memory    │    │
│  │  ([]byte)    │          cudaMemcpyHostToDevice    │ (LayerWeights)  │    │
│  └──────────────┘                                    └────────┬────────┘    │
│                                                               │              │
│  ┌──────────────┐                                             │              │
│  │  Embeddings  │ ──► GPUEmbeddings ──────────────────────────┤              │
│  └──────────────┘                                             │              │
│                                                               │              │
│  ┌──────────────┐                                             │              │
│  │   LM Head    │ ──► GPULMHead ──────────────────────────────┤              │
│  └──────────────┘                                             │              │
│                                                               │              │
│  ════════════════════════════════════════════════════════════════════════   │
│                              INFERENCE TIME                                  │
│  ════════════════════════════════════════════════════════════════════════   │
│                                                               │              │
│  ┌──────────────┐                                             │              │
│  │ "Hello, my   │ ──► SentencePiece.Encode() ──► [1, 15043, ...]             │
│  │  name is"    │                                             │              │
│  └──────────────┘                                             │              │
│         │                                                     │              │
│         │ for each token:                                     │              │
│         ▼                                                     ▼              │
│  ┌──────────────┐    GPUEmbeddings.Lookup()     ┌─────────────────┐         │
│  │  Token ID    │ ─────────────────────────────►│  Hidden State   │         │
│  └──────────────┘                               │    (GPU FP16)   │         │
│                                                 └────────┬────────┘         │
│                                                          │                   │
│                              for layer in 0..N-1:        │                   │
│                                         ┌────────────────┘                   │
│                                         │                                    │
│                                         ▼                                    │
│                              ┌─────────────────┐                             │
│                              │ cuda_layer_     │                             │
│                              │ forward()       │                             │
│                              │  • RMSNorm      │                             │
│                              │  • QKV Proj     │                             │
│                              │  • RoPE         │                             │
│                              │  • Attention    │                             │
│                              │  • FFN          │                             │
│                              └────────┬────────┘                             │
│                                       │                                      │
│                                       ▼                                      │
│                              ┌─────────────────┐                             │
│                              │  Hidden State   │                             │
│                              │  (next layer)   │                             │
│                              └────────┬────────┘                             │
│                                       │                                      │
│                              (loop until all layers done)                    │
│                                       │                                      │
│                                       ▼                                      │
│                              ┌─────────────────┐                             │
│                              │ GPULMHead.      │                             │
│                              │ Forward()       │                             │
│                              │                 │                             │
│                              │ hidden @ lm_head│                             │
│                              └────────┬────────┘                             │
│                                       │                                      │
│                                       ▼                                      │
│                              ┌─────────────────┐                             │
│                              │    Logits       │                             │
│                              │  [vocab_size]   │                             │
│                              │    (CPU FP32)   │                             │
│                              └────────┬────────┘                             │
│                                       │                                      │
│                                       ▼                                      │
│                              ┌─────────────────┐                             │
│                              │ Sampler.Sample()│                             │
│                              │  temperature    │                             │
│                              │  top_p          │                             │
│                              └────────┬────────┘                             │
│                                       │                                      │
│                                       ▼                                      │
│                              ┌─────────────────┐                             │
│                              │  Next Token ID  │                             │
│                              └────────┬────────┘                             │
│                                       │                                      │
│                              (loop until EOS or max_tokens)                  │
│                                       │                                      │
│                                       ▼                                      │
│                              ┌─────────────────┐                             │
│                              │ SentencePiece.  │                             │
│                              │ Decode()        │                             │
│                              └────────┬────────┘                             │
│                                       │                                      │
│                                       ▼                                      │
│                              ┌─────────────────┐                             │
│                              │ "Sarah, and I   │                             │
│                              │  am a software  │                             │
│                              │  engineer..."   │                             │
│                              └─────────────────┘                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Validation Checklist

### Build & Compile
- [ ] `make build-coordinator` compila com `-tags cuda`
- [ ] Sem warnings de CGO
- [ ] Linkagem com libcudart, libcublas

### Unit Tests (GPU Required)
- [ ] `TestCreateLayerWeightsFromHost` - upload funciona
- [ ] `TestGPUEmbeddingsLookup` - lookup retorna dados corretos
- [ ] `TestGPULMHeadForward` - GEMM produz logits válidos
- [ ] `TestCUDALayerExecutorForward` - layer forward funciona

### E2E Tests (TinyLlama)
- [ ] `TestRealInference_TinyLlama` - output é texto coerente
- [ ] `TestChatCompletions_RealModel` - API retorna resposta válida
- [ ] `TestStreaming_RealTokens` - SSE envia tokens reais

### Performance
- [ ] Time-to-first-token < 500ms (TinyLlama)
- [ ] Throughput > 10 tokens/sec (TinyLlama, RTX 4090)
- [ ] GPU memory usage matches expected

### Memory Safety
- [ ] `cuda-memcheck` sem leaks
- [ ] Proper cleanup em erro paths
- [ ] `defer` para todos os recursos GPU

---

## Test Cases Específicos

### Test 1: Coherent Output Validation
```go
func TestRealInference_CoherentOutput(t *testing.T) {
    engine := setupEngineWithTinyLlama(t)

    resp, err := engine.Generate(ctx, &inference.GenerateRequest{
        Prompt:      "The capital of France is",
        MaxTokens:   10,
        Temperature: 0.0, // Greedy for determinism
    })
    require.NoError(t, err)

    // Output should contain "Paris" or related words
    output := strings.ToLower(resp.Text)
    assert.True(t,
        strings.Contains(output, "paris") ||
        strings.Contains(output, "city") ||
        strings.Contains(output, "french"),
        "Expected coherent output about Paris, got: %s", resp.Text,
    )
}
```

### Test 2: Not Garbage
```go
func TestRealInference_NotGarbage(t *testing.T) {
    engine := setupEngineWithTinyLlama(t)

    resp, err := engine.Generate(ctx, &inference.GenerateRequest{
        Prompt:    "Hello, how are you?",
        MaxTokens: 20,
    })
    require.NoError(t, err)

    // Should not contain hex escape sequences
    assert.NotContains(t, resp.Text, "<0x")

    // Should contain at least some English words
    words := strings.Fields(resp.Text)
    englishWords := 0
    for _, w := range words {
        if isEnglishWord(w) {
            englishWords++
        }
    }
    assert.Greater(t, englishWords, len(words)/2,
        "Expected mostly English words, got: %s", resp.Text)
}
```

### Test 3: API Integration
```go
func TestChatCompletions_RealModel(t *testing.T) {
    server := startServerWithTinyLlama(t)
    defer server.Shutdown(ctx)

    req := ChatCompletionRequest{
        Model: "tinyllama",
        Messages: []Message{
            {Role: "user", Content: "What is 2+2?"},
        },
        MaxTokens: 20,
    }

    resp, err := sendChatRequest(server.URL, req)
    require.NoError(t, err)

    content := resp.Choices[0].Message.Content

    // Should mention "4" or "four"
    assert.True(t,
        strings.Contains(content, "4") ||
        strings.Contains(strings.ToLower(content), "four"),
        "Expected answer about 4, got: %s", content,
    )
}
```

---

## Success Metrics

| Metric | Target | Validation |
|--------|--------|------------|
| Output coherence | >80% English words | Word frequency check |
| No garbage | 0 hex escapes | String contains `<0x` |
| API latency (TTFT) | <500ms | Timer |
| Throughput | >10 tok/s | Counter |
| Memory leaks | 0 | cuda-memcheck |
| Test pass rate | 100% | CI |

---

## Out of Scope

- Multi-GPU tensor parallelism
- INT4/INT8 quantization
- Flash Attention optimization
- Batch inference (batch_size > 1)
- KV cache persistence across requests
- Model sharding across nodes

---

## References

- `gpu/engine/layer.cu` - CUDA layer implementation
- `gpu/bindings/gpu.go` - Existing CGO bindings
- `pkg/model/weight_loader.go` - SafeTensors loading
- `pkg/inference/engine.go` - Inference engine
- `api/server.go` - HTTP server
- `models/tinyllama/` - Test model
