# Brainstorming: E2E Tests com Inferência Real

**Data:** 2026-01-21
**Sessão:** Opção C - Full TDD com CUDA Real
**Objetivo:** Confiança em produção + Detecção de regressões

---

## 1. Análise do Estado Atual

### 1.1 Componentes Existentes

| Componente | Status | Localização |
|------------|--------|-------------|
| **CUDA Kernels** | ✅ Completo | `gpu/cuda/kernels.cu` |
| RMSNorm | ✅ | FP16 otimizado com half2 |
| SiLU | ✅ | Ativação swish |
| Add/Mul | ✅ | Operações elementares |
| RoPE | ✅ | Rotary embeddings |
| Softmax | ✅ | Attention softmax |
| GEMM | ✅ | FP16 e INT8 quantizado |
| Attention | ✅ | Com KV Cache |
| **Layer Forward** | ⚠️ Parcial | `gpu/engine/layer.cu` |
| LayerForward | ✅ | Lógica completa |
| cuda_load_layer_weights | ❌ | Retorna erro "not implemented" |
| **Go Weight Loading** | ✅ Completo | `pkg/model/loader.go` |
| SafeTensors parsing | ✅ | Sharded e single file |
| LoadLayerWeights | ✅ | Todos 9 tensores |
| LoadEmbeddings/LMHead | ✅ | Funcionais |
| MmapLoader | ✅ | Para modelos grandes |
| **Tokenizer** | ✅ Completo | `pkg/model/tokenizer.go` |
| SentencePiece | ✅ | Encode/Decode implementado |
| **Golden Data** | ✅ Existente | `tests/golden/` |
| Kernel tests | ✅ | rmsnorm, silu, gemm |
| Layer 0 weights | ✅ | 9 arquivos .bin |
| Layer 0 intermediários | ✅ | input, output, q/k/v proj |

### 1.2 Lacunas Identificadas

#### Gap 1: MockTokenizer nos Testes E2E
```go
// tests/e2e/inference_test.go - PROBLEMA
func (t *MockTokenizer) Encode(text string) ([]int, error) {
    tokens := make([]int, len(text))
    for i, c := range text {
        tokens[i] = int(c) % t.vocabSize  // ← NÃO É TOKENIZAÇÃO REAL
    }
    return tokens, nil
}
```
**Impacto:** Testes não validam fluxo real de tokenização → bugs em produção.

#### Gap 2: Forward Pass com Dados Mock
```go
// tests/e2e/api_test.go - PROBLEMA
func createTestEngine() *inference.Engine {
    engine.SetTokenizer(NewMockTokenizer())
    // Nenhum peso real carregado
    // Inferência retorna dados mock
}
```
**Impacto:** Nunca testamos computação GPU real → regressões não detectadas.

#### Gap 3: Pesos Nunca Transferidos para GPU
```c
// gpu/engine/layer.cu - PROBLEMA
extern "C" int cuda_load_layer_weights(void** weights, const char* path) {
    fprintf(stderr, "load_layer_weights not implemented...");
    return -1;  // ← SEMPRE FALHA
}
```
**Impacto:** Go carrega pesos mas não consegue enviá-los para CUDA.

#### Gap 4: Conexão Go ↔ CUDA Incompleta
```
Go (loader.go)              CUDA (layer.cu)
     │                            │
     ├─ LoadLayerWeights()        │
     │  → []byte (host)           │
     │                            │
     └─────── ??? ───────────────→ cuda_layer_weights_t (GPU)
                                  │
        LACUNA: Não existe função para
        transferir pesos do Go para
        estrutura CUDA
```

---

## 2. Plano de Implementação - Opção C

### 2.1 Arquitetura Alvo

```
┌─────────────────────────────────────────────────────────────────┐
│                        E2E Test Flow                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│   │   Go Test    │───→│ Real Weights │───→│   CUDA GPU   │     │
│   │ (model_test) │    │  (safetens)  │    │   Forward    │     │
│   └──────────────┘    └──────────────┘    └──────────────┘     │
│          │                   │                   │              │
│          │                   │                   ▼              │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│   │    Real      │───→│   Golden     │───→│   Compare    │     │
│   │  Tokenizer   │    │    Data      │    │  (tolerance) │     │
│   └──────────────┘    └──────────────┘    └──────────────┘     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Tarefas por Fase

#### FASE 1: Bridge Go → CUDA (PRP-05)

| # | Tarefa | Descrição | Complexidade |
|---|--------|-----------|--------------|
| 1.1 | LayerWeightsFromGo() | Função Go que aloca GPU e copia pesos | Média |
| 1.2 | cuda_create_layer_weights_from_host() | Função CUDA que recebe ponteiros host | Média |
| 1.3 | Binding CGO | Conectar Go com CUDA função | Baixa |
| 1.4 | Teste unitário | Carregar layer 0 de tests/golden/ | Baixa |

**Estrutura proposta:**
```go
// gpu/bindings/gpu.go
func CreateLayerWeightsFromHost(
    qProj, kProj, vProj, oProj []byte,  // Attention
    gateProj, upProj, downProj []byte,   // FFN
    attnNorm, ffnNorm []byte,            // Norms
    config *types.LlamaConfig,
) (*LayerWeights, error)
```

```c
// gpu/engine/layer.cu
extern "C" int cuda_create_layer_weights_from_host(
    cuda_layer_weights_t** out,
    void* q_proj, void* k_proj, void* v_proj, void* o_proj,
    void* gate_proj, void* up_proj, void* down_proj,
    void* attn_norm, void* ffn_norm,
    int hidden_size, int intermediate_size,
    int num_heads, int num_kv_heads, int head_dim
);
```

#### FASE 2: Full Model Loading (PRP-06)

| # | Tarefa | Descrição | Complexidade |
|---|--------|-----------|--------------|
| 2.1 | ModelLoader GPU | Carregar todas 32 layers na GPU | Alta |
| 2.2 | Embeddings GPU | Transfer token embeddings | Média |
| 2.3 | LMHead GPU | Transfer output projection | Média |
| 2.4 | Memory management | Gerenciar VRAM 24GB | Alta |

**Considerações RTX 4090 (24GB):**
- Llama-2-7B FP16: ~14GB
- KV Cache 2048 tokens: ~1GB
- Overhead: ~1GB
- Margem: ~8GB ✅

#### FASE 3: Real LayerExecutor (PRP-07)

| # | Tarefa | Descrição | Complexidade |
|---|--------|-----------|--------------|
| 3.1 | CUDALayerExecutor | Implementar interface LayerExecutor | Alta |
| 3.2 | Forward com pesos reais | Usar pesos carregados | Média |
| 3.3 | KV Cache integração | Cache incremental para geração | Alta |
| 3.4 | Multi-token forward | Batch de tokens (prefill) | Alta |

**Interface existente:**
```go
// pkg/inference/types.go
type LayerExecutor interface {
    Forward(input *types.Tensor, cache *KVCache, pos int) (*types.Tensor, error)
    LoadWeights(weights *model.LoadedLayerWeights) error
}
```

#### FASE 4: E2E com Inferência Real (PRP-08)

| # | Tarefa | Descrição | Complexidade |
|---|--------|-----------|--------------|
| 4.1 | Substituir MockTokenizer | Usar SentencePiece real | Baixa |
| 4.2 | Golden para full forward | Gerar output esperado via PyTorch | Média |
| 4.3 | E2E layer test | Input → Layer 0 → Compare golden | Média |
| 4.4 | E2E full model test | Prompt → Completion → Validate | Alta |

**Testes propostos:**
```go
func TestRealLayerForward(t *testing.T) {
    // 1. Load golden weights
    weights := loadGoldenWeights(t, "tests/golden/layer_0_weights/")

    // 2. Create real executor
    executor := NewCUDALayerExecutor(config)
    executor.LoadWeights(weights)

    // 3. Load golden input
    input := loadGoldenTensor(t, "tests/golden/layer_0_input.bin")

    // 4. Forward
    output, err := executor.Forward(input, nil, 0)
    require.NoError(t, err)

    // 5. Compare with golden output
    expected := loadGoldenTensor(t, "tests/golden/layer_0_output.bin")
    assertTensorClose(t, output, expected, 5e-3) // FP16 tolerance
}
```

#### FASE 5: Geração Completa (PRP-09)

| # | Tarefa | Descrição | Complexidade |
|---|--------|-----------|--------------|
| 5.1 | Full inference pipeline | Embeddings → Layers → LMHead → Sampling | Alta |
| 5.2 | Golden para geração | PyTorch reference para prompts fixos | Média |
| 5.3 | Deterministic sampling | Greedy/fixed seed para reproducibilidade | Baixa |
| 5.4 | Response validation | Comparar texto gerado com esperado | Média |

**Golden generation script:**
```python
# scripts/generate_inference_golden.py
def generate_golden_completion(model, tokenizer, prompt):
    tokens = tokenizer.encode(prompt)
    with torch.no_grad():
        output = model.generate(
            torch.tensor([tokens]),
            max_new_tokens=32,
            do_sample=False,  # Greedy for determinism
        )
    response = tokenizer.decode(output[0])
    return tokens, output[0].tolist(), response
```

---

## 3. Dependências entre Tarefas

```
FASE 1: Bridge Go → CUDA
   │
   ├──→ FASE 2: Full Model Loading
   │       │
   │       └──→ FASE 3: Real LayerExecutor
   │               │
   │               ├──→ FASE 4: E2E Layer Tests
   │               │
   │               └──→ FASE 5: Full Generation Tests
   │
   └──→ (Paralelo) Golden Data Generation
```

---

## 4. Critérios de Sucesso

### 4.1 Testes Layer-Level (FASE 4)
- [ ] Layer forward match PyTorch ±0.5% (FP16 tolerance)
- [ ] Intermediários (Q/K/V projections) match
- [ ] KV Cache funciona corretamente
- [ ] RoPE aplicado nas posições corretas

### 4.2 Testes Full Generation (FASE 5)
- [ ] Tokenização idêntica ao HuggingFace
- [ ] Top-1 logits match PyTorch
- [ ] Greedy decode produz texto idêntico
- [ ] Streaming SSE funciona com inferência real

### 4.3 Regressão (Contínuo)
- [ ] CI roda testes com GPU
- [ ] Quebra de tolerância falha o build
- [ ] Tempo de execução monitorado (<30s para layer test)

---

## 5. Riscos e Mitigações

| Risco | Probabilidade | Impacto | Mitigação |
|-------|---------------|---------|-----------|
| FP16 precision divergence | Média | Alto | Tolerância ajustável, documentar diferenças |
| VRAM overflow | Baixa | Alto | Carregar layers sob demanda |
| Golden data staleness | Média | Médio | Script automatizado + hash check |
| CI sem GPU | Alta | Alto | Testes marcados com `//go:build cuda` |

---

## 6. Estimativa de Esforço

| Fase | PRPs | Complexidade | Dependências |
|------|------|--------------|--------------|
| 1. Bridge Go→CUDA | PRP-05 | Média | Nenhuma |
| 2. Full Model Load | PRP-06 | Alta | PRP-05 |
| 3. LayerExecutor | PRP-07 | Alta | PRP-05 |
| 4. E2E Layer Tests | PRP-08 | Média | PRP-07 |
| 5. Generation Tests | PRP-09 | Alta | PRP-06, PRP-07, PRP-08 |

---

## 7. Próximos Passos

1. **Criar PRP-05:** Bridge Go → CUDA para transfer de pesos
2. **Gerar golden data adicional:** Precisa de output do full forward pass
3. **Implementar em TDD:** RED (testes falham) → GREEN (implementar) → REFACTOR

---

## 8. Decisão

**Aprovado:** Opção C - Full TDD com CUDA Real

**Justificativa:**
- Kernels já existem (80% do trabalho CUDA feito)
- Golden data disponível para validação
- RTX 4090 comporta Llama-2-7B com margem
- Investimento alto, mas ROI em confiança de produção

---

*Documento gerado durante sessão de brainstorming em 2026-01-21*
