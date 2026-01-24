# Data Flow Diagram: TransferableConfig Lifecycle

## Overview

This diagram shows how model configuration data flows through the system, from the coordinator's loaded model to the worker's GPU memory allocation. The TransferableConfig struct serves as the wire format for P2P transfer.

## Diagram

```mermaid
flowchart TB
    subgraph Coordinator["Coordinator Node"]
        A[types.LlamaConfig] --> B[FromLlamaConfig]
        B --> C[TransferableConfig]
        C --> D[json.Marshal]
        D --> E[JSON bytes ~450B]
    end

    subgraph P2P["P2P Protocol Layer"]
        E --> F[SendModelConfig]
        F --> G["sendExtendedMessage<br/>MsgType=0x07"]
        G --> H["Network Stream<br/>/neurogrid/tensor/1.0.0"]
    end

    subgraph Worker["Worker Node"]
        H --> I[handleExtendedMessage]
        I --> J[modelConfigHandler]
        J --> K[DeserializeConfig]
        K --> L[json.Unmarshal]
        L --> M[TransferableConfig]
        M --> N[ToLlamaConfig]
        N --> O[types.LlamaConfig]
        O --> P[worker.modelConfig]
        P --> Q[GPU Memory Allocation]
    end

    style C fill:#e1f5fe
    style M fill:#e1f5fe
    style E fill:#fff3e0
    style H fill:#f3e5f5
```

## Complete Initialization Flow

```mermaid
flowchart TB
    subgraph Coordinator["Coordinator Node (pkg/inference/coordinator.go)"]
        A1[Peer Connected] --> A2{configSent?}
        A2 -->|No| A3[sendConfigToPeer]
        A2 -->|Yes| A4[Skip to weights]
        A3 --> A5[SerializeConfig]
        A5 --> A6[protocol.SendModelConfig]
        A6 --> A7[configSent = true]
        A7 --> A8[Sleep 100ms]
        A8 --> A4
        A4 --> A9[distributeWeightsToPeer]
    end

    subgraph Network["P2P Network"]
        A6 --> N1[MsgTypeModelConfig 0x07]
        A9 --> N2[MsgTypeWeights 0x03 chunked]
    end

    subgraph Worker["Worker Node (cmd/worker/main.go)"]
        N1 --> W1[handleModelConfig]
        W1 --> W2[DeserializeConfig]
        W2 --> W3[worker.modelConfig = config]
        N2 --> W4[handleWeights]
        W4 --> W5{modelConfig != nil?}
        W5 -->|Yes| W6[CreateLayerWeightsFromHost]
        W5 -->|No| W7[Log error - skip GPU]
        W6 --> W8[gpuWeights map]
        W8 --> W9[Initialize KV cache]
        W9 --> W10[weightsReady = true]
    end
```

## Data Flow Steps

| Step | Input | Process | Output |
|------|-------|---------|--------|
| 1 | LlamaConfig | FromLlamaConfig() | TransferableConfig |
| 2 | TransferableConfig | json.Marshal() | JSON bytes |
| 3 | JSON bytes | SendModelConfig() | P2P stream |
| 4 | P2P stream | handleExtendedMessage() | Raw bytes |
| 5 | Raw bytes | json.Unmarshal() | TransferableConfig |
| 6 | TransferableConfig | ToLlamaConfig() | LlamaConfig |
| 7 | LlamaConfig | GPU init | Allocated buffers |

## Data Transformations

### Coordinator Side (Serialization)

```go
// Input: types.LlamaConfig from model loader
config := &types.LlamaConfig{
    HiddenSize:       4096,
    IntermediateSize: 14336,
    NumLayers:        32,
    // ...
}

// Transform to wire format
tc := inference.FromLlamaConfig(config, "mistral-7b")

// Output: JSON bytes
// {"model_name":"mistral-7b","hidden_size":4096,...}
data, _ := json.Marshal(tc)
```

### Wire Format

```
Extended Header (25 bytes):
+--------+----------+--------+------------+----------+
| MsgType| LayerID  | SeqID  | RequestID  | DataLen  |
| 0x07   | 0        | 0      | 0          | ~450     |
+--------+----------+--------+------------+----------+
| 1B     | 4B       | 8B     | 8B         | 4B       |
+--------+----------+--------+------------+----------+

JSON Payload (~450 bytes):
{
  "model_name": "mistral-7b-instruct",
  "hidden_size": 4096,
  "intermediate_size": 14336,
  "num_layers": 32,
  "num_heads": 32,
  "num_kv_heads": 8,
  "head_dim": 128,
  "vocab_size": 32000,
  "max_seq_len": 4096,
  "rms_norm_eps": 0.000001
}
```

### Worker Side (Deserialization)

```go
// Input: JSON bytes from P2P
// {"model_name":"mistral-7b","hidden_size":4096,...}

// Transform from wire format
config, modelName, _ := inference.DeserializeConfig(data)

// Output: types.LlamaConfig for GPU operations
// Used by handleWeights() for:
// - layer_weights allocation
// - KV cache sizing
// - attention buffer allocation
```

## State Diagram

```mermaid
stateDiagram-v2
    [*] --> WaitingForConfig: Worker starts (no --model)
    WaitingForConfig --> ConfigReceived: MsgTypeModelConfig
    ConfigReceived --> WaitingForWeights: DeserializeConfig OK
    WaitingForWeights --> LoadingWeights: MsgTypeWeights
    LoadingWeights --> LoadingWeights: More chunks
    LoadingWeights --> Ready: All layers loaded
    Ready --> [*]

    ConfigReceived --> WaitingForConfig: DeserializeConfig error
    LoadingWeights --> Error: GPU upload failed
```

## Coordinator sendConfigToPeer Flow

```mermaid
flowchart TD
    A[sendConfigToPeer called] --> B{configSent for peer?}
    B -->|Yes| C[Return true - already sent]
    B -->|No| D{config nil?}
    D -->|Yes| C
    D -->|No| E[SerializeConfig]
    E --> F{Error?}
    F -->|Yes| G[Log error, return false]
    F -->|No| H[SendModelConfig]
    H --> I{Error?}
    I -->|Yes| J[Log error, return false]
    I -->|No| K[configSent = true]
    K --> L[Sleep 100ms]
    L --> M[Return true]
```

## Weight Handling Data Flow

```mermaid
flowchart TD
    A[handleWeights called] --> B{chunkBuffer exists?}
    B -->|No| C[Create new chunkBuffer]
    B -->|Yes| D[Get existing buffer]
    C --> D
    D --> E[Store chunk at index]
    E --> F{All chunks?}
    F -->|No| G[Return - wait for more]
    F -->|Yes| H[Concatenate chunks]
    H --> I[DeserializeLayerWeights]
    I --> J[Store in layerWeights map]
    J --> K{modelConfig set?}
    K -->|No| L[Log warning - skip GPU upload]
    K -->|Yes| M[CreateLayerWeightsFromHost]
    M --> N[Store in gpuWeights]
    N --> O[Create KV cache]
    O --> P[Update layer range]
    P --> Q[Delete chunk buffer]
    Q --> R[weightsReady = true]
    R --> S[SendWeightsAck]
```

## Error Handling

| Error | Location | Cause | Recovery |
|-------|----------|-------|----------|
| ErrNilConfig | Coordinator | Nil LlamaConfig passed | Return error, don't send |
| ErrEmptyModelName | Coordinator | Empty model name | Return error, don't send |
| ErrEmptyData | Worker | Zero-length payload | Log error, ignore message |
| ErrNullConfig | Worker | JSON literal "null" | Log error, ignore message |
| json.SyntaxError | Worker | Malformed JSON | Log error, ignore message |
| modelConfig nil | Worker handleWeights | Config not yet received | Log warning, skip GPU upload |

## Invariants

1. **Config size < 1KB**: TransferableConfig JSON never exceeds 1KB (single message)
2. **All fields required**: No optional fields in TransferableConfig
3. **Lossless round-trip**: FromLlamaConfig -> ToLlamaConfig preserves all values
4. **Config precedes weights**: modelConfig must be set before handleWeights() can upload to GPU
5. **100ms delay**: Coordinator sleeps 100ms after config to ensure worker processes it

## Files

| File | Responsibility |
|------|----------------|
| `pkg/types/config.go` | LlamaConfig definition |
| `pkg/inference/config_transfer.go` | TransferableConfig + serialization |
| `pkg/inference/coordinator.go` | sendConfigToPeer, configSent tracking |
| `p2p/protocol.go` | Wire format and message routing |
| `cmd/worker/main.go` | handleModelConfig, handleWeights |

## Related Documentation

- [ADR-006: Config Transfer Protocol](../decisions/ADR-006-config-transfer-protocol.md)
- [Sequence Diagram: Config Transfer](sequence-config-transfer.md)
- [Sequence Diagram: Worker Modes](sequence-worker-modes.md)
- [C4 Component Diagram](c4-component-distributed-inference.md)

---

*Updated 2025-01-24 - Added Phase 3-5 implementation details (coordinator sendConfigToPeer, worker handleWeights flow)*
